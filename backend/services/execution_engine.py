from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import pandas as pd

from backend.services.dashboard_engine import build_dashboard_output
from backend.services.dag_execution import build_execution_graph
from backend.services.excel_engine import run_excel_analysis
from backend.services.intelligent_optimizer import select_best_plan
from backend.services.optimizer import optimize_plan as optimize_single_plan
from backend.services.performance_tracker import (
    build_optimization_summary,
    estimate_cost,
    get_tool_performance_memory,
    record_tool_performance,
    track_performance,
)
from backend.services.sql_engine import run_sql_analysis


TOOL_SQL = "SQL"
TOOL_EXCEL = "EXCEL"
TOOL_PYTHON = "PYTHON"
TOOL_BI = "BI"


def _result_to_dataframe(result: Any) -> pd.DataFrame | None:
    if isinstance(result, pd.DataFrame):
        return result.copy()
    if isinstance(result, list) and result and all(isinstance(item, dict) for item in result):
        return pd.DataFrame(result)
    if isinstance(result, dict):
        for key in ("result", "table", "rows"):
            candidate = result.get(key)
            if isinstance(candidate, pd.DataFrame):
                return candidate.copy()
            if isinstance(candidate, list) and candidate and all(isinstance(item, dict) for item in candidate):
                return pd.DataFrame(candidate)
        scalar_payload = {
            key: value
            for key, value in result.items()
            if not isinstance(value, (dict, list, tuple, set, pd.DataFrame))
        }
        if scalar_payload:
            return pd.DataFrame([scalar_payload])
    return None


def _render_sql_plan(sql_plan: str | None) -> str:
    rendered_plan = str(sql_plan or "").strip() or "SELECT * FROM df"
    return f"-- SQL simulation\n{rendered_plan}"


def _render_excel_logic(excel_analysis: dict[str, Any] | None) -> str:
    payload = dict(excel_analysis or {})
    return "# Excel simulation\nexcel_logic = " + json.dumps(
        {
            "pivot_table": payload.get("pivot_table") or {},
            "aggregations": payload.get("aggregations") or {},
            "summary": payload.get("summary") or {},
        },
        indent=2,
        default=str,
    )


def _render_dashboard(dashboard: dict[str, Any] | None) -> str:
    return "# BI dashboard simulation\ndashboard = " + json.dumps(
        dict(dashboard or {"charts": [], "kpis": []}),
        indent=2,
        default=str,
    )


def _normalize_dependencies(value: Any) -> list[int]:
    dependencies: list[int] = []
    for item in list(value or []):
        try:
            dependency = int(item)
        except (TypeError, ValueError):
            continue
        if dependency > 0 and dependency not in dependencies:
            dependencies.append(dependency)
    return dependencies


def _basic_normalize_plan(plan: list[dict[str, Any]] | None, *, df: pd.DataFrame | None = None) -> list[dict[str, Any]]:
    normalized_steps: list[dict[str, Any]] = []
    for index, item in enumerate(list(plan or []), start=1):
        if not isinstance(item, dict):
            continue
        tool = str(item.get("tool") or TOOL_PYTHON).strip().upper()
        if tool not in {TOOL_SQL, TOOL_EXCEL, TOOL_PYTHON, TOOL_BI}:
            continue
        task = str(item.get("task") or "").strip()
        if not task:
            continue
        step = {
            "step": int(item.get("step") or index),
            "tool": tool,
            "task": task,
            "query": str(item.get("query") or task).strip() or task,
            "depends_on": _normalize_dependencies(item.get("depends_on")),
            "uses_context": bool(item.get("uses_context")),
            "cost_estimate": estimate_cost(item, df=df),
        }
        if item.get("sql_plan") is not None:
            step["sql_plan"] = str(item.get("sql_plan") or "").strip() or None
        if isinstance(item.get("excel_logic"), dict) and item.get("excel_logic"):
            step["excel_logic"] = dict(item.get("excel_logic") or {})
        python_steps = [str(part).strip() for part in list(item.get("python_steps") or []) if str(part).strip()]
        if python_steps:
            step["python_steps"] = python_steps
        if item.get("fallback_reason") is not None:
            step["fallback_reason"] = str(item.get("fallback_reason") or "").strip() or None

        if index > 1 and not step["depends_on"]:
            task_text = " ".join(
                str(step.get(key) or "").strip().lower()
                for key in ("task", "query")
            )
            if step.get("uses_context") or "current context" in task_text:
                step["depends_on"] = [index - 1]
                step["uses_context"] = True
        normalized_steps.append(step)

    for index, step in enumerate(normalized_steps, start=1):
        step["step"] = index
        step["depends_on"] = [dependency for dependency in list(step.get("depends_on") or []) if dependency < index]
        step["cost_estimate"] = estimate_cost(step, df=df)
    return normalized_steps


def _copy_tables(tables: dict[str, pd.DataFrame] | None, df: pd.DataFrame) -> dict[str, Any]:
    copied_tables: dict[str, Any] = {}
    for name, value in dict(tables or {"df": df}).items():
        if isinstance(value, pd.DataFrame):
            copied_tables[str(name)] = value.copy()
        else:
            copied_tables[str(name)] = value
    copied_tables["df"] = df.copy()
    return copied_tables


def _build_fallback_step(step: dict[str, Any], fallback_reason: str) -> dict[str, Any]:
    return {
        "step": int(step.get("step") or 0),
        "tool": TOOL_EXCEL,
        "task": "Fallback to an Excel-style analyst summary.",
        "query": str(step.get("query") or step.get("task") or "").strip() or "Fallback to Excel summary",
        "depends_on": list(step.get("depends_on") or []),
        "uses_context": bool(step.get("uses_context")),
        "fallback_reason": str(fallback_reason).strip() or None,
        "cost_estimate": estimate_cost({"tool": TOOL_EXCEL}),
    }


def _execute_single_step(
    step: dict[str, Any],
    df: pd.DataFrame,
    *,
    query: str,
    current_result: Any,
    analysis_plan: dict[str, Any],
    preflight: dict[str, Any],
    tables: dict[str, pd.DataFrame] | None,
    python_runner: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    step_payload = dict(step)
    step_payload["cost_estimate"] = estimate_cost(step_payload)
    tool = str(step_payload.get("tool") or TOOL_PYTHON).strip().upper()
    step_query = str(step_payload.get("query") or step_payload.get("task") or query).strip() or query
    step_warnings: list[str] = []
    step_error = None
    module_validation = None
    analysis_method = None
    fix_applied = False
    fixed_code = None
    fix_status = "No automatic fix was attempted."
    generated_code = None
    result = None
    excel_analysis = None
    dashboard = None

    if tool == TOOL_SQL:
        sql_payload = run_sql_analysis(
            step_query,
            df,
            tables=tables,
            plan=analysis_plan,
            preflight=preflight,
        )
        step_warnings.extend(list(sql_payload.get("warnings") or []))
        if sql_payload.get("unsupported") or sql_payload.get("result") is None:
            step_error = step_warnings[0] if step_warnings else "SQL simulation could not execute safely."
        else:
            result = sql_payload.get("result")
            generated_code = _render_sql_plan(sql_payload.get("sql_plan"))
            step_payload["sql_plan"] = sql_payload.get("sql_plan")
        analysis_method = "deterministic_sql"
        module_validation = "VALID\nDeterministic SQL simulation selected."

    elif tool == TOOL_EXCEL:
        excel_payload = run_excel_analysis(
            step_query,
            df,
            plan=analysis_plan,
            preflight=preflight,
        )
        step_warnings.extend(list(excel_payload.get("warnings") or []))
        result = excel_payload.get("result")
        excel_analysis = excel_payload.get("excel_analysis")
        generated_code = _render_excel_logic(excel_analysis)
        step_payload["excel_logic"] = {
            "pivot_table": dict((excel_analysis or {}).get("pivot_table") or {}),
            "aggregations": dict((excel_analysis or {}).get("aggregations") or {}),
        }
        analysis_method = "deterministic_excel"
        module_validation = "VALID\nDeterministic Excel analysis selected."

    elif tool == TOOL_BI:
        dashboard_payload = build_dashboard_output(
            step_query,
            df,
            result=current_result,
            plan=analysis_plan,
            preflight=preflight,
        )
        step_warnings.extend(list(dashboard_payload.get("warnings") or []))
        dashboard = dashboard_payload.get("dashboard")
        if not list((dashboard or {}).get("charts") or []):
            step_error = step_warnings[0] if step_warnings else "Dashboard generation could not find a reliable chartable combination."
        else:
            result = {"dashboard": dashboard}
            generated_code = _render_dashboard(dashboard)
        analysis_method = "deterministic_bi"
        module_validation = "VALID\nDeterministic BI dashboard generation selected."

    elif tool == TOOL_PYTHON:
        if python_runner is None:
            step_error = "Python orchestration requested a Python step runner, but none was provided."
            analysis_method = "multi_tool_python"
        else:
            python_payload = python_runner(
                step=step_payload,
                query=step_query,
                df=df,
                current_result=current_result,
                analysis_plan=analysis_plan,
                preflight=preflight,
            )
            step_warnings.extend(list(python_payload.get("warnings") or []))
            step_error = str(python_payload.get("error") or "").strip() or None
            module_validation = str(python_payload.get("module_validation") or "VALID\nPython orchestration step selected.")
            analysis_method = str(python_payload.get("analysis_method") or "multi_tool_python")
            result = python_payload.get("result")
            generated_code = str(
                python_payload.get("final_code")
                or python_payload.get("generated_code")
                or ""
            ).strip() or None
            python_steps = [str(item).strip() for item in list(python_payload.get("python_steps") or []) if str(item).strip()]
            if python_steps:
                step_payload["python_steps"] = python_steps
            fix_applied = bool(python_payload.get("fix_applied"))
            fixed_code = str(
                python_payload.get("fixed_code")
                or python_payload.get("final_code")
                or python_payload.get("generated_code")
                or ""
            ).strip() or None
            fix_status = str(python_payload.get("fix_status") or fix_status)

    else:
        step_error = f"Unsupported tool `{tool}` in execution plan."

    result_frame = _result_to_dataframe(result)
    return {
        "step": step_payload,
        "result": result,
        "result_frame": result_frame,
        "warnings": step_warnings,
        "error": step_error,
        "module_validation": module_validation,
        "analysis_method": analysis_method,
        "generated_code": generated_code,
        "excel_analysis": excel_analysis,
        "dashboard": dashboard,
        "fix_applied": fix_applied,
        "fixed_code": fixed_code,
        "fix_status": fix_status,
    }


def _execute_step_with_tracking(
    step: dict[str, Any],
    df: pd.DataFrame,
    *,
    query: str,
    current_result: Any,
    analysis_plan: dict[str, Any],
    preflight: dict[str, Any],
    tables: dict[str, pd.DataFrame] | None,
    python_runner: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    start_time = time.perf_counter()
    payload = _execute_single_step(
        step,
        df,
        query=query,
        current_result=current_result,
        analysis_plan=analysis_plan,
        preflight=preflight,
        tables=tables,
        python_runner=python_runner,
    )
    end_time = time.perf_counter()
    trace_entry = track_performance(
        payload["step"],
        start_time,
        end_time,
        df=df,
        status="failed" if payload.get("error") else "completed",
        warnings=payload.get("warnings"),
        error=payload.get("error"),
    )
    record_tool_performance(
        payload["step"].get("tool"),
        trace_entry["execution_time_ms"],
        status=trace_entry["status"],
    )
    return {
        "payload": payload,
        "trace_entries": [trace_entry],
    }


def _execute_with_fallback(
    step: dict[str, Any],
    df: pd.DataFrame,
    *,
    query: str,
    current_result: Any,
    analysis_plan: dict[str, Any],
    preflight: dict[str, Any],
    tables: dict[str, pd.DataFrame] | None,
    python_runner: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    primary_execution = _execute_step_with_tracking(
        step,
        df,
        query=query,
        current_result=current_result,
        analysis_plan=analysis_plan,
        preflight=preflight,
        tables=tables,
        python_runner=python_runner,
    )
    payload = primary_execution["payload"]
    trace_entries = list(primary_execution["trace_entries"] or [])
    warnings = list(payload.get("warnings") or [])
    final_step = dict(payload["step"])

    if not payload.get("error"):
        return {
            "payload": payload,
            "trace_entries": trace_entries,
            "warnings": warnings,
            "plan_step": final_step,
        }

    if str(final_step.get("tool") or "").strip().upper() == TOOL_EXCEL:
        return {
            "payload": payload,
            "trace_entries": trace_entries,
            "warnings": warnings,
            "plan_step": final_step,
        }

    fallback_step = _build_fallback_step(final_step, str(payload.get("error") or ""))
    fallback_execution = _execute_step_with_tracking(
        fallback_step,
        df,
        query=query,
        current_result=current_result,
        analysis_plan=analysis_plan,
        preflight=preflight,
        tables=tables,
        python_runner=None,
    )
    fallback_payload = fallback_execution["payload"]
    fallback_trace = dict(fallback_execution["trace_entries"][0])
    fallback_trace["status"] = "fallback_completed"
    fallback_trace["fallback_tool"] = TOOL_EXCEL
    trace_entries.append(fallback_trace)
    warnings.extend(list(fallback_payload.get("warnings") or []))

    if fallback_payload.get("excel_analysis") is not None:
        fallback_step["excel_logic"] = {
            "pivot_table": dict((fallback_payload.get("excel_analysis") or {}).get("pivot_table") or {}),
            "aggregations": dict((fallback_payload.get("excel_analysis") or {}).get("aggregations") or {}),
        }
    final_payload = dict(fallback_payload)
    final_payload["error"] = None
    final_payload["fix_applied"] = bool(payload.get("fix_applied") or fallback_payload.get("fix_applied"))
    final_payload["fixed_code"] = payload.get("fixed_code") or fallback_payload.get("fixed_code")
    final_payload["fix_status"] = payload.get("fix_status") or fallback_payload.get("fix_status")
    return {
        "payload": final_payload,
        "trace_entries": trace_entries,
        "warnings": warnings,
        "plan_step": fallback_step,
        "primary_error": str(payload.get("error") or "").strip() or None,
    }


def _resolve_step_context(
    step: dict[str, Any],
    *,
    base_df: pd.DataFrame,
    base_tables: dict[str, pd.DataFrame] | None,
    step_outputs: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    dependencies = [int(item) for item in list(step.get("depends_on") or []) if int(item) > 0]
    context_df = base_df.copy()
    context_result: Any = None
    context_tables = _copy_tables(base_tables, context_df)

    if not dependencies:
        return {
            "df": context_df,
            "current_result": context_result,
            "tables": context_tables,
        }

    dependency_results: dict[str, Any] = {}
    last_dependency_payload: dict[str, Any] | None = None
    for dependency in dependencies:
        dependency_payload = dict(step_outputs.get(dependency) or {})
        if not dependency_payload:
            continue
        last_dependency_payload = dependency_payload
        dependency_results[str(dependency)] = dependency_payload.get("result")
        dependency_frame = dependency_payload.get("result_frame")
        if isinstance(dependency_frame, pd.DataFrame):
            context_tables[f"step_{dependency}"] = dependency_frame.copy()
        else:
            context_tables[f"step_{dependency}"] = dependency_payload.get("result")

    if last_dependency_payload is not None:
        dependency_frame = last_dependency_payload.get("result_frame")
        if isinstance(dependency_frame, pd.DataFrame):
            context_df = dependency_frame.copy()
            context_tables["df"] = context_df
        if len(dependencies) == 1:
            context_result = last_dependency_payload.get("result")
        else:
            context_result = {"dependency_results": dependency_results}

    return {
        "df": context_df,
        "current_result": context_result,
        "tables": context_tables,
    }


def _execute_step_batch(
    steps: list[dict[str, Any]],
    *,
    query: str,
    analysis_plan: dict[str, Any],
    preflight: dict[str, Any],
    base_df: pd.DataFrame,
    base_tables: dict[str, pd.DataFrame] | None,
    step_outputs: dict[int, dict[str, Any]],
    python_runner: Callable[..., dict[str, Any]] | None = None,
    enable_parallel: bool = True,
) -> list[dict[str, Any]]:
    if not steps:
        return []

    step_inputs = [
        (
            dict(step),
            _resolve_step_context(
                step,
                base_df=base_df,
                base_tables=base_tables,
                step_outputs=step_outputs,
            ),
        )
        for step in sorted(steps, key=lambda item: int(item.get("step") or 0))
    ]

    if len(step_inputs) == 1 or not enable_parallel:
        results: list[dict[str, Any]] = []
        for step, context in step_inputs:
            results.append(
                _execute_with_fallback(
                    step,
                    context["df"],
                    query=query,
                    current_result=context["current_result"],
                    analysis_plan=analysis_plan,
                    preflight=preflight,
                    tables=context["tables"],
                    python_runner=python_runner,
                )
            )
        return results

    with ThreadPoolExecutor(max_workers=len(step_inputs)) as executor:
        futures = [
            executor.submit(
                _execute_with_fallback,
                step,
                context["df"],
                query=query,
                current_result=context["current_result"],
                analysis_plan=dict(analysis_plan or {}),
                preflight=dict(preflight or {}),
                tables=context["tables"],
                python_runner=python_runner,
            )
            for step, context in step_inputs
        ]
        results = [future.result() for future in futures]
    return sorted(results, key=lambda item: int((item.get("plan_step") or {}).get("step") or 0))


def execute_plan(
    plan: list[dict[str, Any]] | None,
    df: pd.DataFrame,
    *,
    query: str,
    analysis_plan: dict[str, Any],
    preflight: dict[str, Any],
    tables: dict[str, pd.DataFrame] | None = None,
    python_runner: Callable[..., dict[str, Any]] | None = None,
    optimize: bool = True,
    enable_parallel: bool = True,
    constraints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    baseline_plan = _basic_normalize_plan(plan, df=df)
    selected_plan_metadata: dict[str, Any] = {
        "plans_considered": 1,
        "selected_plan_score": 0.0,
        "constraints_applied": dict(constraints or {}),
        "parallel_execution": False,
        "optimized": False,
    }
    execution_plan = list(baseline_plan)
    distinct_tools = {
        str(step.get("tool") or "").strip().upper()
        for step in baseline_plan
    }
    use_intelligent_optimization = bool(optimize and (constraints or len(distinct_tools) > 1))

    if use_intelligent_optimization:
        try:
            optimizer_payload = select_best_plan(
                query,
                df,
                base_plan=baseline_plan,
                constraints=constraints,
                preflight=preflight,
                tool_performance=get_tool_performance_memory(),
            )
            candidate_plan = list(optimizer_payload.get("selected_plan") or [])
            if candidate_plan:
                execution_plan = _basic_normalize_plan(candidate_plan, df=df)
            selected_plan_metadata = dict(optimizer_payload.get("optimization") or selected_plan_metadata)
        except Exception:
            execution_plan = list(baseline_plan)
    elif optimize:
        execution_plan = _basic_normalize_plan(
            optimize_single_plan(baseline_plan, df=df, preflight=preflight),
            df=df,
        )
        selected_plan_metadata["optimized"] = len(execution_plan) != len(baseline_plan)
        selected_plan_metadata["selected_plan_score"] = 1.0 if execution_plan else 0.0
    plan_was_optimized = bool(optimize and len(execution_plan) > 0)
    graph = build_execution_graph(execution_plan)

    execution_trace: list[dict[str, Any]] = []
    code_segments: list[str] = []
    warnings: list[str] = []
    excel_analysis = None
    dashboard = None
    module_validation = "VALID\nMulti-tool orchestration plan selected."
    analysis_method = "multi_tool_orchestration"
    error = None
    fix_applied = False
    fixed_code = None
    fix_status = "No automatic fix was attempted."
    mutated_plan: list[dict[str, Any]] = []
    parallel_execution_used = False
    terminal_results: dict[int, Any] = {}
    step_outputs: dict[int, dict[str, Any]] = {}
    code_segments_by_step: dict[int, str] = {}
    base_df = df.copy()
    base_tables = _copy_tables(tables, base_df)

    if graph.get("has_cycle"):
        error = "Execution plan contains circular dependencies."
    else:
        for batch in list(graph.get("batches") or []):
            step_batch = [
                dict(graph["nodes"][step_id])
                for step_id in batch
                if step_id in graph.get("nodes", {})
            ]
            if not step_batch:
                continue
            use_parallel_for_batch = enable_parallel and len(step_batch) > 1
            if use_parallel_for_batch:
                parallel_execution_used = True

            batch_results = _execute_step_batch(
                step_batch,
                query=query,
                analysis_plan=analysis_plan,
                preflight=preflight,
                base_df=base_df,
                base_tables=base_tables,
                step_outputs=step_outputs,
                python_runner=python_runner,
                enable_parallel=use_parallel_for_batch,
            )

            for result_bundle in batch_results:
                payload = dict(result_bundle.get("payload") or {})
                plan_step = dict(result_bundle.get("plan_step") or payload.get("step") or {})
                step_id = int(plan_step.get("step") or 0)
                mutated_plan.append(plan_step)
                execution_trace.extend(list(result_bundle.get("trace_entries") or []))
                warnings.extend(list(result_bundle.get("warnings") or []))

                module_validation = str(payload.get("module_validation") or module_validation)
                analysis_method = str(payload.get("analysis_method") or analysis_method)
                fix_applied = bool(fix_applied or payload.get("fix_applied"))
                if payload.get("fixed_code"):
                    fixed_code = str(payload.get("fixed_code") or "").strip() or fixed_code
                if payload.get("fix_status"):
                    fix_status = str(payload.get("fix_status") or fix_status)

                if payload.get("generated_code"):
                    code_segments_by_step[step_id] = str(payload.get("generated_code"))

                if payload.get("excel_analysis") is not None:
                    excel_analysis = payload.get("excel_analysis")
                if payload.get("dashboard") is not None:
                    dashboard = payload.get("dashboard")

                step_outputs[step_id] = payload
                if step_id in set(graph.get("terminal_nodes") or []):
                    terminal_results[step_id] = payload.get("result")

                if payload.get("error"):
                    error = str(payload.get("error"))
                    break

            if error:
                break

    ordered_code_segments = [
        code_segments_by_step[step_id]
        for step_id in sorted(code_segments_by_step)
        if str(code_segments_by_step.get(step_id) or "").strip()
    ]
    code_segments.extend(ordered_code_segments)
    final_code = "\n\n".join(segment for segment in code_segments if str(segment).strip()).strip() or None
    if fix_applied and not fixed_code:
        fixed_code = final_code

    final_result: Any = None
    terminal_nodes = sorted(int(step_id) for step_id in list(graph.get("terminal_nodes") or []))
    if len(terminal_nodes) == 1 and terminal_nodes[0] in step_outputs:
        terminal_payload = dict(step_outputs.get(terminal_nodes[0]) or {})
        final_result = terminal_payload.get("result")
        if final_result is None and terminal_payload.get("result_frame") is not None:
            final_result = terminal_payload.get("result_frame")
    elif terminal_nodes:
        if parallel_execution_used:
            final_result = {
                "parallel_results": {
                    str(step_id): terminal_results.get(step_id)
                    for step_id in terminal_nodes
                }
            }
        else:
            last_terminal_payload = dict(step_outputs.get(terminal_nodes[-1]) or {})
            final_result = last_terminal_payload.get("result")
            if final_result is None and last_terminal_payload.get("result_frame") is not None:
                final_result = last_terminal_payload.get("result_frame")
    else:
        final_result = base_df

    optimization = build_optimization_summary(
        execution_trace,
        optimized=plan_was_optimized,
        parallel_execution=parallel_execution_used,
        plans_considered=selected_plan_metadata.get("plans_considered"),
        selected_plan_score=selected_plan_metadata.get("selected_plan_score"),
        constraints_applied=selected_plan_metadata.get("constraints_applied"),
    )
    if selected_plan_metadata.get("cost_estimate"):
        optimization["cost_estimate"] = str(selected_plan_metadata.get("cost_estimate") or optimization.get("cost_estimate"))
    if not error and "parallel_execution" in selected_plan_metadata:
        optimization["parallel_execution"] = bool(parallel_execution_used or selected_plan_metadata.get("parallel_execution"))

    return {
        "result": final_result,
        "execution_plan": mutated_plan or execution_plan,
        "execution_trace": execution_trace,
        "warnings": list(dict.fromkeys(str(item) for item in warnings if str(item).strip())),
        "excel_analysis": excel_analysis,
        "dashboard": dashboard,
        "generated_code": final_code,
        "final_code": final_code,
        "error": error,
        "analysis_method": analysis_method,
        "module_validation": module_validation,
        "fix_applied": fix_applied,
        "fix_status": fix_status,
        "fixed_code": fixed_code,
        "optimization": optimization,
    }
