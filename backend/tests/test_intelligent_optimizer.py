import time
import unittest
from unittest import mock

import pandas as pd

from backend.services.dag_execution import build_execution_graph
from backend.services.execution_engine import execute_plan
from backend.services.intelligent_optimizer import (
    generate_candidate_plans,
    score_plan,
    select_best_plan,
)
from backend.services.tool_planner import build_execution_plan


class IntelligentOptimizerTests(unittest.TestCase):
    def setUp(self):
        self.sales_df = pd.DataFrame(
            {
                "order_date": pd.date_range("2025-01-01", periods=14, freq="D"),
                "region": ["North", "South", "North", "West", "South", "North", "West", "South", "North", "West", "South", "North", "East", "East"],
                "customer": ["A", "B", "A", "C", "B", "D", "C", "D", "A", "C", "B", "D", "E", "F"],
                "sales": [100, 140, 125, 90, 160, 180, 95, 170, 190, 110, 175, 205, 150, 165],
                "revenue": [1000, 1400, 1250, 900, 1600, 1800, 950, 1700, 1900, 1100, 1750, 2050, 1500, 1650],
            }
        )
        self.base_preflight = {"warnings": [], "blocking_errors": []}

    def test_generate_candidate_plans_returns_multiple_execution_strategies(self):
        base_plan = build_execution_plan(
            "Top customers + prediction + dashboard",
            self.sales_df,
            plan={
                "intent": "prediction",
                "analysis_type": "aggregation",
                "metric_column": "revenue",
                "group_column": "customer",
                "datetime_column": "order_date",
            },
            preflight=self.base_preflight,
        )

        candidates = generate_candidate_plans(
            "Top customers + prediction + dashboard",
            self.sales_df,
            base_plan=base_plan,
            preflight=self.base_preflight,
        )

        tool_sequences = [tuple(step["tool"] for step in plan) for plan in candidates]
        self.assertGreaterEqual(len(candidates), 3)
        self.assertIn(("SQL", "PYTHON", "BI"), tool_sequences)
        self.assertIn(("PYTHON", "BI"), tool_sequences)
        self.assertIn(("EXCEL", "PYTHON", "BI"), tool_sequences)

    def test_select_best_plan_prefers_reliable_plan_by_default(self):
        base_plan = build_execution_plan(
            "Top customers + prediction + dashboard",
            self.sales_df,
            plan={
                "intent": "prediction",
                "analysis_type": "aggregation",
                "metric_column": "revenue",
                "group_column": "customer",
                "datetime_column": "order_date",
            },
            preflight=self.base_preflight,
        )

        selection = select_best_plan(
            "Top customers + prediction + dashboard",
            self.sales_df,
            base_plan=base_plan,
            constraints=None,
            preflight=self.base_preflight,
        )

        self.assertGreaterEqual(selection["optimization"]["plans_considered"], 3)
        self.assertGreater(selection["optimization"]["selected_plan_score"], 0.0)
        self.assertEqual([step["tool"] for step in selection["selected_plan"]], ["SQL", "PYTHON", "BI"])

    def test_select_best_plan_honors_low_budget_mode(self):
        base_plan = build_execution_plan(
            "Sales trends + dashboard",
            self.sales_df,
            plan={
                "intent": "visualization",
                "analysis_type": "aggregation",
                "metric_column": "sales",
                "group_column": "region",
                "datetime_column": "order_date",
            },
            preflight=self.base_preflight,
        )

        selection = select_best_plan(
            "Sales trends + dashboard",
            self.sales_df,
            base_plan=base_plan,
            constraints={"budget": "low", "priority": "cost"},
            preflight=self.base_preflight,
        )

        self.assertEqual(selection["optimization"]["constraints_applied"]["budget"], "low")
        self.assertEqual([step["tool"] for step in selection["selected_plan"]], ["EXCEL", "BI"])

    def test_select_best_plan_honors_high_speed_mode(self):
        base_plan = build_execution_plan(
            "Sales trends + dashboard",
            self.sales_df,
            plan={
                "intent": "visualization",
                "analysis_type": "aggregation",
                "metric_column": "sales",
                "group_column": "region",
                "datetime_column": "order_date",
            },
            preflight=self.base_preflight,
        )

        relaxed_selection = select_best_plan(
            "Sales trends + dashboard",
            self.sales_df,
            base_plan=base_plan,
            constraints={"max_execution_time": 200, "priority": "speed"},
            preflight=self.base_preflight,
        )
        constrained_selection = select_best_plan(
            "Sales trends + dashboard",
            self.sales_df,
            base_plan=base_plan,
            constraints={"max_execution_time": 120, "priority": "speed"},
            preflight=self.base_preflight,
        )

        self.assertEqual(constrained_selection["optimization"]["constraints_applied"]["priority"], "speed")
        self.assertEqual(constrained_selection["optimization"]["constraints_applied"]["max_execution_time"], 120)
        self.assertTrue(constrained_selection["selected_plan"])
        self.assertLess(
            constrained_selection["optimization"]["selected_plan_score"],
            relaxed_selection["optimization"]["selected_plan_score"],
        )

    def test_score_plan_uses_constraints_and_reliability(self):
        reliable_plan = [
            {"step": 1, "tool": "SQL", "task": "Get top customers", "query": "Top customers"},
            {"step": 2, "tool": "PYTHON", "task": "Run forecast", "query": "Predict revenue", "depends_on": [1], "uses_context": True},
            {"step": 3, "tool": "BI", "task": "Build dashboard", "query": "Build dashboard", "depends_on": [2], "uses_context": True},
        ]
        cheaper_plan = [
            {"step": 1, "tool": "PYTHON", "task": "Run forecast", "query": "Predict revenue"},
            {"step": 2, "tool": "BI", "task": "Build dashboard", "query": "Build dashboard", "depends_on": [1], "uses_context": True},
        ]

        reliable_score = score_plan(
            reliable_plan,
            {"priority": "reliability"},
            df=self.sales_df,
            query="Top customers + prediction + dashboard",
            preflight=self.base_preflight,
        )
        cheaper_score = score_plan(
            cheaper_plan,
            {"priority": "reliability"},
            df=self.sales_df,
            query="Top customers + prediction + dashboard",
            preflight=self.base_preflight,
        )

        self.assertGreater(reliable_score["reliability"], cheaper_score["reliability"])
        self.assertGreater(reliable_score["score"], cheaper_score["score"])

    def test_build_execution_graph_exposes_parallel_batches(self):
        plan = [
            {"step": 1, "tool": "SQL", "task": "Get top customers", "query": "Top customers"},
            {"step": 2, "tool": "EXCEL", "task": "Show sales by region", "query": "Show sales by region"},
            {"step": 3, "tool": "PYTHON", "task": "Combine both contexts", "query": "Analyze both", "depends_on": [1, 2], "uses_context": True},
        ]

        graph = build_execution_graph(plan)

        self.assertEqual(graph["batches"], [[1, 2], [3]])
        self.assertEqual(graph["terminal_nodes"], [3])

    def test_execute_plan_runs_dag_batches_in_parallel(self):
        plan = [
            {"step": 1, "tool": "SQL", "task": "Get top customers", "query": "Top customers"},
            {"step": 2, "tool": "EXCEL", "task": "Show sales by region", "query": "Show sales by region"},
            {"step": 3, "tool": "PYTHON", "task": "Analyze the dependency results", "query": "Analyze both", "depends_on": [1, 2], "uses_context": True},
        ]

        def slow_sql(*args, **kwargs):
            time.sleep(0.05)
            return {
                "result": pd.DataFrame({"customer": ["A"], "revenue": [5900.0]}),
                "sql_plan": "SELECT customer, SUM(revenue) FROM df GROUP BY customer",
                "warnings": [],
                "unsupported": False,
            }

        def slow_excel(*args, **kwargs):
            time.sleep(0.05)
            return {
                "result": pd.DataFrame({"region": ["North"], "sales": [620.0]}),
                "excel_analysis": {
                    "pivot_table": {"index": ["region"], "values": ["sales"], "aggfunc": "sum"},
                    "aggregations": {"sales_sum": 620.0},
                    "summary": {"group_column": "region", "metric_column": "sales"},
                },
                "warnings": [],
            }

        python_observations = {}

        def python_runner(**kwargs):
            dependency_results = dict((kwargs.get("current_result") or {}).get("dependency_results") or {})
            python_observations["dependency_result_count"] = len(dependency_results)
            return {
                "result": pd.DataFrame({"combined_score": [6520.0]}),
                "generated_code": "result = pd.DataFrame({'combined_score': [6520.0]})",
                "final_code": "result = pd.DataFrame({'combined_score': [6520.0]})",
                "error": None,
                "fix_applied": False,
                "fix_status": "Execution passed without needing an automatic fix.",
                "fixed_code": None,
                "analysis_method": "test_python_runner",
                "module_validation": "VALID\nPython step selected.",
                "python_steps": ["read dependency results", "return combined frame"],
                "warnings": [],
            }

        with mock.patch("backend.services.execution_engine.run_sql_analysis", side_effect=slow_sql):
            with mock.patch("backend.services.execution_engine.run_excel_analysis", side_effect=slow_excel):
                serial_start = time.perf_counter()
                serial_payload = execute_plan(
                    plan,
                    self.sales_df,
                    query="Top customers + sales by region + analyze",
                    analysis_plan={"intent": "analysis", "metric_column": "sales", "group_column": "region"},
                    preflight=self.base_preflight,
                    tables={"df": self.sales_df},
                    python_runner=python_runner,
                    optimize=False,
                    enable_parallel=False,
                )
                serial_elapsed = time.perf_counter() - serial_start

                parallel_start = time.perf_counter()
                parallel_payload = execute_plan(
                    plan,
                    self.sales_df,
                    query="Top customers + sales by region + analyze",
                    analysis_plan={"intent": "analysis", "metric_column": "sales", "group_column": "region"},
                    preflight=self.base_preflight,
                    tables={"df": self.sales_df},
                    python_runner=python_runner,
                    optimize=False,
                    enable_parallel=True,
                )
                parallel_elapsed = time.perf_counter() - parallel_start

        self.assertEqual(python_observations["dependency_result_count"], 2)
        self.assertFalse(serial_payload["optimization"]["parallel_execution"])
        self.assertTrue(parallel_payload["optimization"]["parallel_execution"])
        self.assertLess(parallel_elapsed, serial_elapsed)
        self.assertIn("plans_considered", parallel_payload["optimization"])
        self.assertIn("selected_plan_score", parallel_payload["optimization"])


if __name__ == "__main__":
    unittest.main()
