import builtins
import json
import os
import re
import threading
import time
from pathlib import Path

import pandas as pd
from dotenv import dotenv_values
try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None

from backend.analysis_contract import (
    build_analysis_contract,
    build_analysis_plan,
    build_deterministic_analysis_code,
    classify_analysis_intent,
    ensure_analysis_contract_defaults,
    validate_analysis_request,
)
from backend.data_quality import build_data_quality_report
from backend.insight_engine import generate_decision_grade_insights
from backend.aidssist_runtime.config import get_settings
from backend.aidssist_runtime.metrics import observe_llm_call
from backend.services.learning_engine import get_learning_patterns
from backend.services.failure_logging import get_failure_patterns, log_failure
from backend.services.limitations import build_limitations
from backend.services.ml_postprocessor import postprocess_ml_output
from backend.services.ml_schema_validator import validate_ml_output
from backend.services.model_quality import build_simple_prediction_diagnostics, interpret_model_quality
from backend.services.result_consistency import build_analysis_consistency, build_reproducibility_metadata
from backend.services.execution_engine import execute_plan
from backend.services.data_intelligence import detect_dataset_type
from backend.services.ml_intelligence import build_ml_intelligence
from backend.services.mode_router import decide_analysis_mode
from backend.services.trust_layer import build_risk_statement
from backend.services.dashboard_engine import build_dashboard_output
from backend.services.excel_engine import run_excel_analysis
from backend.services.sql_engine import run_sql_analysis
from backend.services.tool_planner import build_execution_plan, determine_analysis_mode, determine_primary_tool
from backend.services.tool_selector import select_tool
from backend.workflow_store import WorkflowStore


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
ENV_FILE_PATH = PROJECT_ROOT / ".env"
LOG_PATH = BASE_DIR / "logs.txt"
ALLOWED_IMPORT_ROOTS = {"pandas", "numpy", "matplotlib", "seaborn", "sklearn"}
DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"
DEFAULT_GROQ_MODEL = DEFAULT_GEMINI_MODEL
PIPELINE_CACHE_VERSION = get_settings().pipeline_cache_version
DEFAULT_TEST_CASES = (
    {"query": "Top 5 rows", "type": "easy"},
    {"query": "Average sales by region", "type": "medium"},
    {"query": "Why did revenue drop", "type": "hard"},
)
PLACEHOLDER_KEY_PATTERNS = (
    re.compile(r"(^|[-_])your($|[-_])"),
    re.compile(r"(^|[-_])(replace|placeholder|example|sample|dummy|fake)($|[-_])"),
    re.compile(r"(^|[-_])real[-_]key($|[-_])"),
    re.compile(r"(^|[-_])(api|openai|google|gemini|groq)[-_](key|token)([-_](here|value))?($|[-_])"),
    re.compile(r"(^|[-_])here($|[-_])"),
)
_GEMINI_CONCURRENCY_SEMAPHORE = threading.BoundedSemaphore(
    max(1, int(get_settings().provider_max_concurrency_per_process))
)


def _normalize_config_value(value):
    return str(value or "").strip().strip('"').strip("'")


def _read_env_file_value(*key_names):
    env_values = dotenv_values(ENV_FILE_PATH)

    for key_name in key_names:
        config_value = _normalize_config_value(env_values.get(key_name))
        if config_value:
            return config_value

    return ""


def _looks_like_placeholder_api_key(api_key):
    normalized_key = _normalize_config_value(api_key)

    if not normalized_key:
        return False

    lowered_key = normalized_key.lower()

    if any(character.isspace() for character in normalized_key):
        return True

    if any(character in normalized_key for character in "*<>"):
        return True

    return any(pattern.search(lowered_key) for pattern in PLACEHOLDER_KEY_PATTERNS)


def get_groq_configuration_status():
    return get_gemini_configuration_status()


def _ensure_groq_configuration():
    _ensure_gemini_configuration()


def _resolve_groq_model_name(model):
    return _resolve_gemini_model_name(model)


def _get_gemini_api_key():
    file_api_key = _read_env_file_value("GEMINI_API_KEY")

    if file_api_key:
        return file_api_key

    for key_name in ("GEMINI_API_KEY",):
        config_value = _normalize_config_value(os.getenv(key_name))
        if config_value:
            return config_value

    return ""


def _get_gemini_api_key_source():
    if _read_env_file_value("GEMINI_API_KEY"):
        return ".env"

    for key_name in ("GEMINI_API_KEY",):
        if _normalize_config_value(os.getenv(key_name)):
            return "your shell environment"

    return ".env"


def _get_gemini_model_name():
    model_name = _read_env_file_value("GEMINI_MODEL")
    if model_name:
        return model_name

    return _normalize_config_value(os.getenv("GEMINI_MODEL")) or DEFAULT_GEMINI_MODEL


def get_gemini_configuration_status():
    if genai is None or genai_types is None:
        return (
            False,
            "Gemini SDK is not installed. Install dependencies with `pip install -r requirements.txt` before running analysis.",
        )

    api_key = _get_gemini_api_key()
    key_source = _get_gemini_api_key_source()

    if not api_key:
        return (
            False,
            "Gemini API key is missing. Add your real GEMINI_API_KEY to .env before running analysis.",
        )

    if _looks_like_placeholder_api_key(api_key):
        return (
            False,
            f"Gemini API key in {key_source} is still using a placeholder value. Replace it with your real key before running analysis.",
        )

    return True, None


def _ensure_gemini_configuration():
    is_ready, message = get_gemini_configuration_status()
    if not is_ready:
        raise RuntimeError(message)


def get_provider_configuration_status():
    is_ready, message = get_gemini_configuration_status()
    if is_ready:
        return True, "Gemini Flash is ready for code generation, analysis, explanations, and research-backed outputs."
    return is_ready, message


def _ensure_builder_configuration():
    _ensure_gemini_configuration()


def _resolve_gemini_model_name(model):
    normalized_model = _normalize_config_value(model)

    if normalized_model.startswith("gemini"):
        return normalized_model

    return _get_gemini_model_name()


def _normalize_message_content(content):
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                text_parts.append(item)

        return "\n".join(part for part in text_parts if part)

    return str(content)


def _build_groq_messages(messages):
    normalized_messages = []

    for message in messages or []:
        role = str(message.get("role", "user")).strip().lower()
        text = _normalize_message_content(message.get("content", ""))

        if not text:
            continue

        if role not in {"system", "user", "assistant"}:
            role = "user"

        normalized_messages.append({"role": role, "content": text})

    if not normalized_messages:
        normalized_messages = [{"role": "user", "content": ""}]

    return normalized_messages


def _generate_groq_content(*, model, messages):
    prompt = "\n\n".join(
        _normalize_message_content(message.get("content", ""))
        for message in _build_groq_messages(messages)
    ).strip()
    return _generate_gemini_content(
        model=_resolve_groq_model_name(model),
        prompt=prompt,
        use_search=False,
    )

def _extract_gemini_text(response):
    text = _normalize_config_value(getattr(response, "text", ""))
    if text:
        return text

    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            part_text = _normalize_config_value(getattr(part, "text", ""))
            if part_text:
                return part_text

    raise RuntimeError("Gemini returned no text content.")


def _format_gemini_exception(error):
    message = _normalize_config_value(error)
    lowered_message = message.lower()

    if any(token in lowered_message for token in ("api key", "invalid api key", "permission", "unauthorized", "401", "403")):
        return "Gemini rejected the API key. Update GEMINI_API_KEY with a valid key and rerun the app."
    if any(token in lowered_message for token in ("quota", "429", "resource_exhausted", "rate limit")):
        return "Gemini rate or quota limit reached. Check your Gemini usage and billing details, then try again."
    if any(token in lowered_message for token in ("timeout", "timed out", "connection", "network", "dns")):
        return "The app could not reach Gemini. Check your internet connection and try again."
    if message:
        return f"Gemini request failed: {message}"

    return f"Gemini request failed: {type(error).__name__}"


def _generate_gemini_content(*, model, prompt, use_search=False):
    _ensure_gemini_configuration()

    if genai is None or genai_types is None:
        raise RuntimeError(
            "Gemini SDK is not installed. Install dependencies with `pip install -r requirements.txt` before running analysis."
        )

    config_kwargs = {
        "thinkingConfig": genai_types.ThinkingConfig(
            thinkingLevel=genai_types.ThinkingLevel.HIGH,
        )
    }
    if use_search:
        config_kwargs["tools"] = [genai_types.Tool(googleSearch=genai_types.GoogleSearch())]

    config = genai_types.GenerateContentConfig(**config_kwargs) if config_kwargs else None
    client = genai.Client(api_key=_get_gemini_api_key())
    contents = [
        genai_types.Content(
            role="user",
            parts=[genai_types.Part.from_text(text=str(prompt))],
        )
    ]

    started = time.perf_counter()
    succeeded = False
    try:
        with _GEMINI_CONCURRENCY_SEMAPHORE:
            response = client.models.generate_content(
                model=_resolve_gemini_model_name(model),
                contents=contents,
                config=config,
            )
        succeeded = True
        return _extract_gemini_text(response)
    except Exception as error:
        raise RuntimeError(_format_gemini_exception(error)) from error
    finally:
        observe_llm_call("gemini", time.perf_counter() - started, success=succeeded)


def get_openai_configuration_status():
    return get_gemini_configuration_status()


def _ensure_openai_configuration():
    _ensure_gemini_configuration()


def _get_openai_api_key():
    return _get_gemini_api_key()


def _get_openai_api_key_source():
    return _get_gemini_api_key_source()


def _read_template(filename):
    return (BASE_DIR / filename).read_text(encoding="utf-8")


def _fill_template(template, values):
    for key, value in values.items():
        template = template.replace(f"{{{key}}}", value)
    return template


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]

    if level != 0 or root not in ALLOWED_IMPORT_ROOTS:
        raise ImportError(f"Import of '{name}' is not allowed")

    return builtins.__import__(name, globals, locals, fromlist, level)


def get_default_test_cases():
    return [dict(test_case) for test_case in DEFAULT_TEST_CASES]


def sanitize_query(query):
    return query.strip().lower()


def log_step(query, code, error):
    with LOG_PATH.open("a", encoding="utf-8") as file_handle:
        file_handle.write(f"\nQUERY: {query}\n")
        file_handle.write(f"CODE:\n{code}\n")
        file_handle.write(f"ERROR: {error}\n")


def build_df_info(df):
    return (
        f"Columns: {list(df.columns)}\n"
        f"Dtypes: {df.dtypes.astype(str).to_dict()}\n"
        f"Sample:\n{df.head().to_string()}"
    )


def _format_dataframe_context(df):
    return {
        "df_head": df.head().to_string(),
        "df_columns": "\n".join(str(column) for column in df.columns),
        "df_dtypes": df.dtypes.astype(str).to_string(),
        "df_shape": str(df.shape),
    }


def _format_result(result):
    if hasattr(result, "to_string"):
        try:
            return result.to_string()
        except Exception:
            pass

    return str(result)


def _normalize_column_token(value):
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _clean_column_query_term(term):
    cleaned = str(term or "").strip().lower()
    cleaned = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", cleaned)
    cleaned = re.sub(r"^(the|a|an)\s+", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _find_matching_column(term, df):
    cleaned_term = _clean_column_query_term(term)
    normalized_term = _normalize_column_token(cleaned_term)
    if not normalized_term:
        return None

    exact_matches = []
    contains_matches = []
    reversed_contains_matches = []

    for column in df.columns:
        column_name = str(column)
        normalized_column = _normalize_column_token(column_name)
        if not normalized_column:
            continue
        if normalized_column == normalized_term:
            exact_matches.append(column_name)
        elif normalized_term in normalized_column:
            contains_matches.append(column_name)
        elif normalized_column in normalized_term:
            reversed_contains_matches.append(column_name)

    if exact_matches:
        return exact_matches[0]
    if contains_matches:
        return contains_matches[0]
    if reversed_contains_matches:
        return reversed_contains_matches[0]
    return None


def _build_top_rows_code(row_count):
    safe_count = max(1, int(row_count))
    return f"result = df.head({safe_count}).copy()"


def _build_average_by_group_code(metric_column, group_column):
    average_label = f"average_{_normalize_column_token(metric_column) or 'value'}"
    return "\n".join(
        (
            "import pandas as pd",
            "",
            f"metric_column = {metric_column!r}",
            f"group_column = {group_column!r}",
            "",
            "if group_column not in df.columns:",
            '    raise ValueError(f"Required column {group_column!r} is missing from the dataframe.")',
            "if metric_column not in df.columns:",
            '    raise ValueError(f"Required column {metric_column!r} is missing from the dataframe.")',
            "",
            "metric_values = pd.to_numeric(df[metric_column], errors='coerce')",
            "analysis_df = df.assign(_aidssist_metric_value=metric_values).dropna(subset=[group_column, '_aidssist_metric_value'])",
            "if analysis_df.empty:",
            '    raise ValueError("No valid rows are available after coercing the metric column to numeric.")',
            "",
            "result = (",
            "    analysis_df.groupby(group_column, dropna=False)['_aidssist_metric_value']",
            "    .mean()",
            "    .sort_values(ascending=False)",
            f"    .reset_index(name={average_label!r})",
            ")",
        )
    )


def _get_deterministic_general_code(user_query, df):
    normalized_query = sanitize_query(user_query)

    top_rows_match = re.search(r"\btop\s+(\d+)\s+rows?\b", normalized_query)
    if top_rows_match:
        row_count = int(top_rows_match.group(1))
        return {
            "label": f"top-{row_count}-rows shortcut",
            "code": _build_top_rows_code(row_count),
        }

    average_by_match = re.search(
        r"\b(?:average|avg|mean)\s+(?P<metric>.+?)\s+by\s+(?P<group>[a-z0-9_ \-]+)\b",
        normalized_query,
    )
    if average_by_match:
        metric_column = _find_matching_column(average_by_match.group("metric"), df)
        group_column = _find_matching_column(average_by_match.group("group"), df)
        if metric_column and group_column and metric_column != group_column:
            return {
                "label": f"average-by-group shortcut ({metric_column} by {group_column})",
                "code": _build_average_by_group_code(metric_column, group_column),
            }

    return None


def _validation_is_incorrect(validation):
    for line in str(validation).splitlines():
        normalized = line.strip().upper()
        if normalized:
            return normalized == "INCORRECT"
    return False


def _query_is_valid(validation_text):
    for line in str(validation_text).splitlines():
        normalized = line.strip().upper()
        if normalized:
            return normalized == "VALID"
    return False


def _validation_reason(validation_text):
    lines = [line.strip() for line in str(validation_text).splitlines() if line.strip()]
    if len(lines) >= 2:
        return lines[1]
    if lines:
        return lines[0]
    return "No validation reason was produced."


def _is_complex_query(user_query, operation):
    query = str(user_query).lower()
    keywords = (
        "compare",
        "forecast",
        "predict",
        "segment",
        "correlation",
        "trend",
        "multi",
        "breakdown",
    )
    complexity_markers = query.count(" and ") + query.count(" then ") + query.count(",")

    return (
        str(operation).strip().upper() == "MACHINE_LEARNING"
        or len(query.split()) > 12
        or complexity_markers >= 2
        or any(keyword in query for keyword in keywords)
    )


def _prepare_code_for_execution(generated_code, model=DEFAULT_GEMINI_MODEL):
    optimized_code = optimize_code(generated_code, model=model)
    robust_code = improve_robustness(optimized_code, model=model)

    return optimized_code, robust_code


def _combine_stage_code(*stage_blocks):
    rendered_blocks = []

    for title, code in stage_blocks:
        normalized_code = str(code or "").strip()
        if not normalized_code:
            continue
        rendered_blocks.append(f"# {title}\n{normalized_code}")

    return "\n\n".join(rendered_blocks) or None


def _coerce_forecast_dataframe(prepared_result):
    if isinstance(prepared_result, pd.DataFrame):
        return prepared_result

    if isinstance(prepared_result, pd.Series):
        column_name = prepared_result.name or "sales"
        return prepared_result.rename(column_name).reset_index()

    return None


def _render_template_prompt(filename, values):
    return _fill_template(_read_template(filename), values)


def _run_groq_template_prompt(filename, values, model=DEFAULT_GEMINI_MODEL):
    prompt = _render_template_prompt(filename, values)
    return _generate_groq_content(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    ).strip()


def _run_gemini_template_prompt(
    filename,
    values,
    model=DEFAULT_GEMINI_MODEL,
    use_search=False,
):
    prompt = _render_template_prompt(filename, values)
    return _generate_gemini_content(
        model=model,
        prompt=prompt,
        use_search=use_search,
    ).strip()


def rewrite_query(user_query, df, model=DEFAULT_GEMINI_MODEL):
    return _run_gemini_template_prompt(
        "query_rewrite_prompt.txt",
        {
            "user_query": str(user_query),
            "df_columns": "\n".join(str(column) for column in df.columns),
            "df_head": df.head().to_string(),
        },
        model=model,
    )


def decide_operation(user_query, model=DEFAULT_GEMINI_MODEL):
    return _run_gemini_template_prompt(
        "decision_prompt.txt",
        {
            "user_query": str(user_query),
        },
        model=model,
    )


def validate_query(user_query, model=DEFAULT_GEMINI_MODEL):
    return _run_gemini_template_prompt(
        "query_validation_prompt.txt",
        {
            "user_query": str(user_query),
        },
        model=model,
    )


def optimize_code(generated_code, model=DEFAULT_GEMINI_MODEL):
    return _run_groq_template_prompt(
        "optimize_code_prompt.txt",
        {
            "generated_code": str(generated_code),
        },
        model=model,
    )


def improve_robustness(generated_code, model=DEFAULT_GEMINI_MODEL):
    return _run_groq_template_prompt(
        "robustness_prompt.txt",
        {
            "generated_code": str(generated_code),
        },
        model=model,
    )


def break_down_task(user_query, model=DEFAULT_GEMINI_MODEL):
    return _run_gemini_template_prompt(
        "breakdown_prompt.txt",
        {
            "user_query": str(user_query),
        },
        model=model,
    )


def simplify_complex_query(user_query, model=DEFAULT_GEMINI_MODEL):
    return _run_gemini_template_prompt(
        "complexity_control_prompt.txt",
        {
            "user_query": str(user_query),
        },
        model=model,
    )


def evaluate_confidence(user_query, result, model=DEFAULT_GEMINI_MODEL):
    return _run_gemini_template_prompt(
        "confidence_prompt.txt",
        {
            "user_query": str(user_query),
            "result": _format_result(result),
        },
        model=model,
    )


def evaluate_system_performance(user_query, expected_output, result, model=DEFAULT_GEMINI_MODEL):
    return _run_gemini_template_prompt(
        "performance_evaluation_prompt.txt",
        {
            "user_query": str(user_query),
            "expected_output": str(expected_output),
            "result": _format_result(result),
        },
        model=model,
    )


def analyze_system_weaknesses(user_query, result, model=DEFAULT_GEMINI_MODEL):
    return _run_gemini_template_prompt(
        "system_weakness_prompt.txt",
        {
            "user_query": str(user_query),
            "result": _format_result(result),
        },
        model=model,
    )


def diagnose_system_failure(user_query, result, model=DEFAULT_GEMINI_MODEL):
    return _run_gemini_template_prompt(
        "failure_diagnosis_prompt.txt",
        {
            "user_query": str(user_query),
            "result": _format_result(result),
        },
        model=model,
    )


def evaluate_test_performance(user_query, result, model=DEFAULT_GEMINI_MODEL):
    return _run_gemini_template_prompt(
        "test_performance_prompt.txt",
        {
            "user_query": str(user_query),
            "result": _format_result(result),
        },
        model=model,
    )


def evaluate_result_risk(result, model=DEFAULT_GEMINI_MODEL):
    return _run_gemini_template_prompt(
        "result_risk_prompt.txt",
        {
            "result": _format_result(result),
        },
        model=model,
    )


def create_demo_script(features, model=DEFAULT_GEMINI_MODEL):
    return _run_gemini_template_prompt(
        "demo_script_prompt.txt",
        {
            "features": str(features),
        },
        model=model,
    )


def create_resume_description(project, features, model=DEFAULT_GEMINI_MODEL):
    return _run_gemini_template_prompt(
        "resume_description_prompt.txt",
        {
            "project": str(project),
            "features": str(features),
        },
        model=model,
    )


def create_github_readme(project, model=DEFAULT_GEMINI_MODEL):
    return _run_gemini_template_prompt(
        "github_readme_prompt.txt",
        {
            "project": str(project),
        },
        model=model,
    )


def identify_use_cases(features, model=DEFAULT_GEMINI_MODEL):
    return _run_gemini_template_prompt(
        "use_cases_prompt.txt",
        {
            "features": str(features),
        },
        model=model,
    )


def identify_unique_selling_points(system, model=DEFAULT_GEMINI_MODEL):
    return _run_gemini_template_prompt(
        "product_differentiation_prompt.txt",
        {
            "system": str(system),
        },
        model=model,
    )


def format_response(user_query, result, model=DEFAULT_GEMINI_MODEL):
    return _run_gemini_template_prompt(
        "response_format_prompt.txt",
        {
            "user_query": str(user_query),
            "result": _format_result(result),
        },
        model=model,
    )


def generate_power_bi_brief(user_query, result, df, model=DEFAULT_GEMINI_MODEL):
    return _run_gemini_template_prompt(
        "power_bi_brief_prompt.txt",
        {
            "user_query": str(user_query),
            "result": _format_result(result),
            "df_columns": "\n".join(str(column) for column in df.columns),
            "df_dtypes": df.dtypes.astype(str).to_string(),
            "df_shape": str(df.shape),
        },
        model=model,
        use_search=True,
    )


def recommend_business_actions(user_query, result, model=DEFAULT_GEMINI_MODEL):
    return _run_gemini_template_prompt(
        "business_strategy_prompt.txt",
        {
            "user_query": str(user_query),
            "result": _format_result(result),
        },
        model=model,
        use_search=True,
    )


def tell_data_story(user_query, result, model=DEFAULT_GEMINI_MODEL):
    return _run_gemini_template_prompt(
        "storytelling_prompt.txt",
        {
            "user_query": str(user_query),
            "result": _format_result(result),
        },
        model=model,
        use_search=True,
    )


def compare_groups(user_query, df, model=DEFAULT_GEMINI_MODEL):
    return _run_gemini_template_prompt(
        "comparison_prompt.txt",
        {
            "user_query": str(user_query),
            "df_head": df.head().to_string(),
        },
        model=model,
    )


def suggest_root_causes(result, df, model=DEFAULT_GEMINI_MODEL):
    return _run_gemini_template_prompt(
        "root_cause_prompt.txt",
        {
            "result": _format_result(result),
            "df_head": df.head().to_string(),
        },
        model=model,
        use_search=True,
    )


def analyze_scenarios(user_query, result, model=DEFAULT_GEMINI_MODEL):
    return _run_gemini_template_prompt(
        "scenario_prompt.txt",
        {
            "user_query": str(user_query),
            "result": _format_result(result),
        },
        model=model,
        use_search=True,
    )


def adapt_response_by_user_type(user_type, result, model=DEFAULT_GEMINI_MODEL):
    return _run_gemini_template_prompt(
        "adaptive_response_prompt.txt",
        {
            "user_type": str(user_type),
            "result": _format_result(result),
        },
        model=model,
    )


def prioritize_insights(insights, model=DEFAULT_GEMINI_MODEL):
    return _run_gemini_template_prompt(
        "insight_priority_prompt.txt",
        {
            "insights": str(insights),
        },
        model=model,
    )


def summarize_for_non_technical_user(result, model=DEFAULT_GEMINI_MODEL):
    return _run_gemini_template_prompt(
        "simple_summary_prompt.txt",
        {
            "result": _format_result(result),
        },
        model=model,
    )


def suggest_follow_up_questions(user_query, result, model=DEFAULT_GEMINI_MODEL):
    return _run_gemini_template_prompt(
        "follow_up_prompt.txt",
        {
            "user_query": str(user_query),
            "result": _format_result(result),
        },
        model=model,
    )


def explain_error_simple(error_message, model=DEFAULT_GEMINI_MODEL):
    return _run_gemini_template_prompt(
        "error_explanation_prompt.txt",
        {
            "error_message": str(error_message),
        },
        model=model,
    )


def handle_system_failure(error_message, user_query, model=DEFAULT_GEMINI_MODEL):
    return _run_gemini_template_prompt(
        "failure_help_prompt.txt",
        {
            "error_message": str(error_message),
            "user_query": str(user_query),
        },
        model=model,
    )


def assess_data_quality(df, model=DEFAULT_GEMINI_MODEL):
    return _run_gemini_template_prompt(
        "data_quality_prompt.txt",
        {
            "df_head": df.head().to_string(),
        },
        model=model,
    )


def explain_solution_steps(user_query, generated_code, result, model=DEFAULT_GEMINI_MODEL):
    return _run_gemini_template_prompt(
        "solution_explanation_prompt.txt",
        {
            "user_query": str(user_query),
            "generated_code": str(generated_code),
            "result": _format_result(result),
        },
        model=model,
    )


def detect_anomalies(df, model=DEFAULT_GEMINI_MODEL):
    return _run_gemini_template_prompt(
        "anomaly_prompt.txt",
        {
            "df_head": df.head().to_string(),
        },
        model=model,
    )


def suggest_model_features(df, model=DEFAULT_GEMINI_MODEL):
    return _run_gemini_template_prompt(
        "feature_suggestion_prompt.txt",
        {
            "df_head": df.head().to_string(),
            "df_columns": "\n".join(str(column) for column in df.columns),
            "df_dtypes": df.dtypes.astype(str).to_string(),
        },
        model=model,
    )


def generate_ratings_analysis_code(df, model=DEFAULT_GEMINI_MODEL):
    return _run_gemini_template_prompt(
        "ratings_analysis_prompt.txt",
        {
            **_format_dataframe_context(df),
        },
        model=model,
    )


def generate_forecast_prep_code(df, model=DEFAULT_GEMINI_MODEL):
    return _run_gemini_template_prompt(
        "forecast_prep_prompt.txt",
        {
            **_format_dataframe_context(df),
        },
        model=model,
    )


def generate_sales_prediction_code(df, model=DEFAULT_GEMINI_MODEL):
    return _run_gemini_template_prompt(
        "sales_prediction_prompt.txt",
        {
            **_format_dataframe_context(df),
        },
        model=model,
    )


def validate_ratings_dataset(df, model=DEFAULT_GEMINI_MODEL):
    return _run_gemini_template_prompt(
        "ratings_validation_prompt.txt",
        {
            **_format_dataframe_context(df),
        },
        model=model,
    )


def validate_forecast_dataset(df, model=DEFAULT_GEMINI_MODEL):
    return _run_gemini_template_prompt(
        "forecast_validation_prompt.txt",
        {
            **_format_dataframe_context(df),
        },
        model=model,
    )


def detect_intent(query):
    normalized_query = str(query).lower()

    if "rating" in normalized_query:
        return "rating"
    if "predict" in normalized_query or "forecast" in normalized_query:
        return "forecast"
    return "general"


def detect_analysis_intent(user_query):
    simple_intent = detect_intent(user_query)

    if simple_intent == "rating":
        return "RATINGS"
    if simple_intent == "forecast":
        return "FORECAST"
    return "GENERAL"


def _normalize_system_decision_payload(decision):
    payload = dict(decision or {})
    selected_mode = str(
        payload.get("selected_mode")
        or payload.get("mode")
        or "analysis"
    ).strip().lower()
    if selected_mode not in {"forecast", "ml", "analysis"}:
        selected_mode = "analysis"

    reason = str(payload.get("reason") or "").strip()
    suggestion = str(payload.get("suggestion") or "").strip()
    confidence = payload.get("confidence")
    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError):
        confidence_value = 0.0
    confidence_value = max(0.0, min(confidence_value, 1.0))

    if not suggestion:
        suggestion = {
            "forecast": "Using forecasting",
            "ml": "Switching to predictive modeling",
            "analysis": "Using aggregation and exploratory analysis",
        }.get(selected_mode, "Using analysis")

    return {
        "selected_mode": selected_mode,
        "reason": reason,
        "suggestion": suggestion,
        "confidence": confidence_value,
    }


def _build_system_decision(df, user_query, routing_override=None):
    if routing_override is not None:
        return _normalize_system_decision_payload(routing_override)

    decision = decide_analysis_mode(df, user_query)
    return _normalize_system_decision_payload(decision)


def _apply_system_mode_to_plan(plan, system_decision, dataset_intelligence):
    updated_plan = dict(plan or {})
    selected_mode = str((system_decision or {}).get("selected_mode") or "analysis").strip().lower()

    updated_plan["dataset_intelligence"] = dict(dataset_intelligence or {})
    updated_plan["system_decision"] = dict(system_decision or {})

    if selected_mode == "forecast":
        updated_plan["analysis_type"] = "time_series"
        updated_plan["analysis_route"] = "forecasting.py"
        updated_plan["method"] = "forecasting_pipeline"
    elif selected_mode == "ml":
        updated_plan["analysis_type"] = "ml"
        updated_plan["analysis_route"] = "ml_pipeline"
        if str(updated_plan.get("intent") or "").strip().lower() == "prediction":
            updated_plan["method"] = (
                "deterministic_prediction"
                if updated_plan.get("target_column")
                else "llm_assisted"
            )
    else:
        if str(updated_plan.get("intent") or "").strip().lower() == "prediction":
            updated_plan["analysis_type"] = "general"
            updated_plan["analysis_route"] = "pandas_analysis"
            updated_plan["method"] = "deterministic" if (
                updated_plan.get("metric_column") or updated_plan.get("group_column")
            ) else "llm_assisted"

    return updated_plan


def _build_forecast_fallback_override(failure_reason):
    return _normalize_system_decision_payload(
        {
            "mode": "ml",
            "reason": (
                "Forecasting could not run safely because "
                f"{str(failure_reason or 'the dataset is not forecast-ready').strip()}"
            ),
            "suggestion": "Switching to predictive modeling",
            "confidence": 0.82,
        }
    )


def _coerce_contract_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        lines = [line.strip(" -•\t") for line in value.splitlines() if line.strip()]
        if lines:
            return lines
        stripped = value.strip()
        return [stripped] if stripped else []
    return [str(value).strip()]


def _coerce_contract_text(value):
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, list):
        lines = [str(item).strip() for item in value if str(item).strip()]
        if not lines:
            return None
        if len(lines) == 1:
            return lines[0]
        return "\n".join(f"- {line}" for line in lines) or None
    return str(value).strip() or None


def _record_failure(failure_events, query, error, stage, *, store=None, metadata=None):
    record = log_failure(
        query,
        error,
        stage,
        store=store,
        metadata=metadata,
    )
    failure_events.append(record)
    return record


def _normalize_contract_payload(contract):
    return ensure_analysis_contract_defaults(contract)


def _empty_execution_plan():
    return {
        "sql_plan": None,
        "python_steps": [],
        "excel_logic": {},
        "fallback_reason": None,
    }


def _normalize_execution_plan_output(value):
    return list(ensure_analysis_contract_defaults({"execution_plan": value}).get("execution_plan") or [])


def _normalize_execution_trace_output(execution_plan, execution_trace):
    return list(
        ensure_analysis_contract_defaults(
            {
                "execution_plan": execution_plan,
                "execution_trace": execution_trace,
            }
        ).get("execution_trace")
        or []
    )


def _normalize_optimization_output(execution_trace, optimization):
    return dict(
        ensure_analysis_contract_defaults(
            {
                "execution_trace": execution_trace,
                "optimization": optimization,
            }
        ).get("optimization")
        or {
            "execution_time_total": 0,
            "cost_estimate": "low",
            "optimized": False,
            "parallel_execution": False,
            "plans_considered": 1,
            "selected_plan_score": 0.0,
            "constraints_applied": {},
        }
    )


def _plan_uses_python(plan):
    execution_plan = _normalize_execution_plan_output((plan or {}).get("execution_plan"))
    if any(str(step.get("tool") or "").strip().upper() == "PYTHON" for step in execution_plan):
        return True
    return str((plan or {}).get("tool_used") or "PYTHON").strip().upper() == "PYTHON"


def _execution_route_label(execution_plan):
    steps = _normalize_execution_plan_output(execution_plan)
    if not steps:
        return "No execution steps planned."
    return " -> ".join(
        f"{step.get('tool')}: {step.get('task')}"
        for step in steps
    )


def _render_sql_plan_code(sql_plan):
    normalized_plan = str(sql_plan or "").strip()
    if not normalized_plan:
        normalized_plan = "SELECT * FROM df"
    return f"-- SQL simulation\n{normalized_plan}"


def _render_excel_logic_code(excel_analysis):
    payload = dict(excel_analysis or {})
    summary = payload.get("summary") or {}
    if isinstance(summary, dict):
        summary_line = ", ".join(
            f"{key}={value}"
            for key, value in summary.items()
            if value is not None and str(value).strip()
        )
    else:
        summary_line = str(summary or "").strip()
    rendered = {
        "pivot_table": payload.get("pivot_table") or {},
        "aggregations": payload.get("aggregations") or {},
    }
    header = "# Excel simulation"
    if summary_line:
        header += f"\n# {summary_line}"
    return f"{header}\nexcel_logic = {json.dumps(rendered, indent=2, default=str)}"


def _render_dashboard_code(dashboard):
    normalized_dashboard = dict(dashboard or {"charts": [], "kpis": []})
    return "# BI dashboard simulation\ndashboard = " + json.dumps(normalized_dashboard, indent=2, default=str)


def _build_analysis_reliability_payload(
    *,
    query,
    df,
    result,
    plan,
    insights=None,
    workflow_context=None,
    preflight=None,
    data_score=None,
    store=None,
    failure_events=None,
    use_llm_limitations=True,
):
    workflow_context = workflow_context or {}
    failure_events = failure_events or []
    owns_store = False
    if store is None and workflow_context.get("source_fingerprint"):
        try:
            store = WorkflowStore()
            owns_store = True
        except Exception:
            store = None
    reliability_warnings: list[str] = []
    model_metrics = {"mae": None, "r2": None}
    explanation = {"top_features": [], "impact": []}
    ml_intelligence = None
    data_quality = build_data_quality_report(df)
    resolved_tool_used = str(plan.get("tool_used") or "PYTHON").strip().upper()
    model_quality = "weak"
    risk = build_risk_statement(data_quality, model_quality)
    uses_python = _plan_uses_python(plan)
    if not uses_python:
        model_quality = "not_applicable"
    else:
        try:
            raw_ml_output = build_ml_intelligence(
                df,
                user_query=query,
                target_hint=plan.get("target_column") or plan.get("metric_column"),
                insights=list(insights or []),
            )
            ml_intelligence = postprocess_ml_output(raw_ml_output, df)
            validate_ml_output(ml_intelligence)
        except Exception as error:
            ml_intelligence = {
                "error": str(error),
                "fallback": "analysis_mode",
            }
            reliability_warnings.append(f"ML intelligence fallback activated: {error}")

        ml_metrics = dict((ml_intelligence or {}).get("metrics") or {})
        if ml_metrics:
            model_metrics = {
                "mae": ml_metrics.get("mae"),
                "r2": ml_metrics.get("r2"),
                "accuracy": ml_metrics.get("accuracy"),
            }
        top_features = list((ml_intelligence or {}).get("top_features") or [])
        feature_importance = dict((ml_intelligence or {}).get("feature_importance") or {})
        if top_features and feature_importance:
            explanation = {
                "top_features": top_features,
                "impact": [float(feature_importance.get(feature) or 0.0) for feature in top_features],
            }
        reliability_warnings.extend(list((ml_intelligence or {}).get("warnings") or []))
        model_quality = interpret_model_quality(model_metrics.get("mae"), model_metrics.get("r2"))

        if str(plan.get("intent") or "") == "prediction" and not list((ml_intelligence or {}).get("top_features") or []):
            diagnostics = build_simple_prediction_diagnostics(
                df,
                target_column=plan.get("target_column") or plan.get("metric_column"),
                datetime_column=plan.get("datetime_column"),
            )
            model_metrics = diagnostics.get("model_metrics") or model_metrics
            explanation = diagnostics.get("explanation") or explanation
            reliability_warnings.extend(diagnostics.get("warnings") or [])
            model_quality = str(diagnostics.get("model_quality") or model_quality)

    risk = build_risk_statement(data_quality, model_quality)

    consistency = build_analysis_consistency(
        store=store,
        result=result,
        source_fingerprint=workflow_context.get("source_fingerprint"),
        query=query,
        analysis_intent=plan.get("intent"),
    )
    limitations = build_limitations(
        query=query,
        result=result,
        df=df,
        warnings=list((preflight or {}).get("warnings") or []) + reliability_warnings,
        data_score=data_score,
        data_quality=data_quality,
        model_metrics=model_metrics,
        model_quality=model_quality,
        risk=risk,
        explanation=explanation,
        inconsistency_detected=bool(consistency.get("inconsistency_detected")),
        analysis_type=plan.get("analysis_type"),
        use_llm=use_llm_limitations,
        store=store,
        metadata={
            "source_fingerprint": workflow_context.get("source_fingerprint"),
            "dataset_id": workflow_context.get("dataset_id"),
        },
    )
    reproducibility = build_reproducibility_metadata(
        source_fingerprint=workflow_context.get("source_fingerprint"),
        pipeline_trace=[],
        result_hash=consistency.get("result_hash"),
        consistency_payload=consistency,
    )
    learning_patterns = get_learning_patterns(
        store,
        workflow_context.get("source_fingerprint"),
    ) if store is not None else {}
    if len(failure_events) == 0:
        # Keep the function pure for callers that do not want local tracking.
        pass
    payload = {
        "data_quality": data_quality,
        "model_metrics": model_metrics,
        "explanation": explanation,
        "ml_intelligence": ml_intelligence or {},
        "model_quality": model_quality,
        "risk": risk,
        "reliability_warnings": list(dict.fromkeys(reliability_warnings)),
        "result_hash": consistency.get("result_hash", ""),
        "inconsistency_detected": bool(consistency.get("inconsistency_detected")),
        "limitations": limitations,
        "dataset_fingerprint": str(workflow_context.get("source_fingerprint") or ""),
        "reproducibility": reproducibility,
        "failure_patterns": get_failure_patterns(),
        "learning_patterns": learning_patterns,
    }
    if owns_store and store is not None:
        store.close()
    return payload


def _apply_analysis_contract(output, contract):
    contract = _normalize_contract_payload(contract)
    output["analysis_contract"] = contract
    output["system_decision"] = dict(contract.get("system_decision") or {})
    output["summary"] = contract.get("result_summary")
    output["insights"] = _coerce_contract_text(contract.get("insights"))
    output["recommendations"] = _coerce_contract_list(contract.get("recommendations"))
    output["warnings"] = _coerce_contract_list(contract.get("warnings"))
    output["confidence"] = str(contract.get("confidence") or "")
    output["decision_layer"] = dict(
        contract.get("decision_layer")
        or {
            "decisions": [],
            "top_decision": None,
            "decision_confidence": "low",
            "risk_summary": str(contract.get("risk") or ""),
        }
    )
    output["business_decisions"] = _coerce_contract_text(contract.get("recommendations"))
    output["data_quality"] = dict(
        contract.get("data_quality")
        or {
            "score": 0.0,
            "data_quality_score": 0.0,
            "issues": [],
            "warnings": [],
            "profile": {},
            "data_profile": {},
            "anomalies": {},
        }
    )
    output["cleaning_report"] = dict(contract.get("cleaning_report") or {})
    output["model_metrics"] = dict(contract.get("model_metrics") or {"mae": None, "r2": None})
    output["explanation"] = dict(contract.get("explanation") or {"top_features": [], "impact": []})
    output["ml_intelligence"] = dict(contract.get("ml_intelligence") or {})
    output["model_quality"] = str(contract.get("model_quality") or "weak")
    output["risk"] = str(contract.get("risk") or "")
    output["result_hash"] = str(contract.get("result_hash") or "")
    output["dataset_fingerprint"] = str(contract.get("dataset_fingerprint") or "")
    output["tool_used"] = str(contract.get("tool_used") or "PYTHON")
    output["analysis_mode"] = str(contract.get("analysis_mode") or "ad-hoc")
    output["execution_plan"] = list(contract.get("execution_plan") or [])
    output["execution_trace"] = list(contract.get("execution_trace") or [])
    output["optimization"] = dict(contract.get("optimization") or {})
    output["excel_analysis"] = dict(contract.get("excel_analysis") or {}) or None
    output["dashboard"] = dict(contract.get("dashboard") or {}) or None
    output["forecast_metadata"] = dict(contract.get("forecast_metadata") or {}) or None
    output["context"] = dict(contract.get("context") or {}) or None
    output["suggestions"] = list(contract.get("suggestions") or [])
    output["recommended_next_step"] = str(contract.get("recommended_next_step") or "") or None
    output["suggested_questions"] = _coerce_contract_list(contract.get("suggested_questions"))
    output["active_filter"] = str(contract.get("active_filter") or "") or None
    output["visualization_type"] = str(contract.get("visualization_type") or "") or None
    output["reproducibility"] = dict(
        contract.get("reproducibility")
        or {
            "dataset_fingerprint": "",
            "pipeline_trace_hash": "",
            "result_hash": "",
            "consistent_with_prior_runs": True,
            "prior_hash_count": 0,
            "consistency_validated": False,
        }
    )
    output["inconsistency_detected"] = bool(contract.get("inconsistency_detected"))
    output["limitations"] = _coerce_contract_list(contract.get("limitations"))
    output["failure_patterns"] = dict(output.get("failure_patterns") or get_failure_patterns())
    return output


def _build_analysis_contract_payload(
    *,
    query,
    df,
    result,
    code,
    plan,
    preflight,
    method,
    error=None,
    summary_override=None,
    insights_override=None,
    workflow_context=None,
    data_score=None,
    store=None,
    reliability_payload=None,
    tool_used=None,
    analysis_mode=None,
    execution_plan=None,
    execution_trace=None,
    optimization=None,
    excel_analysis=None,
    dashboard=None,
    forecast_metadata=None,
    system_decision=None,
    context=None,
    suggestions=None,
    recommended_next_step=None,
    suggested_questions=None,
    active_filter=None,
    visualization_type=None,
):
    reliability_payload = dict(reliability_payload or {})
    merged_preflight = dict(preflight or {})
    merged_preflight["warnings"] = list(
        dict.fromkeys(
            _coerce_contract_list((preflight or {}).get("warnings"))
            + _coerce_contract_list(reliability_payload.get("reliability_warnings"))
        )
    )
    contract = build_analysis_contract(
        query=query,
        df=df,
        result=result,
        executed_code=code,
        plan=plan,
        preflight=merged_preflight,
        method=method,
        data_quality=reliability_payload.get("data_quality"),
        cleaning_report=(workflow_context or {}).get("cleaning_report"),
        model_metrics=reliability_payload.get("model_metrics"),
        explanation=reliability_payload.get("explanation"),
        ml_intelligence=reliability_payload.get("ml_intelligence"),
        model_quality=reliability_payload.get("model_quality"),
        risk=reliability_payload.get("risk"),
        result_hash=reliability_payload.get("result_hash"),
        dataset_fingerprint=reliability_payload.get("dataset_fingerprint"),
        reproducibility=reliability_payload.get("reproducibility"),
        inconsistency_detected=reliability_payload.get("inconsistency_detected", False),
        limitations=reliability_payload.get("limitations"),
        insights=_coerce_contract_list(insights_override) if insights_override else None,
        learning_patterns=reliability_payload.get("learning_patterns"),
        tool_used=tool_used,
        analysis_mode=analysis_mode,
        execution_plan=execution_plan,
        execution_trace=execution_trace,
        optimization=optimization,
        excel_analysis=excel_analysis,
        dashboard=dashboard,
        forecast_metadata=forecast_metadata,
        system_decision=system_decision,
        context=context,
        suggestions=suggestions,
        recommended_next_step=recommended_next_step,
        suggested_questions=suggested_questions,
        active_filter=active_filter,
        visualization_type=visualization_type,
    )

    if error:
        warnings = _coerce_contract_list(contract.get("warnings"))
        warnings.append(str(error))
        contract["warnings"] = list(dict.fromkeys(warnings))
        contract["result_summary"] = (
            f"The analysis could not be completed reliably for '{query}'. "
            f"Reason: {error}"
        )

    if summary_override:
        contract["result_summary"] = str(summary_override).strip()

    if insights_override:
        contract["insights"] = _coerce_contract_list(insights_override)

    if not contract.get("recommendations"):
        contract["recommendations"] = []
    if not contract.get("warnings"):
        contract["warnings"] = []

    contract["warnings"] = list(dict.fromkeys(_coerce_contract_list(contract.get("warnings"))))
    contract["reproducibility"] = build_reproducibility_metadata(
        source_fingerprint=contract.get("dataset_fingerprint") or reliability_payload.get("dataset_fingerprint"),
        pipeline_trace=[],
        result_hash=contract.get("result_hash"),
        consistency_payload={
            "result_hash": contract.get("result_hash"),
            "inconsistency_detected": contract.get("inconsistency_detected"),
            "prior_hash_count": (contract.get("reproducibility") or {}).get("prior_hash_count"),
            "consistency_validated": (contract.get("reproducibility") or {}).get("consistency_validated"),
        },
    )
    return _normalize_contract_payload(contract)


def _build_data_score(preflight, *, error=None):
    warning_count = len(preflight.get("warnings") or [])
    blocking_count = len(preflight.get("blocking_errors") or [])
    base_quality_score = float(((preflight.get("data_quality") or {}).get("score")) or 10.0) * 10.0
    score = min(100, base_quality_score)
    score -= (warning_count * 4) + (blocking_count * 20)
    if error:
        score -= 10
    score = max(0, min(100, score))
    if score >= 85:
        band = "strong"
    elif score >= 70:
        band = "usable"
    elif score >= 50:
        band = "watch"
    else:
        band = "weak"
    return {
        "score": int(score),
        "band": band,
        "warning_count": warning_count,
        "blocking_issue_count": blocking_count,
    }


def _build_pipeline_trace(
    *,
    query,
    detected_intent,
    analysis_plan,
    analysis_contract,
    preflight,
    method,
    error=None,
    cache_state=None,
    memory_state=None,
    data_quality=None,
    model_metrics=None,
    model_quality=None,
    risk=None,
    explanation=None,
    limitations=None,
    logged_failure_count=0,
    inconsistency_detected=False,
):
    analysis_intent = str((analysis_contract or {}).get("intent") or analysis_plan.get("intent") or "analysis")
    tool_used = str((analysis_contract or {}).get("tool_used") or analysis_plan.get("tool_used") or "PYTHON").strip().upper()
    analysis_mode = str((analysis_contract or {}).get("analysis_mode") or analysis_plan.get("analysis_mode") or "ad-hoc").strip().lower()
    system_decision = dict((analysis_contract or {}).get("system_decision") or analysis_plan.get("system_decision") or {})
    selected_mode = str(system_decision.get("selected_mode") or "").strip().lower()
    execution_plan = list((analysis_contract or {}).get("execution_plan") or _normalize_execution_plan_output(analysis_plan.get("execution_plan")))
    execution_trace = list((analysis_contract or {}).get("execution_trace") or [])
    optimization = dict((analysis_contract or {}).get("optimization") or {})
    data_score = _build_data_score(preflight, error=error)
    uses_integrated_ml = (
        any(str(step.get("tool") or "").strip().upper() == "PYTHON" for step in execution_plan)
        and (
            selected_mode in {"forecast", "ml"}
            or detected_intent == "forecast"
            or analysis_intent == "prediction"
            or "prediction" in str(method)
            or "forecast" in str(method)
        )
    )
    ml_intelligence = dict((analysis_contract or {}).get("ml_intelligence") or {})
    execution_status = "completed"
    if error:
        execution_status = "failed"
    elif preflight.get("blocking_errors"):
        execution_status = "skipped"

    insight_status = "completed" if analysis_contract and (
        analysis_contract.get("result_summary")
        or analysis_contract.get("insights")
        or analysis_contract.get("recommendations")
    ) else "pending"

    model_stage_status = "completed" if uses_integrated_ml else "skipped"
    explainability_status = "completed" if uses_integrated_ml else "skipped"
    if uses_integrated_ml and not any(value is not None for value in dict(model_metrics or {}).values()):
        model_stage_status = "skipped"
    if uses_integrated_ml and not list((explanation or {}).get("top_features") or []):
        explainability_status = "skipped"
    failure_logging_status = "completed" if logged_failure_count or error else "skipped"
    consistency_status = "completed" if (analysis_contract or {}).get("result_hash") else "skipped"

    return [
        {
            "stage": "user_query",
            "title": "User Query",
            "status": "completed",
            "detail": {"query": str(query or "").strip()},
        },
        {
            "stage": "intent_detection",
            "title": "Intent Detection",
            "status": "completed",
            "detail": {
                "legacy_intent": detected_intent,
                "analysis_intent": analysis_intent,
                "analysis_type": analysis_plan.get("analysis_type"),
                "analysis_route": analysis_plan.get("analysis_route"),
                "system_decision": system_decision,
            },
        },
        {
            "stage": "contract_execution",
            "title": "Contract Execution",
            "status": "completed",
            "detail": {
                "required_columns": list(analysis_plan.get("required_columns") or []),
                "method": str(method or ""),
                "steps": list(analysis_plan.get("steps") or []),
                "datetime_column": analysis_plan.get("datetime_column"),
                "target_column": analysis_plan.get("target_column"),
                "tool_used": tool_used,
                "analysis_mode": analysis_mode,
                "execution_plan": execution_plan,
                "execution_trace": execution_trace,
                "optimization": optimization,
            },
        },
        {
            "stage": "forecast_ml",
            "title": "Forecast / ML (integrated)",
            "status": "completed" if uses_integrated_ml else "skipped",
            "detail": {
                "enabled": uses_integrated_ml,
                "method": str(method or ""),
                "target_column": analysis_plan.get("target_column"),
                "tool_used": tool_used,
            },
        },
        {
            "stage": "validation_data_score",
            "title": "Validation + Data Score",
            "status": "failed" if preflight.get("blocking_errors") else "completed",
            "detail": {
                "data_score": data_score,
                "data_quality": dict(data_quality or {}),
                "warnings": list(preflight.get("warnings") or []),
                "blocking_errors": list(preflight.get("blocking_errors") or []),
            },
        },
        {
            "stage": "execution",
            "title": "Execution",
            "status": execution_status,
            "detail": {
                "error": error,
            },
        },
        {
            "stage": "model_evaluation",
            "title": "Model Evaluation",
            "status": model_stage_status,
            "detail": {
                "metrics": dict(model_metrics or {"mae": None, "r2": None}),
            },
        },
        {
            "stage": "explainability",
            "title": "Explainability",
            "status": explainability_status,
            "detail": {
                "top_features": list((explanation or {}).get("top_features") or []),
            },
        },
        {
            "stage": "ml_intelligence",
            "title": "ML Intelligence",
            "status": "completed" if ml_intelligence.get("target") else ("skipped" if not uses_integrated_ml else "pending"),
            "detail": {
                "target": ml_intelligence.get("target"),
                "problem_type": ml_intelligence.get("problem_type"),
                "feature_count": len(ml_intelligence.get("features") or []),
                "top_features": list(ml_intelligence.get("top_features") or []),
            },
        },
        {
            "stage": "trust_layer",
            "title": "Trust Layer",
            "status": "completed",
            "detail": {
                "model_quality": str(model_quality or ""),
                "risk": str(risk or ""),
                "data_quality_score": float((data_quality or {}).get("score") or 0.0),
            },
        },
        {
            "stage": "decision_engine",
            "title": "Decision Engine",
            "status": "completed" if ((analysis_contract or {}).get("decision_layer") or {}).get("decisions") else "skipped",
            "detail": {
                "decision_count": len((((analysis_contract or {}).get("decision_layer") or {}).get("decisions") or [])),
                "decision_confidence": str((((analysis_contract or {}).get("decision_layer") or {}).get("decision_confidence") or "low")),
                "top_decision": ((((analysis_contract or {}).get("decision_layer") or {}).get("top_decision") or {}).get("action")),
            },
        },
        {
            "stage": "learning_engine",
            "title": "Learning Engine",
            "status": "completed" if (((analysis_contract or {}).get("decision_layer") or {}).get("learning_insights")) else "skipped",
            "detail": {
                "pattern_count": len(((((analysis_contract or {}).get("decision_layer") or {}).get("learning_insights") or {}).get("patterns") or [])),
                "confidence_adjustment": ((((analysis_contract or {}).get("decision_layer") or {}).get("learning_insights") or {}).get("confidence_adjustment")),
                "risk_adjustment": ((((analysis_contract or {}).get("decision_layer") or {}).get("learning_insights") or {}).get("risk_adjustment")),
            },
        },
        {
            "stage": "insight_decisions",
            "title": "Insight + Decisions",
            "status": insight_status,
            "detail": {
                "summary": (analysis_contract or {}).get("result_summary"),
                "recommendation_count": len((analysis_contract or {}).get("recommendations") or []),
            },
        },
        {
            "stage": "failure_logging",
            "title": "Failure Logging",
            "status": failure_logging_status,
            "detail": {"logged_failures": int(logged_failure_count or 0)},
        },
        {
            "stage": "consistency_check",
            "title": "Consistency Check",
            "status": consistency_status,
            "detail": {
                "result_hash": str((analysis_contract or {}).get("result_hash") or ""),
                "inconsistency_detected": bool(inconsistency_detected),
                "dataset_fingerprint": str((analysis_contract or {}).get("dataset_fingerprint") or ""),
                "limitations": list(limitations or []),
            },
        },
        {
            "stage": "caching",
            "title": "Caching",
            "status": str((cache_state or {}).get("status") or "pending"),
            "detail": dict(cache_state or {"status": "pending"}),
        },
        {
            "stage": "memory_update",
            "title": "Memory Update",
            "status": str((memory_state or {}).get("status") or "pending"),
            "detail": dict(memory_state or {"status": "pending"}),
        },
    ]


def _apply_pipeline_metadata(
    output,
    *,
    query,
    detected_intent,
    analysis_plan,
    preflight,
    method,
    error=None,
    cache_state=None,
    memory_state=None,
):
    analysis_contract = output.get("analysis_contract") or {}
    output["data_score"] = _build_data_score(preflight, error=error)
    output["failure_patterns"] = dict(output.get("failure_patterns") or get_failure_patterns())
    output["cache_status"] = dict(cache_state or {"status": "pending"})
    output["memory_update"] = dict(memory_state or {"status": "pending"})
    output["pipeline_trace"] = _build_pipeline_trace(
        query=query,
        detected_intent=detected_intent,
        analysis_plan=analysis_plan,
        analysis_contract=analysis_contract,
        preflight=preflight,
        method=method,
        error=error,
        cache_state=cache_state,
        memory_state=memory_state,
        data_quality=output.get("data_quality"),
        model_metrics=output.get("model_metrics"),
        model_quality=output.get("model_quality"),
        risk=output.get("risk"),
        explanation=output.get("explanation"),
        limitations=output.get("limitations"),
        logged_failure_count=len(output.get("failure_events") or []),
        inconsistency_detected=bool(output.get("inconsistency_detected")),
    )
    output["reproducibility"] = build_reproducibility_metadata(
        source_fingerprint=output.get("dataset_fingerprint") or (analysis_contract or {}).get("dataset_fingerprint"),
        pipeline_trace=output.get("pipeline_trace"),
        result_hash=output.get("result_hash") or (analysis_contract or {}).get("result_hash"),
        consistency_payload={
            "result_hash": output.get("result_hash") or (analysis_contract or {}).get("result_hash"),
            "inconsistency_detected": output.get("inconsistency_detected"),
            "prior_hash_count": ((analysis_contract or {}).get("reproducibility") or {}).get("prior_hash_count"),
            "consistency_validated": ((analysis_contract or {}).get("reproducibility") or {}).get("consistency_validated"),
        },
    )
    if analysis_contract:
        analysis_contract["reproducibility"] = dict(output["reproducibility"])
        analysis_contract["dataset_fingerprint"] = str(
            analysis_contract.get("dataset_fingerprint") or output["reproducibility"].get("dataset_fingerprint") or ""
        )
        output["analysis_contract"] = ensure_analysis_contract_defaults(analysis_contract)
    return output


def _run_execution_stage(user_query, code, df):
    result, error = execute_code(code, df)
    log_step(user_query, code, error)
    return result, error


def _build_simple_pipeline_output(
    output,
    *,
    intent,
    analysis_plan=None,
    system_decision=None,
    module_validation,
    build_query,
    build_plan,
    generated_code,
    code,
    status,
    test_error,
    error,
    workflow_context=None,
    result=None,
    fix_applied=False,
    fix_status=None,
    fixed_code=None,
    tool_used="PYTHON",
    analysis_mode="ad-hoc",
    execution_plan=None,
    excel_analysis=None,
    dashboard=None,
    execution_trace=None,
    optimization=None,
):
    output.update(
        {
            "intent": intent,
            "analysis_contract": None,
            "analysis_plan": analysis_plan or {},
            "system_decision": dict(system_decision or {}),
            "module_validation": module_validation,
            "build_query": build_query,
            "build_plan": build_plan,
            "generated_code": generated_code,
            "test_status": status,
            "test_error": test_error,
            "fix_applied": fix_applied,
            "fix_status": fix_status or "No automatic fix was attempted.",
            "fixed_code": fixed_code,
            "code": code,
            "result": result,
            "error": error,
            "validation": module_validation,
            "explanation": None,
            "model_metrics": {"mae": None, "r2": None},
            "summary": None,
            "packaged_output": None,
            "power_bi_brief": None,
            "insights": None,
            "tool_used": str(tool_used or "PYTHON"),
            "analysis_mode": str(analysis_mode or "ad-hoc"),
            "execution_plan": _normalize_execution_plan_output(execution_plan),
            "execution_trace": _normalize_execution_trace_output(execution_plan, execution_trace),
            "optimization": _normalize_optimization_output(execution_trace, optimization),
            "excel_analysis": dict(excel_analysis or {}) or None,
            "dashboard": dict(dashboard or {}) or None,
            "forecast_metadata": None,
            "suggested_questions": [],
            "active_filter": None,
            "visualization_type": None,
            "recommendations": [],
            "warnings": [],
            "confidence": "",
            "data_quality": {
                "score": 0.0,
                "data_quality_score": 0.0,
                "issues": [],
                "warnings": [],
                "profile": {},
                "data_profile": {},
                "anomalies": {},
            },
            "result_hash": "",
            "dataset_fingerprint": "",
            "reproducibility": {
                "dataset_fingerprint": "",
                "pipeline_trace_hash": "",
                "result_hash": "",
                "consistent_with_prior_runs": True,
                "prior_hash_count": 0,
                "consistency_validated": False,
            },
            "inconsistency_detected": False,
            "limitations": [],
            "model_quality": "weak",
            "risk": "",
            "data_score": {},
            "pipeline_trace": [],
            "cache_status": {"status": "pending"},
            "memory_update": {"status": "pending"},
            "failure_events": [],
            "failure_patterns": get_failure_patterns(),
            "decision_layer": {
                "decisions": [],
                "top_decision": None,
                "decision_confidence": "low",
                "risk_summary": "",
                "learning_insights": {
                    "patterns": [],
                    "confidence_adjustment": "No historical outcomes available; using base confidence.",
                    "risk_adjustment": "No historical outcomes available; using base risk.",
                },
            },
            "business_decisions": None,
            "storytelling": None,
            "anomalies": None,
            "workflow_context": workflow_context or {},
        }
    )
    return output


def _execute_orchestrated_python_step(
    *,
    step,
    query,
    df,
    current_result,
    analysis_plan,
    preflight,
    max_retries,
    model,
):
    del current_result, preflight
    step_query = str(query or step.get("query") or step.get("task") or "").strip()
    normalized_step_query = step_query.lower()
    has_current_datetime = bool(
        analysis_plan.get("datetime_column")
        and analysis_plan.get("datetime_column") in getattr(df, "columns", [])
    )
    uses_forecast_path = (
        str(analysis_plan.get("intent") or "").strip().lower() == "prediction"
        and has_current_datetime
        and (
            "forecast" in normalized_step_query
            or "next " in normalized_step_query
            or detect_intent(step_query) == "forecast"
        )
    )

    if uses_forecast_path:
        module_validation = validate_forecast_dataset(df, model=model)
        python_steps = [
            "Validate forecast inputs.",
            "Prepare time features and forecast-ready data.",
            "Run the existing Python prediction pipeline.",
        ]
        if not _query_is_valid(module_validation):
            validation_message = module_validation or "INVALID\nForecast validation failed."
            return {
                "result": None,
                "generated_code": None,
                "final_code": None,
                "error": f"Dataset validation failed. {validation_message}",
                "fix_applied": False,
                "fix_status": "No automatic fix was attempted.",
                "fixed_code": None,
                "analysis_method": "multi_tool_forecast",
                "module_validation": module_validation,
                "python_steps": python_steps,
                "warnings": [],
            }

        prep_code = generate_forecast_prep_code(df, model=model)
        prep_execution = _execute_with_fix_details(
            step_query,
            df,
            prep_code,
            max_retries=max_retries,
            model=model,
        )
        if prep_execution["error"]:
            return {
                "result": None,
                "generated_code": prep_code,
                "final_code": prep_execution["code"],
                "error": f"Forecast preparation failed. {prep_execution['error']}",
                "fix_applied": prep_execution["fix_applied"],
                "fix_status": prep_execution["fix_status"],
                "fixed_code": prep_execution["fixed_code"],
                "analysis_method": "multi_tool_forecast",
                "module_validation": module_validation,
                "python_steps": python_steps,
                "warnings": [],
            }

        prepared_df = _coerce_forecast_dataframe(prep_execution["result"])
        if prepared_df is None or prepared_df.empty:
            return {
                "result": None,
                "generated_code": prep_code,
                "final_code": prep_execution["code"],
                "error": "Forecast preparation did not produce a usable tabular dataset.",
                "fix_applied": prep_execution["fix_applied"],
                "fix_status": prep_execution["fix_status"],
                "fixed_code": prep_execution["fixed_code"],
                "analysis_method": "multi_tool_forecast",
                "module_validation": module_validation,
                "python_steps": python_steps,
                "warnings": [],
            }

        prediction_code = generate_sales_prediction_code(prepared_df, model=model)
        prediction_execution = _execute_with_fix_details(
            step_query,
            prepared_df,
            prediction_code,
            max_retries=max_retries,
            model=model,
        )
        combined_code = _combine_stage_code(
            ("Forecast Preparation", prep_execution["code"]),
            ("Sales Prediction", prediction_execution["code"]),
        )
        fix_applied = prep_execution["fix_applied"] or prediction_execution["fix_applied"]
        fixed_code = combined_code if fix_applied else None
        fix_status = prediction_execution["fix_status"]
        if prep_execution["fix_applied"] and prediction_execution["fix_applied"]:
            fix_status = "Automatic fixes were applied to forecast preparation and prediction."
        elif prep_execution["fix_applied"]:
            fix_status = "Automatic fix applied to forecast preparation."
        return {
            "result": prediction_execution["result"],
            "generated_code": _combine_stage_code(
                ("Forecast Preparation", prep_code),
                ("Sales Prediction", prediction_code),
            ),
            "final_code": combined_code,
            "error": prediction_execution["error"],
            "fix_applied": fix_applied,
            "fix_status": fix_status,
            "fixed_code": fixed_code,
            "analysis_method": "multi_tool_forecast",
            "module_validation": module_validation,
            "python_steps": python_steps,
            "warnings": [],
        }

    deterministic_shortcut = build_deterministic_analysis_code(step_query, df, analysis_plan)
    if deterministic_shortcut is not None:
        generated_code = deterministic_shortcut["code"]
        python_steps = [
            "Apply a deterministic Python shortcut to the current context.",
            "Execute the generated Python analysis.",
            "Validate the Python step result.",
        ]
        analysis_method = "multi_tool_deterministic_python"
    else:
        generated_code = generate_code(step_query, df, model=model)
        python_steps = [
            "Generate Python analysis code for the current context.",
            "Execute the generated Python analysis.",
            "Validate the Python step result.",
        ]
        analysis_method = "multi_tool_python"

    execution_details = _execute_with_fix_details(
        step_query,
        df,
        generated_code,
        max_retries=max_retries,
        model=model,
    )
    return {
        "result": execution_details["result"],
        "generated_code": generated_code,
        "final_code": execution_details["code"],
        "error": execution_details["error"],
        "fix_applied": execution_details["fix_applied"],
        "fix_status": execution_details["fix_status"],
        "fixed_code": execution_details["fixed_code"],
        "analysis_method": analysis_method,
        "module_validation": "VALID\nPython orchestration step selected.",
        "python_steps": python_steps,
        "warnings": [],
    }


def build_master_prompt(user_query, df):
    template = _read_template("master_prompt.txt")
    return _fill_template(
        template,
        {
            **_format_dataframe_context(df),
            "df_info": build_df_info(df),
            "user_query": str(user_query),
        },
    )


def generate_plan(user_query, df, model=DEFAULT_GEMINI_MODEL):
    return _run_gemini_template_prompt(
        "plan_prompt.txt",
        {
            **_format_dataframe_context(df),
            "user_query": str(user_query),
        },
        model=model,
    )


def understand_columns(user_query, df, model=DEFAULT_GEMINI_MODEL):
    return _run_gemini_template_prompt(
        "column_prompt.txt",
        {
            **_format_dataframe_context(df),
            "user_query": str(user_query),
        },
        model=model,
    )


def explain_columns(df, model=DEFAULT_GEMINI_MODEL):
    return understand_columns("", df, model=model)


def generate_code(user_query, df, model=DEFAULT_GEMINI_MODEL):
    prompt = build_master_prompt(user_query, df)
    return _generate_groq_content(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    ).strip()


def fix_code(code, error, df, user_query, model=DEFAULT_GEMINI_MODEL):
    return _run_groq_template_prompt(
        "error_prompt.txt",
        {
            **_format_dataframe_context(df),
            "generated_code": str(code),
            "error_message": str(error),
            "user_query": str(user_query),
        },
        model=model,
    )


def regenerate_code(user_query, generated_code, validation_feedback, df, model=DEFAULT_GEMINI_MODEL):
    return _run_groq_template_prompt(
        "regenerate_prompt.txt",
        {
            **_format_dataframe_context(df),
            "user_query": str(user_query),
            "generated_code": str(generated_code),
            "validation_feedback": str(validation_feedback),
        },
        model=model,
    )


def generate_insights(user_query, result, df, model=DEFAULT_GEMINI_MODEL):
    return generate_decision_grade_insights(
        user_query=str(user_query),
        result=result,
        df=df,
        model=model,
        prompt_runner=_run_gemini_template_prompt,
        format_result=_format_result,
        format_dataframe_context=_format_dataframe_context,
    )


def quality_check(user_query, result, model=DEFAULT_GEMINI_MODEL):
    return _run_gemini_template_prompt(
        "quality_check_prompt.txt",
        {
            "user_query": str(user_query),
            "result": _format_result(result),
        },
        model=model,
    )


def _build_allowed_execution_builtins():
    return {
        "__import__": _safe_import,
        "abs": abs,
        "all": all,
        "any": any,
        "AssertionError": AssertionError,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "Exception": Exception,
        "float": float,
        "getattr": getattr,
        "hasattr": hasattr,
        "int": int,
        "isinstance": isinstance,
        "iter": iter,
        "KeyError": KeyError,
        "len": len,
        "list": list,
        "LookupError": LookupError,
        "max": max,
        "min": min,
        "NameError": NameError,
        "next": next,
        "print": print,
        "range": range,
        "round": round,
        "RuntimeError": RuntimeError,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "TypeError": TypeError,
        "ValueError": ValueError,
        "zip": zip,
    }


def safe_execute(code, df):
    namespace = {"__builtins__": _build_allowed_execution_builtins(), "df": df}

    try:
        exec(code, namespace, namespace)
        return namespace.get("result"), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def execute_code(code, df):
    return safe_execute(code, df)


def _execute_with_fix_loop(user_query, df, code, max_retries=3, model=DEFAULT_GEMINI_MODEL):
    error = None
    attempts = max(1, int(max_retries))

    for _ in range(attempts):
        result, error = execute_code(code, df)
        log_step(user_query, code, error)

        if not error:
            return result, code, None

        code = fix_code(code, error, df, user_query, model=model)

    return None, code, error


def _execute_with_fix_details(user_query, df, code, max_retries=3, model=DEFAULT_GEMINI_MODEL):
    result, final_code, error = _execute_with_fix_loop(
        user_query,
        df,
        code,
        max_retries=max_retries,
        model=model,
    )
    fix_applied = final_code != code

    if error:
        if fix_applied:
            fix_status = "Automatic fix retries were attempted, but execution still failed."
        else:
            fix_status = "Execution failed before an automatic fix could recover the analysis."
    else:
        if fix_applied:
            fix_status = "Automatic fix applied successfully after an execution error."
        else:
            fix_status = "Execution passed without needing an automatic fix."

    return {
        "result": result,
        "code": final_code,
        "error": error,
        "fix_applied": fix_applied,
        "fix_status": fix_status,
        "fixed_code": final_code if fix_applied else None,
    }


def run_pipeline(user_query, df, max_retries=3, model=DEFAULT_GEMINI_MODEL):
    code = generate_code(user_query, df, model=model)
    return _execute_with_fix_loop(
        user_query,
        df,
        code,
        max_retries=max_retries,
        model=model,
    )


def run_validated_pipeline(user_query, df, max_retries=3, model=DEFAULT_GEMINI_MODEL):
    result, code, error = run_pipeline(user_query, df, max_retries=max_retries, model=model)
    validation = None

    if error:
        return result, code, error, validation

    validation = quality_check(user_query, result, model=model)

    if _validation_is_incorrect(validation):
        improved_code = regenerate_code(user_query, code, validation, df, model=model)
        result, error = execute_code(improved_code, df)
        code = improved_code

        if error:
            fixed_code = fix_code(code, error, df, user_query, model=model)
            result, error = execute_code(fixed_code, df)
            code = fixed_code

        if not error:
            validation = quality_check(user_query, result, model=model)

    return result, code, error, validation


def run_builder_pipeline(
    user_query,
    df,
    max_retries=3,
    model=DEFAULT_GEMINI_MODEL,
    workflow_context=None,
    _routing_override=None,
):
    _ensure_builder_configuration()

    output = {
        "query": str(user_query),
        "intent": None,
        "analysis_contract": None,
        "system_decision": {},
        "analysis_plan": {},
        "module_validation": None,
        "build_query": None,
        "build_plan": None,
        "generated_code": None,
        "test_status": None,
        "test_error": None,
        "fix_applied": False,
        "fix_status": None,
        "fixed_code": None,
        "code": None,
        "result": None,
        "error": None,
        "validation": None,
        "explanation": None,
        "model_metrics": {"mae": None, "r2": None},
        "summary": None,
        "packaged_output": None,
        "power_bi_brief": None,
        "insights": None,
        "tool_used": "PYTHON",
        "analysis_mode": "ad-hoc",
        "execution_plan": _empty_execution_plan(),
        "execution_trace": [],
        "optimization": {
            "execution_time_total": 0,
            "cost_estimate": "low",
            "optimized": False,
            "parallel_execution": False,
            "plans_considered": 1,
            "selected_plan_score": 0.0,
            "constraints_applied": {},
        },
        "excel_analysis": None,
        "dashboard": None,
        "forecast_metadata": None,
        "suggested_questions": [],
        "active_filter": None,
        "visualization_type": None,
        "recommendations": [],
        "warnings": [],
        "confidence": "",
        "data_quality": {
            "score": 0.0,
            "data_quality_score": 0.0,
            "issues": [],
            "warnings": [],
            "profile": {},
            "data_profile": {},
            "anomalies": {},
        },
        "result_hash": "",
        "dataset_fingerprint": "",
        "reproducibility": {
            "dataset_fingerprint": "",
            "pipeline_trace_hash": "",
            "result_hash": "",
            "consistent_with_prior_runs": True,
            "prior_hash_count": 0,
            "consistency_validated": False,
        },
        "inconsistency_detected": False,
        "limitations": [],
        "model_quality": "weak",
        "risk": "",
        "data_score": {},
        "pipeline_trace": [],
        "cache_status": {"status": "pending"},
        "memory_update": {"status": "pending"},
        "failure_events": [],
        "failure_patterns": get_failure_patterns(),
        "decision_layer": {
            "decisions": [],
            "top_decision": None,
            "decision_confidence": "low",
            "risk_summary": "",
            "learning_insights": {
                "patterns": [],
                "confidence_adjustment": "No historical outcomes available; using base confidence.",
                "risk_adjustment": "No historical outcomes available; using base risk.",
            },
        },
        "business_decisions": None,
        "storytelling": None,
        "anomalies": None,
        "workflow_context": workflow_context or {},
    }
    failure_events = output["failure_events"]

    detected_intent = detect_intent(user_query)
    improved_query = str(user_query).strip()
    dataset_intelligence = detect_dataset_type(df)
    system_decision = _build_system_decision(df, improved_query, routing_override=_routing_override)
    output["system_decision"] = dict(system_decision)
    if detected_intent == "rating":
        analysis_intent = classify_analysis_intent(improved_query)
    elif system_decision["selected_mode"] in {"forecast", "ml"}:
        analysis_intent = "prediction"
    else:
        analysis_intent = classify_analysis_intent(improved_query)
        if analysis_intent == "prediction":
            analysis_intent = "analysis"

    analysis_plan = build_analysis_plan(improved_query, df, intent=analysis_intent)
    analysis_plan = _apply_system_mode_to_plan(
        analysis_plan,
        system_decision,
        dataset_intelligence,
    )
    preflight = validate_analysis_request(improved_query, df, analysis_plan)
    if _routing_override and str(_routing_override.get("reason") or "").strip():
        preflight = dict(preflight)
        preflight["warnings"] = list(
            dict.fromkeys(
                list(preflight.get("warnings") or [])
                + [str(_routing_override.get("reason")).strip()]
            )
        )
    analysis_method = str(analysis_plan.get("method") or "llm_assisted")
    uses_forecast_mode = (
        detected_intent != "rating"
        and system_decision.get("selected_mode") == "forecast"
    )
    module_validation = None
    generated_code = None
    final_code = None
    final_result = None
    error = None
    prep_code = None
    fix_applied = False
    fix_status = "No automatic fix was attempted."
    fixed_code = None
    summary_override = None
    insights_override = None
    required_columns_label = ", ".join(analysis_plan.get("required_columns") or []) or "auto-detect"
    transformation_label = "; ".join(analysis_plan.get("transformations") or []) or "No extra transforms required."
    tool_warnings: list[str] = []
    execution_plan = _empty_execution_plan()
    execution_trace = []
    optimization_payload = dict(output.get("optimization") or {})
    planned_execution_steps = []
    plan_already_executed = False
    excel_analysis = None
    dashboard_payload = None
    tool_selection = {
        "tool_used": "PYTHON",
        "analysis_mode": "ad-hoc",
        "reason": "Default Python execution path.",
        "fallback_reason": None,
    }
    if detected_intent == "rating":
        tool_selection = {
            "tool_used": "PYTHON",
            "analysis_mode": "ad-hoc",
            "reason": "Ratings analysis stays on the legacy Python path.",
            "fallback_reason": None,
        }
    else:
        tool_selection = select_tool(
            improved_query,
            df,
            plan=analysis_plan,
            preflight=preflight,
        )

    analysis_plan["tool_used"] = tool_selection["tool_used"]
    analysis_plan["analysis_mode"] = tool_selection["analysis_mode"]
    analysis_plan["tool_reason"] = tool_selection["reason"]
    if detected_intent != "rating":
        planned_execution_steps = build_execution_plan(
            improved_query,
            df,
            plan=analysis_plan,
            preflight=preflight,
        )
        if planned_execution_steps:
            analysis_plan["execution_plan"] = planned_execution_steps
    if tool_selection.get("fallback_reason"):
        execution_plan["fallback_reason"] = str(tool_selection["fallback_reason"])
        tool_warnings.append(str(tool_selection["fallback_reason"]))
        if planned_execution_steps:
            planned_execution_steps[-1]["fallback_reason"] = str(tool_selection["fallback_reason"])

    if len(planned_execution_steps) > 1:
        analysis_plan["tool_used"] = determine_primary_tool(
            planned_execution_steps,
            default=analysis_plan.get("tool_used"),
        )
        analysis_plan["analysis_mode"] = determine_analysis_mode(
            planned_execution_steps,
            intent=analysis_intent,
            default=analysis_plan.get("analysis_mode"),
        )
        analysis_plan["tool_reason"] = (
            "Multi-tool orchestration selected an ordered execution chain for the request."
        )
        analysis_plan["steps"] = [
            f"{step.get('tool')}: {step.get('task')}"
            for step in planned_execution_steps
        ]

    critical_blocking_errors = bool(preflight.get("blocking_errors")) and (df.empty or len(df.columns) == 0)
    if critical_blocking_errors:
        blocking_message = " ".join(preflight["blocking_errors"])
        plan = (
            f"Intent Detection: {detected_intent}\n"
            f"Analysis Contract Intent: {analysis_intent}\n"
            f"Tool Selection: {analysis_plan['tool_used']} ({analysis_plan['analysis_mode']})\n"
            f"Required Columns: {required_columns_label}\n"
            f"Transforms: {transformation_label}\n"
            f"Execution Path: blocked by validation -> {blocking_message}"
        )
        pipeline_output = _build_simple_pipeline_output(
            output,
            intent=detected_intent,
            analysis_plan=analysis_plan,
            system_decision=system_decision,
            module_validation=blocking_message,
            build_query=improved_query,
            build_plan=plan,
            generated_code=None,
            code=None,
            status="SKIPPED",
            test_error=blocking_message,
            error=blocking_message,
            workflow_context=workflow_context,
            tool_used=analysis_plan["tool_used"],
            analysis_mode=analysis_plan["analysis_mode"],
            execution_plan=planned_execution_steps or execution_plan,
            execution_trace=execution_trace,
            optimization=optimization_payload,
        )
        _record_failure(failure_events, improved_query, blocking_message, "validation_blocked")
        reliability_payload = _build_analysis_reliability_payload(
            query=improved_query,
            df=df,
            result=None,
            plan=analysis_plan,
            workflow_context=workflow_context,
            preflight=preflight,
            data_score=_build_data_score(preflight, error=blocking_message),
            failure_events=failure_events,
            use_llm_limitations=False,
        )
        contract = _build_analysis_contract_payload(
            query=improved_query,
            df=df,
            result=None,
            code=None,
            plan=analysis_plan,
            preflight=preflight,
            method=analysis_method,
            error=blocking_message,
            workflow_context=workflow_context,
            data_score=_build_data_score(preflight, error=blocking_message),
            reliability_payload=reliability_payload,
            tool_used=analysis_plan["tool_used"],
            analysis_mode=analysis_plan["analysis_mode"],
            execution_plan=planned_execution_steps or execution_plan,
            execution_trace=execution_trace,
            optimization=optimization_payload,
            system_decision=system_decision,
        )
        pipeline_output = _apply_analysis_contract(pipeline_output, contract)
        pipeline_output["failure_events"] = list(failure_events)
        return _apply_pipeline_metadata(
            pipeline_output,
            query=improved_query,
            detected_intent=detected_intent,
            analysis_plan=analysis_plan,
            preflight=preflight,
            method=analysis_method,
            error=blocking_message,
        )

    if detected_intent == "rating":
        module_validation = validate_ratings_dataset(df, model=model)
        if _query_is_valid(module_validation):
            generated_code = generate_ratings_analysis_code(df, model=model)
        analysis_method = "specialized_rating_module"
        execution_plan["python_steps"] = [
            "Validate ratings dataset compatibility.",
            "Generate Python ratings analysis code.",
            "Execute and verify the ratings result.",
        ]
        plan = (
            f"Intent Detection: {detected_intent}\n"
            f"Analysis Contract Intent: {analysis_intent}\n"
            f"Tool Selection: {analysis_plan['tool_used']} ({analysis_plan['analysis_mode']})\n"
            f"Tool Reason: {analysis_plan['tool_reason']}\n"
            f"Validation: {module_validation}\n"
            f"Required Columns: {required_columns_label}\n"
            f"Transforms: {transformation_label}\n"
            "Execution Path: ratings analysis -> execution -> result"
        )
    elif len(planned_execution_steps) > 1:
        orchestration_payload = execute_plan(
            planned_execution_steps,
            df,
            query=improved_query,
            analysis_plan=analysis_plan,
            preflight=preflight,
            tables={"df": df},
            python_runner=lambda **kwargs: _execute_orchestrated_python_step(
                max_retries=max_retries,
                model=model,
                **kwargs,
            ),
            constraints=dict(analysis_plan.get("optimization_constraints") or {}),
        )
        plan_already_executed = True
        execution_plan = orchestration_payload.get("execution_plan") or planned_execution_steps
        execution_trace = orchestration_payload.get("execution_trace") or []
        optimization_payload = dict(orchestration_payload.get("optimization") or optimization_payload)
        analysis_plan["execution_plan"] = execution_plan
        tool_warnings.extend(list(orchestration_payload.get("warnings") or []))
        excel_analysis = orchestration_payload.get("excel_analysis")
        dashboard_payload = orchestration_payload.get("dashboard")
        generated_code = orchestration_payload.get("generated_code")
        final_code = orchestration_payload.get("final_code")
        final_result = orchestration_payload.get("result")
        error = orchestration_payload.get("error")
        analysis_method = str(orchestration_payload.get("analysis_method") or "multi_tool_orchestration")
        module_validation = str(
            orchestration_payload.get("module_validation")
            or "VALID\nMulti-tool orchestration selected."
        )
        fix_applied = bool(orchestration_payload.get("fix_applied"))
        fix_status = str(orchestration_payload.get("fix_status") or fix_status)
        fixed_code = orchestration_payload.get("fixed_code")
        analysis_plan["steps"] = [
            f"{step.get('tool')}: {step.get('task')}"
            for step in execution_plan
        ]
        for trace_event in execution_trace:
            if trace_event.get("error"):
                _record_failure(
                    failure_events,
                    improved_query,
                    trace_event.get("error"),
                    f"execution_step_{trace_event.get('step')}",
                )
        plan = (
            f"Intent Detection: {detected_intent}\n"
            f"Analysis Contract Intent: {analysis_intent}\n"
            f"Tool Selection: {analysis_plan['tool_used']} ({analysis_plan['analysis_mode']})\n"
            f"Tool Reason: {analysis_plan['tool_reason']}\n"
            f"Validation: {module_validation}\n"
            f"Required Columns: {required_columns_label}\n"
            f"Transforms: {transformation_label}\n"
            f"Execution Path: {_execution_route_label(execution_plan)}"
        )
    elif uses_forecast_mode and analysis_plan["tool_used"] == "PYTHON":
        module_validation = validate_forecast_dataset(df, model=model)
        if _query_is_valid(module_validation):
            prep_code = generate_forecast_prep_code(df, model=model)
            generated_code = _combine_stage_code(("Forecast Preparation", prep_code))
        analysis_method = "specialized_forecast_module"
        execution_plan["python_steps"] = [
            "Validate forecast inputs.",
            "Prepare time features and forecast-ready data.",
            "Run the existing Python prediction pipeline.",
        ]
        plan = (
            f"Intent Detection: {detected_intent}\n"
            f"Analysis Contract Intent: {analysis_intent}\n"
            f"Tool Selection: {analysis_plan['tool_used']} ({analysis_plan['analysis_mode']})\n"
            f"Tool Reason: {analysis_plan['tool_reason']}\n"
            f"Validation: {module_validation}\n"
            f"Required Columns: {required_columns_label}\n"
            f"Transforms: {transformation_label}\n"
            "Execution Path: forecast preparation -> sales prediction -> result"
        )
    elif analysis_plan["tool_used"] == "SQL":
        module_validation = "VALID\nDeterministic SQL simulation selected."
        sql_payload = run_sql_analysis(
            improved_query,
            df,
            tables={"df": df},
            plan=analysis_plan,
            preflight=preflight,
        )
        tool_warnings.extend(list(sql_payload.get("warnings") or []))
        if sql_payload.get("unsupported") or sql_payload.get("result") is None:
            excel_payload = run_excel_analysis(
                improved_query,
                df,
                plan=analysis_plan,
                preflight=preflight,
            )
            analysis_plan["tool_used"] = "EXCEL"
            analysis_plan["tool_reason"] = "SQL simulation could not execute safely, so the engine returned an Excel-style analyst summary."
            tool_warnings.extend(list(excel_payload.get("warnings") or []))
            execution_plan["fallback_reason"] = execution_plan.get("fallback_reason") or (
                tool_warnings[0] if tool_warnings else "SQL simulation fell back to Excel summary."
            )
            excel_analysis = excel_payload.get("excel_analysis")
            execution_plan["excel_logic"] = {
                "pivot_table": dict((excel_analysis or {}).get("pivot_table") or {}),
                "aggregations": dict((excel_analysis or {}).get("aggregations") or {}),
            }
            analysis_method = "deterministic_excel"
            generated_code = _render_excel_logic_code(excel_analysis)
            final_code = generated_code
            final_result = excel_payload.get("result")
            plan = (
                f"Intent Detection: {detected_intent}\n"
                f"Analysis Contract Intent: {analysis_intent}\n"
                f"Tool Selection: {analysis_plan['tool_used']} ({analysis_plan['analysis_mode']})\n"
                f"Tool Reason: {analysis_plan['tool_reason']}\n"
                f"Validation: {module_validation}\n"
                f"Required Columns: {required_columns_label}\n"
                f"Transforms: {transformation_label}\n"
                "Execution Path: SQL simulation fallback -> Excel summary -> result"
            )
        else:
            execution_plan["sql_plan"] = sql_payload.get("sql_plan")
            analysis_method = "deterministic_sql"
            generated_code = _render_sql_plan_code(sql_payload.get("sql_plan"))
            final_code = generated_code
            final_result = sql_payload.get("result")
            plan = (
                f"Intent Detection: {detected_intent}\n"
                f"Analysis Contract Intent: {analysis_intent}\n"
                f"Tool Selection: {analysis_plan['tool_used']} ({analysis_plan['analysis_mode']})\n"
                f"Tool Reason: {analysis_plan['tool_reason']}\n"
                f"Validation: {module_validation}\n"
                f"Required Columns: {required_columns_label}\n"
                f"Transforms: {transformation_label}\n"
                "Execution Path: SQL simulation -> result"
            )
    elif analysis_plan["tool_used"] == "EXCEL":
        module_validation = "VALID\nDeterministic Excel analysis selected."
        excel_payload = run_excel_analysis(
            improved_query,
            df,
            plan=analysis_plan,
            preflight=preflight,
        )
        tool_warnings.extend(list(excel_payload.get("warnings") or []))
        excel_analysis = excel_payload.get("excel_analysis")
        execution_plan["excel_logic"] = {
            "pivot_table": dict((excel_analysis or {}).get("pivot_table") or {}),
            "aggregations": dict((excel_analysis or {}).get("aggregations") or {}),
        }
        analysis_method = "deterministic_excel"
        generated_code = _render_excel_logic_code(excel_analysis)
        final_code = generated_code
        final_result = excel_payload.get("result")
        plan = (
            f"Intent Detection: {detected_intent}\n"
            f"Analysis Contract Intent: {analysis_intent}\n"
            f"Tool Selection: {analysis_plan['tool_used']} ({analysis_plan['analysis_mode']})\n"
            f"Tool Reason: {analysis_plan['tool_reason']}\n"
            f"Validation: {module_validation}\n"
            f"Required Columns: {required_columns_label}\n"
            f"Transforms: {transformation_label}\n"
            "Execution Path: Excel summary / pivot simulation -> result"
        )
    elif analysis_plan["tool_used"] == "BI":
        module_validation = "VALID\nDeterministic BI dashboard generation selected."
        dashboard_result = build_dashboard_output(
            improved_query,
            df,
            result=None,
            plan=analysis_plan,
            preflight=preflight,
        )
        tool_warnings.extend(list(dashboard_result.get("warnings") or []))
        dashboard_payload = dashboard_result.get("dashboard")
        if not list((dashboard_payload or {}).get("charts") or []):
            excel_payload = run_excel_analysis(
                improved_query,
                df,
                plan=analysis_plan,
                preflight=preflight,
            )
            analysis_plan["tool_used"] = "EXCEL"
            analysis_plan["tool_reason"] = "BI dashboard generation could not find reliable chartable fields, so the engine returned an Excel-style analyst summary."
            tool_warnings.extend(list(excel_payload.get("warnings") or []))
            execution_plan["fallback_reason"] = execution_plan.get("fallback_reason") or (
                tool_warnings[0] if tool_warnings else "BI dashboard generation fell back to Excel summary."
            )
            excel_analysis = excel_payload.get("excel_analysis")
            execution_plan["excel_logic"] = {
                "pivot_table": dict((excel_analysis or {}).get("pivot_table") or {}),
                "aggregations": dict((excel_analysis or {}).get("aggregations") or {}),
            }
            analysis_method = "deterministic_excel"
            generated_code = _render_excel_logic_code(excel_analysis)
            final_code = generated_code
            final_result = excel_payload.get("result")
            plan = (
                f"Intent Detection: {detected_intent}\n"
                f"Analysis Contract Intent: {analysis_intent}\n"
                f"Tool Selection: {analysis_plan['tool_used']} ({analysis_plan['analysis_mode']})\n"
                f"Tool Reason: {analysis_plan['tool_reason']}\n"
                f"Validation: {module_validation}\n"
                f"Required Columns: {required_columns_label}\n"
                f"Transforms: {transformation_label}\n"
                "Execution Path: BI dashboard fallback -> Excel summary -> result"
            )
        else:
            analysis_method = "deterministic_bi"
            generated_code = _render_dashboard_code(dashboard_payload)
            final_code = generated_code
            final_result = {"dashboard": dashboard_payload}
            plan = (
                f"Intent Detection: {detected_intent}\n"
                f"Analysis Contract Intent: {analysis_intent}\n"
                f"Tool Selection: {analysis_plan['tool_used']} ({analysis_plan['analysis_mode']})\n"
                f"Tool Reason: {analysis_plan['tool_reason']}\n"
                f"Validation: {module_validation}\n"
                f"Required Columns: {required_columns_label}\n"
                f"Transforms: {transformation_label}\n"
                "Execution Path: BI dashboard structure -> result"
            )
    else:
        module_validation = "VALID\nGeneral analysis does not require specialized module validation."
        deterministic_shortcut = build_deterministic_analysis_code(improved_query, df, analysis_plan)
        if deterministic_shortcut is not None:
            generated_code = deterministic_shortcut["code"]
            execution_plan["python_steps"] = [
                "Apply a deterministic Python shortcut.",
                "Execute the generated Python analysis.",
                "Verify the result object.",
            ]
            analysis_method = "deterministic_python"
            plan = (
                f"Intent Detection: {detected_intent}\n"
                f"Analysis Contract Intent: {analysis_intent}\n"
                f"Tool Selection: {analysis_plan['tool_used']} ({analysis_plan['analysis_mode']})\n"
                f"Tool Reason: {analysis_plan['tool_reason']}\n"
                f"Validation: {module_validation}\n"
                f"Required Columns: {required_columns_label}\n"
                f"Transforms: {transformation_label}\n"
                f"Execution Path: {deterministic_shortcut['label']} -> execution -> result"
            )
        else:
            generated_code = generate_code(improved_query, df, model=model)
            execution_plan["python_steps"] = [
                "Generate Python analysis code.",
                "Execute the generated Python analysis.",
                "Validate the result output.",
            ]
            plan = (
                f"Intent Detection: {detected_intent}\n"
                f"Analysis Contract Intent: {analysis_intent}\n"
                f"Tool Selection: {analysis_plan['tool_used']} ({analysis_plan['analysis_mode']})\n"
                f"Tool Reason: {analysis_plan['tool_reason']}\n"
                f"Validation: {module_validation}\n"
                f"Required Columns: {required_columns_label}\n"
                f"Transforms: {transformation_label}\n"
                "Execution Path: general analysis -> execution -> result"
            )

    if tool_warnings:
        preflight = dict(preflight)
        preflight["warnings"] = list(dict.fromkeys(list(preflight.get("warnings") or []) + tool_warnings))

    if not plan_already_executed and not generated_code and analysis_plan["tool_used"] == "PYTHON":
        validation_message = module_validation or "INVALID\nDataset validation failed."
        if uses_forecast_mode and _routing_override is None:
            return run_builder_pipeline(
                user_query,
                df,
                max_retries=max_retries,
                model=model,
                workflow_context=workflow_context,
                _routing_override=_build_forecast_fallback_override(validation_message),
            )
        pipeline_output = _build_simple_pipeline_output(
            output,
            intent=detected_intent,
            analysis_plan=analysis_plan,
            system_decision=system_decision,
            module_validation=module_validation,
            build_query=improved_query,
            build_plan=plan,
            generated_code=None,
            code=None,
            status="SKIPPED",
            test_error=_validation_reason(validation_message),
            error=f"Dataset validation failed.\n{validation_message}",
            workflow_context=workflow_context,
            tool_used=analysis_plan["tool_used"],
            analysis_mode=analysis_plan["analysis_mode"],
            execution_plan=execution_plan,
            execution_trace=execution_trace,
            optimization=optimization_payload,
            excel_analysis=excel_analysis,
            dashboard=dashboard_payload,
        )
        _record_failure(
            failure_events,
            improved_query,
            f"Dataset validation failed. {validation_message}",
            "module_validation",
        )
        reliability_payload = _build_analysis_reliability_payload(
            query=improved_query,
            df=df,
            result=None,
            plan=analysis_plan,
            workflow_context=workflow_context,
            preflight=preflight,
            data_score=_build_data_score(preflight, error=f"Dataset validation failed. {validation_message}"),
            failure_events=failure_events,
            use_llm_limitations=False,
        )
        contract = _build_analysis_contract_payload(
            query=improved_query,
            df=df,
            result=None,
            code=None,
            plan=analysis_plan,
            preflight=preflight,
            method=analysis_method,
            error=f"Dataset validation failed. {validation_message}",
            workflow_context=workflow_context,
            data_score=_build_data_score(preflight, error=f"Dataset validation failed. {validation_message}"),
            reliability_payload=reliability_payload,
            tool_used=analysis_plan["tool_used"],
            analysis_mode=analysis_plan["analysis_mode"],
            execution_plan=execution_plan,
            execution_trace=execution_trace,
            optimization=optimization_payload,
            excel_analysis=excel_analysis,
            dashboard=dashboard_payload,
            system_decision=system_decision,
        )
        pipeline_output = _apply_analysis_contract(pipeline_output, contract)
        pipeline_output["failure_events"] = list(failure_events)
        return _apply_pipeline_metadata(
            pipeline_output,
            query=improved_query,
            detected_intent=detected_intent,
            analysis_plan=analysis_plan,
            preflight=preflight,
            method=analysis_method,
            error=f"Dataset validation failed. {validation_message}",
        )

    if not plan_already_executed and uses_forecast_mode and analysis_plan["tool_used"] == "PYTHON":
        prep_execution = _execute_with_fix_details(
            improved_query,
            df,
            prep_code,
            max_retries=max_retries,
            model=model,
        )
        prepared_result = prep_execution["result"]
        prep_error = prep_execution["error"]

        if prep_error:
            if uses_forecast_mode and _routing_override is None:
                return run_builder_pipeline(
                    user_query,
                    df,
                    max_retries=max_retries,
                    model=model,
                    workflow_context=workflow_context,
                    _routing_override=_build_forecast_fallback_override(prep_error),
                )
            pipeline_output = _build_simple_pipeline_output(
                output,
                intent=detected_intent,
                analysis_plan=analysis_plan,
                system_decision=system_decision,
                module_validation=module_validation,
                build_query=improved_query,
                build_plan=plan,
                generated_code=generated_code,
                code=prep_execution["code"],
                status="FAILED",
                test_error=prep_error,
                error=f"Forecast preparation failed.\n{prep_error}",
                workflow_context=workflow_context,
                fix_applied=prep_execution["fix_applied"],
                fix_status=prep_execution["fix_status"],
                fixed_code=prep_execution["fixed_code"],
                tool_used=analysis_plan["tool_used"],
                analysis_mode=analysis_plan["analysis_mode"],
                execution_plan=execution_plan,
                execution_trace=execution_trace,
                optimization=optimization_payload,
                excel_analysis=excel_analysis,
                dashboard=dashboard_payload,
            )
            _record_failure(failure_events, improved_query, prep_error, "forecast_preparation")
            reliability_payload = _build_analysis_reliability_payload(
                query=improved_query,
                df=df,
                result=None,
                plan=analysis_plan,
                workflow_context=workflow_context,
                preflight=preflight,
                data_score=_build_data_score(preflight, error=f"Forecast preparation failed. {prep_error}"),
                failure_events=failure_events,
                use_llm_limitations=False,
            )
            contract = _build_analysis_contract_payload(
                query=improved_query,
                df=df,
                result=None,
                code=prep_execution["code"],
                plan=analysis_plan,
                preflight=preflight,
                method=analysis_method,
                error=f"Forecast preparation failed. {prep_error}",
                workflow_context=workflow_context,
                data_score=_build_data_score(preflight, error=f"Forecast preparation failed. {prep_error}"),
                reliability_payload=reliability_payload,
                tool_used=analysis_plan["tool_used"],
                analysis_mode=analysis_plan["analysis_mode"],
                execution_plan=execution_plan,
                execution_trace=execution_trace,
                optimization=optimization_payload,
                excel_analysis=excel_analysis,
                dashboard=dashboard_payload,
            )
            pipeline_output = _apply_analysis_contract(pipeline_output, contract)
            pipeline_output["failure_events"] = list(failure_events)
            return _apply_pipeline_metadata(
                pipeline_output,
                query=improved_query,
                detected_intent=detected_intent,
                analysis_plan=analysis_plan,
                preflight=preflight,
                method=analysis_method,
                error=f"Forecast preparation failed. {prep_error}",
            )

        prepared_df = _coerce_forecast_dataframe(prepared_result)
        if prepared_df is None or prepared_df.empty:
            prep_message = "Forecast preparation did not produce a usable tabular dataset."
            if uses_forecast_mode and _routing_override is None:
                return run_builder_pipeline(
                    user_query,
                    df,
                    max_retries=max_retries,
                    model=model,
                    workflow_context=workflow_context,
                    _routing_override=_build_forecast_fallback_override(prep_message),
                )
            pipeline_output = _build_simple_pipeline_output(
                output,
                intent=detected_intent,
                analysis_plan=analysis_plan,
                system_decision=system_decision,
                module_validation=module_validation,
                build_query=improved_query,
                build_plan=plan,
                generated_code=generated_code,
                code=prep_execution["code"],
                status="FAILED",
                test_error=prep_message,
                error=prep_message,
                workflow_context=workflow_context,
                fix_applied=prep_execution["fix_applied"],
                fix_status=prep_execution["fix_status"],
                fixed_code=prep_execution["fixed_code"],
                tool_used=analysis_plan["tool_used"],
                analysis_mode=analysis_plan["analysis_mode"],
                execution_plan=execution_plan,
                execution_trace=execution_trace,
                optimization=optimization_payload,
                excel_analysis=excel_analysis,
                dashboard=dashboard_payload,
            )
            _record_failure(failure_events, improved_query, prep_message, "forecast_preparation")
            reliability_payload = _build_analysis_reliability_payload(
                query=improved_query,
                df=df,
                result=None,
                plan=analysis_plan,
                workflow_context=workflow_context,
                preflight=preflight,
                data_score=_build_data_score(preflight, error=prep_message),
                failure_events=failure_events,
                use_llm_limitations=False,
            )
            contract = _build_analysis_contract_payload(
                query=improved_query,
                df=df,
                result=None,
                code=prep_execution["code"],
                plan=analysis_plan,
                preflight=preflight,
                method=analysis_method,
                error=prep_message,
                workflow_context=workflow_context,
                data_score=_build_data_score(preflight, error=prep_message),
                reliability_payload=reliability_payload,
                tool_used=analysis_plan["tool_used"],
                analysis_mode=analysis_plan["analysis_mode"],
                execution_plan=execution_plan,
                execution_trace=execution_trace,
                optimization=optimization_payload,
                excel_analysis=excel_analysis,
                dashboard=dashboard_payload,
            )
            pipeline_output = _apply_analysis_contract(pipeline_output, contract)
            pipeline_output["failure_events"] = list(failure_events)
            return _apply_pipeline_metadata(
                pipeline_output,
                query=improved_query,
                detected_intent=detected_intent,
                analysis_plan=analysis_plan,
                preflight=preflight,
                method=analysis_method,
                error=prep_message,
            )

        if len(prepared_df) < 2:
            prep_message = "Forecast preparation produced too little data for prediction."
            if uses_forecast_mode and _routing_override is None:
                return run_builder_pipeline(
                    user_query,
                    df,
                    max_retries=max_retries,
                    model=model,
                    workflow_context=workflow_context,
                    _routing_override=_build_forecast_fallback_override(prep_message),
                )
            pipeline_output = _build_simple_pipeline_output(
                output,
                intent=detected_intent,
                analysis_plan=analysis_plan,
                system_decision=system_decision,
                module_validation=module_validation,
                build_query=improved_query,
                build_plan=plan,
                generated_code=generated_code,
                code=prep_execution["code"],
                status="FAILED",
                test_error=prep_message,
                error=prep_message,
                workflow_context=workflow_context,
                fix_applied=prep_execution["fix_applied"],
                fix_status=prep_execution["fix_status"],
                fixed_code=prep_execution["fixed_code"],
                tool_used=analysis_plan["tool_used"],
                analysis_mode=analysis_plan["analysis_mode"],
                execution_plan=execution_plan,
                execution_trace=execution_trace,
                optimization=optimization_payload,
                excel_analysis=excel_analysis,
                dashboard=dashboard_payload,
            )
            _record_failure(failure_events, improved_query, prep_message, "forecast_preparation")
            reliability_payload = _build_analysis_reliability_payload(
                query=improved_query,
                df=df,
                result=None,
                plan=analysis_plan,
                workflow_context=workflow_context,
                preflight=preflight,
                data_score=_build_data_score(preflight, error=prep_message),
                failure_events=failure_events,
                use_llm_limitations=False,
            )
            contract = _build_analysis_contract_payload(
                query=improved_query,
                df=df,
                result=None,
                code=prep_execution["code"],
                plan=analysis_plan,
                preflight=preflight,
                method=analysis_method,
                error=prep_message,
                workflow_context=workflow_context,
                data_score=_build_data_score(preflight, error=prep_message),
                reliability_payload=reliability_payload,
                tool_used=analysis_plan["tool_used"],
                analysis_mode=analysis_plan["analysis_mode"],
                execution_plan=execution_plan,
                execution_trace=execution_trace,
                optimization=optimization_payload,
                excel_analysis=excel_analysis,
                dashboard=dashboard_payload,
            )
            pipeline_output = _apply_analysis_contract(pipeline_output, contract)
            pipeline_output["failure_events"] = list(failure_events)
            return _apply_pipeline_metadata(
                pipeline_output,
                query=improved_query,
                detected_intent=detected_intent,
                analysis_plan=analysis_plan,
                preflight=preflight,
                method=analysis_method,
                error=prep_message,
            )

        prediction_code = generate_sales_prediction_code(prepared_df, model=model)
        generated_code = _combine_stage_code(
            ("Forecast Preparation", prep_code),
            ("Sales Prediction", prediction_code),
        )
        prediction_execution = _execute_with_fix_details(
            improved_query,
            prepared_df,
            prediction_code,
            max_retries=max_retries,
            model=model,
        )
        final_code = _combine_stage_code(
            ("Forecast Preparation", prep_execution["code"]),
            ("Sales Prediction", prediction_execution["code"]),
        )
        final_result = prediction_execution["result"]
        error = prediction_execution["error"]
        fix_applied = prep_execution["fix_applied"] or prediction_execution["fix_applied"]
        fixed_code = final_code if fix_applied else None
        if fix_applied:
            stage_labels = []
            if prep_execution["fix_applied"]:
                stage_labels.append("forecast preparation")
            if prediction_execution["fix_applied"]:
                stage_labels.append("forecast prediction")
            joined_labels = " and ".join(stage_labels)
            if error:
                fix_status = f"Automatic fixes were applied to {joined_labels}, but execution still failed."
            else:
                fix_status = f"Automatic fix applied to {joined_labels}."
        else:
            fix_status = prediction_execution["fix_status"]
    elif not plan_already_executed and analysis_plan["tool_used"] == "PYTHON":
        execution_details = _execute_with_fix_details(
            improved_query,
            df,
            generated_code,
            max_retries=max_retries,
            model=model,
        )
        final_code = execution_details["code"]
        final_result = execution_details["result"]
        error = execution_details["error"]
        fix_applied = execution_details["fix_applied"]
        fix_status = execution_details["fix_status"]
        fixed_code = execution_details["fixed_code"]

    pipeline_output = _build_simple_pipeline_output(
        output,
        intent=detected_intent,
        analysis_plan=analysis_plan,
        system_decision=system_decision,
        module_validation=module_validation,
        build_query=improved_query,
        build_plan=plan,
        generated_code=generated_code,
        code=final_code,
        status="FAILED" if error else "PASSED",
        test_error=error,
        error=error,
        workflow_context=workflow_context,
        result=final_result,
        fix_applied=fix_applied,
        fix_status=fix_status,
        fixed_code=fixed_code,
        tool_used=analysis_plan["tool_used"],
        analysis_mode=analysis_plan["analysis_mode"],
        execution_plan=execution_plan,
        execution_trace=execution_trace,
        optimization=optimization_payload,
        excel_analysis=excel_analysis,
        dashboard=dashboard_payload,
    )

    if error:
        if uses_forecast_mode and _routing_override is None:
            return run_builder_pipeline(
                user_query,
                df,
                max_retries=max_retries,
                model=model,
                workflow_context=workflow_context,
                _routing_override=_build_forecast_fallback_override(error),
            )
        _record_failure(failure_events, improved_query, error, "execution")
        reliability_payload = _build_analysis_reliability_payload(
            query=improved_query,
            df=df,
            result=final_result,
            plan=analysis_plan,
            workflow_context=workflow_context,
            preflight=preflight,
            data_score=_build_data_score(preflight, error=error),
            failure_events=failure_events,
            use_llm_limitations=False,
        )
        contract = _build_analysis_contract_payload(
            query=improved_query,
            df=df,
            result=final_result,
            code=final_code,
            plan=analysis_plan,
            preflight=preflight,
            method=analysis_method,
            error=error,
            workflow_context=workflow_context,
            data_score=_build_data_score(preflight, error=error),
            reliability_payload=reliability_payload,
            tool_used=analysis_plan["tool_used"],
            analysis_mode=analysis_plan["analysis_mode"],
            execution_plan=execution_plan,
            execution_trace=execution_trace,
            optimization=optimization_payload,
            excel_analysis=excel_analysis,
            dashboard=dashboard_payload,
            system_decision=system_decision,
        )
        pipeline_output = _apply_analysis_contract(pipeline_output, contract)
        pipeline_output["failure_events"] = list(failure_events)
        return _apply_pipeline_metadata(
            pipeline_output,
            query=improved_query,
            detected_intent=detected_intent,
            analysis_plan=analysis_plan,
            preflight=preflight,
            method=analysis_method,
            error=error,
        )

    try:
        summary_override = summarize_for_non_technical_user(final_result, model=model)
    except Exception as summary_error:
        _record_failure(failure_events, improved_query, summary_error, "insight_summary")
        summary_override = None

    try:
        insights_override = generate_insights(
            improved_query,
            final_result,
            df,
            model=model,
        )
    except Exception as insights_error:
        _record_failure(failure_events, improved_query, insights_error, "insight_generation")
        insights_override = None

    reliability_payload = _build_analysis_reliability_payload(
        query=improved_query,
        df=df if not uses_forecast_mode else (prepared_df if 'prepared_df' in locals() and prepared_df is not None else df),
        result=final_result,
        plan=analysis_plan,
        insights=_coerce_contract_list(insights_override) if insights_override else None,
        workflow_context=workflow_context,
        preflight=preflight,
        data_score=_build_data_score(preflight, error=None),
        failure_events=failure_events,
        use_llm_limitations=bool((workflow_context or {}).get("enable_limitations_llm")),
    )

    contract = _build_analysis_contract_payload(
        query=improved_query,
        df=df,
        result=final_result,
        code=final_code,
        plan=analysis_plan,
        preflight=preflight,
        method=analysis_method,
        summary_override=summary_override,
        insights_override=insights_override,
        workflow_context=workflow_context,
        data_score=_build_data_score(preflight, error=None),
        reliability_payload=reliability_payload,
        tool_used=analysis_plan["tool_used"],
        analysis_mode=analysis_plan["analysis_mode"],
        execution_plan=execution_plan,
        execution_trace=execution_trace,
        optimization=optimization_payload,
        excel_analysis=excel_analysis,
        dashboard=dashboard_payload,
        system_decision=system_decision,
    )
    pipeline_output = _apply_analysis_contract(pipeline_output, contract)
    pipeline_output["failure_events"] = list(failure_events)
    return _apply_pipeline_metadata(
        pipeline_output,
        query=improved_query,
        detected_intent=detected_intent,
        analysis_plan=analysis_plan,
        preflight=preflight,
        method=analysis_method,
        error=None,
    )


def run_simple_pipeline(user_query, df, max_retries=3, model=DEFAULT_GEMINI_MODEL):
    sanitized_query = sanitize_query(user_query)
    prompt = build_master_prompt(sanitized_query, df)
    result, code, error = run_pipeline(
        sanitized_query,
        df,
        max_retries=max_retries,
        model=model,
    )

    return {
        "query": str(user_query),
        "sanitized_query": sanitized_query,
        "prompt": prompt,
        "code": code,
        "result": result,
        "error": error,
    }
