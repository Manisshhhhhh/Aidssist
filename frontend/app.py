import base64
import json
import os
import sys
from dataclasses import asdict
from html import escape
from io import BytesIO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image
from backend.runtime_preflight import build_setup_commands, get_runtime_preflight_issues


PRE_FLIGHT_ISSUES = get_runtime_preflight_issues("streamlit")
if PRE_FLIGHT_ISSUES:
    st.set_page_config(
        page_title="Aidssist Setup Required",
        layout="wide",
    )
    st.error("Aidssist needs a supported Python environment before the app can start.")
    st.markdown("\n".join(f"- {escape(issue)}" for issue in PRE_FLIGHT_ISSUES))
    st.info("Recreate the virtual environment with Python 3.11 and reinstall the project requirements.")
    st.code(build_setup_commands("streamlit"), language="bash")
    st.stop()

from backend.chart_customization import (
    ChartCustomization,
    build_custom_result_chart,
    get_chart_customization_options,
)
from backend.aidssist_runtime.client import (
    analysis_service_enabled,
    get_runtime_configuration_status,
    run_remote_analysis,
    run_remote_forecast,
)
from backend.forecasting import (
    FREQUENCY_LABELS,
    HORIZON_LABELS,
    ForecastConfig,
    build_forecast_config_signature,
    forecast_config_to_dict,
    forecast_config_from_dict,
    get_forecast_mapping_options,
    persist_forecast_artifact,
    run_forecast_pipeline,
    validate_forecast_config,
)
from backend.data_quality import (
    CleaningOptions,
    create_load_failure_finding,
    finding_to_dict,
    has_blocking_findings,
    summarize_findings,
    validate_dataframe,
    apply_cleaning_plan,
)
from backend.data_sources import (
    CSVSourceConfig,
    DataSourceError,
    ExcelSourceConfig,
    SQLSourceConfig,
    deserialize_source_config,
    load_dataframe_from_source,
    persist_file_source_snapshot,
    serialize_source_config,
    validate_sql_source_config,
)
from backend.dashboard_helpers import (
    BAR_HIGHLIGHT_FIELD,
    ChartSpec,
    LINE_PEAK_FIELD,
    build_column_insight,
    build_chart_takeaway,
    profile_analysis_result,
    profile_dataset,
)
from backend.prompt_pipeline import (
    get_default_test_cases,
    get_provider_configuration_status,
    run_builder_pipeline,
)
from backend.services.data_intelligence import detect_dataset_type
from backend.services.mode_router import decide_analysis_mode
from backend.workflow_store import WorkflowStore
from frontend.step_flow import (
    WORKFLOW_STEPS,
    build_step_label,
    build_step_states,
    resolve_active_step,
)
from frontend.companion_console import render_companion_console
from frontend.workflow_state import (
    build_cleaning_options_signature,
    is_forecast_result_current,
    is_cleaned_dataset_current,
    should_reuse_cleaning_preview,
)


alt.data_transformers.disable_max_rows()

APP_DIR = Path(__file__).resolve().parent
APP_STATE_DIR = PROJECT_ROOT / ".aidssist"
WORKFLOW_DB_PATH = APP_STATE_DIR / "workflow_store.sqlite3"
SOURCE_SNAPSHOT_DIR = APP_STATE_DIR / "source_snapshots"
LOGO_PATH = APP_DIR / "assets" / "logo.png"
FAVICON_PATH = APP_DIR / "assets" / "favicon.png"
LOADING_VIDEO_PATH = APP_DIR / "assets" / "loading.mp4"
DATASET_PROFILE_SCHEMA_VERSION = "2026-03-30-v1"
DATASET_PROFILE_REQUIRED_FIELDS = (
    "dataset_name",
    "dataset_key",
    "row_count",
    "column_count",
    "missing_cell_count",
    "duplicate_row_count",
    "numeric_column_count",
    "categorical_column_count",
    "datetime_column_count",
    "missing_by_column",
    "data_dictionary",
    "overview_charts",
    "numeric_charts",
    "categorical_charts",
    "datetime_charts",
)
PAGE_ICON = None
WORKFLOW_STORE = WorkflowStore(os.getenv("AIDSSIST_DATABASE_URL") or WORKFLOW_DB_PATH)
EDITOR_STATE_KEYS = (
    "workflow_name",
    "source_type",
    "csv_delimiter",
    "csv_encoding",
    "excel_sheet_name",
    "sql_host",
    "sql_port",
    "sql_database",
    "sql_username",
    "sql_password",
    "sql_table_name",
    "sql_query",
    "sql_limit",
    "parse_dates",
    "coerce_numeric_text",
    "trim_strings",
    "drop_duplicates_enabled",
    "fill_numeric_nulls",
    "fill_text_nulls",
    "drop_null_rows_over",
    "drop_null_columns_over",
    "forecast_date_column",
    "forecast_target_column",
    "forecast_driver_columns",
    "forecast_frequency",
    "forecast_horizon",
    "forecast_model_strategy",
    "forecast_training_mode",
    "query_input",
    "chart_kind_pref",
    "chart_x_pref",
    "chart_y_pref",
    "chart_aggregation_pref",
    "chart_palette_pref",
    "chart_title_pref",
    "chart_x_title_pref",
    "chart_y_title_pref",
    "export_csv_enabled",
    "export_json_enabled",
)
DEFAULT_EDITOR_STATE = {
    "workflow_name": "",
    "source_type": "csv",
    "csv_delimiter": ",",
    "csv_encoding": "utf-8",
    "excel_sheet_name": "0",
    "sql_host": "localhost",
    "sql_port": 5432,
    "sql_database": "",
    "sql_username": "",
    "sql_password": "",
    "sql_table_name": "",
    "sql_query": "",
    "sql_limit": 1000,
    "parse_dates": True,
    "coerce_numeric_text": True,
    "trim_strings": True,
    "drop_duplicates_enabled": True,
    "fill_numeric_nulls": "none",
    "fill_text_nulls": "none",
    "drop_null_rows_over": 1.0,
    "drop_null_columns_over": 1.0,
    "forecast_date_column": "",
    "forecast_target_column": "",
    "forecast_driver_columns": [],
    "forecast_frequency": "auto",
    "forecast_horizon": "next_month",
    "forecast_model_strategy": "hybrid",
    "forecast_training_mode": "auto",
    "query_input": "",
    "chart_kind_pref": "auto",
    "chart_x_pref": "",
    "chart_y_pref": "",
    "chart_aggregation_pref": "sum",
    "chart_palette_pref": "orange",
    "chart_title_pref": "",
    "chart_x_title_pref": "",
    "chart_y_title_pref": "",
    "export_csv_enabled": True,
    "export_json_enabled": True,
}

if FAVICON_PATH.exists():
    PAGE_ICON = Image.open(FAVICON_PATH)
elif LOGO_PATH.exists():
    PAGE_ICON = Image.open(LOGO_PATH)

st.set_page_config(
    page_title="Aidssist",
    page_icon=PAGE_ICON,
    layout="wide",
)


def apply_custom_styles():
    st.markdown(
        """
        <style>
        :root {
            --panel-bg: rgba(9, 17, 31, 0.92);
            --panel-bg-soft: rgba(12, 22, 40, 0.78);
            --panel-border: rgba(148, 163, 184, 0.16);
            --panel-shadow: 0 24px 52px rgba(2, 6, 23, 0.34);
            --accent: #f97316;
            --accent-strong: #fb923c;
            --accent-soft: rgba(249, 115, 22, 0.14);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --text-soft: #cbd5e1;
            --success: #22c55e;
            --warning: #f59e0b;
            --error: #ef4444;
            --surface: rgba(15, 23, 42, 0.72);
        }

        * {
            box-sizing: border-box;
        }

        html, body, [class*="css"] {
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
            -webkit-text-size-adjust: 100%;
        }

        .stApp {
            background-color: #08111f;
            background:
                radial-gradient(circle at top left, rgba(249, 115, 22, 0.12), transparent 30%),
                radial-gradient(circle at top right, rgba(56, 189, 248, 0.08), transparent 26%),
                linear-gradient(180deg, #07101d 0%, #0a1324 44%, #08111f 100%);
            color: var(--text-main);
            font-family: "Avenir Next", -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Arial, sans-serif;
            min-height: 100vh;
            overflow-x: hidden;
            -webkit-overflow-scrolling: touch;
        }

        [data-testid="stAppViewContainer"] {
            overflow-x: hidden;
        }

        [data-testid="stHeader"] {
            background: rgba(7, 16, 29, 0.82);
            border-bottom: 1px solid rgba(148, 163, 184, 0.08);
            -webkit-backdrop-filter: blur(14px);
            backdrop-filter: blur(14px);
            box-shadow: 0 12px 28px rgba(2, 6, 23, 0.18);
            padding-top: env(safe-area-inset-top);
        }

        section[data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(11, 20, 36, 0.98), rgba(8, 16, 30, 0.96)),
                radial-gradient(circle at top, rgba(249, 115, 22, 0.08), transparent 42%);
            border-right: 1px solid rgba(148, 163, 184, 0.09);
        }

        section[data-testid="stSidebar"] > div {
            padding-top: max(1.15rem, env(safe-area-inset-top));
            padding-bottom: max(1.15rem, env(safe-area-inset-bottom));
            padding-left: max(0.8rem, env(safe-area-inset-left));
            padding-right: max(0.8rem, env(safe-area-inset-right));
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] > label {
            border-radius: 16px;
            border: 1px solid transparent;
            padding: 0.45rem 0.55rem;
            margin-bottom: 0.35rem;
            background: rgba(15, 23, 42, 0.36);
            transition: background 0.2s ease, border-color 0.2s ease, transform 0.2s ease;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] > label p {
            color: var(--text-soft);
            font-size: 0.95rem;
            font-weight: 600;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
            background: rgba(15, 23, 42, 0.58);
            border-color: rgba(249, 115, 22, 0.18);
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
            background: linear-gradient(135deg, rgba(249, 115, 22, 0.16), rgba(251, 146, 60, 0.08));
            border-color: rgba(249, 115, 22, 0.32);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p {
            color: var(--text-main);
        }

        .block-container {
            width: 100%;
            padding-top: 1.45rem;
            padding-left: max(1.25rem, env(safe-area-inset-left));
            padding-right: max(1.25rem, env(safe-area-inset-right));
            padding-bottom: max(2.5rem, env(safe-area-inset-bottom));
            max-width: 1320px;
        }

        .page-top-spacer {
            height: clamp(3.1rem, 6.5vh, 4.4rem);
        }

        h1, h2, h3 {
            letter-spacing: -0.02em;
        }

        img, svg, canvas {
            max-width: 100%;
        }

        div[data-testid="column"],
        div[data-testid="stHorizontalBlock"] > div {
            min-width: 0;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: linear-gradient(180deg, rgba(10, 18, 33, 0.96), rgba(8, 15, 28, 0.92));
            border: 1px solid var(--panel-border);
            border-radius: 26px;
            box-shadow: var(--panel-shadow);
            overflow: hidden;
            animation: aidssist-fade-up 0.55s ease both;
        }

        div[data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.94), rgba(10, 16, 30, 0.88));
            border: 1px solid rgba(148, 163, 184, 0.14);
            border-radius: 22px;
            padding: 0.35rem 0.4rem;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
            animation: aidssist-fade-up 0.65s ease both;
        }

        div[data-testid="stMetricLabel"] {
            color: var(--text-muted);
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        div[data-testid="stMetricValue"] {
            color: var(--text-main);
            font-size: 1.6rem;
        }

        div[data-testid="stTextInput"] input,
        div[data-testid="stTextArea"] textarea,
        div[data-testid="stNumberInput"] input {
            background: rgba(15, 23, 42, 0.9);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 16px;
            color: var(--text-main);
            font-size: 0.98rem;
            min-height: 3rem;
        }

        div[data-baseweb="select"] > div,
        div[data-testid="stMultiSelect"] [data-baseweb="select"] > div {
            background: rgba(15, 23, 42, 0.9);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 16px;
            min-height: 3rem;
        }

        div[data-testid="stTextInput"] input::placeholder,
        div[data-testid="stTextArea"] textarea::placeholder {
            color: #64748b;
        }

        div[data-testid="stButton"] > button,
        div[data-testid="stFormSubmitButton"] > button {
            border-radius: 18px;
            border: 1px solid rgba(249, 115, 22, 0.25);
            background: linear-gradient(135deg, rgba(249, 115, 22, 0.22), rgba(251, 146, 60, 0.12));
            color: var(--text-main);
            font-weight: 600;
            min-height: 3rem;
            touch-action: manipulation;
            -webkit-tap-highlight-color: transparent;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }

        @media (hover: hover) and (pointer: fine) {
            div[data-testid="stButton"] > button:hover,
            div[data-testid="stFormSubmitButton"] > button:hover {
                border-color: rgba(249, 115, 22, 0.48);
                transform: translateY(-1px);
            }

            section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
                transform: translateY(-1px);
            }
        }

        div[data-testid="stButton"] > button:disabled,
        div[data-testid="stFormSubmitButton"] > button:disabled {
            opacity: 0.5;
            transform: none;
        }

        div[data-testid="stExpander"] {
            border: 1px solid rgba(148, 163, 184, 0.14);
            border-radius: 18px;
            background: rgba(10, 16, 30, 0.72);
        }

        .section-kicker {
            color: #f59e0b;
            font-size: 0.75rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin-bottom: 0.4rem;
            font-weight: 700;
        }

        .section-title {
            color: var(--text-main);
            font-size: 1.7rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
            overflow-wrap: anywhere;
        }

        .section-caption {
            color: var(--text-muted);
            font-size: 0.98rem;
            line-height: 1.6;
            margin-bottom: 0;
            overflow-wrap: anywhere;
            max-width: 52rem;
        }

        .hero-title {
            color: var(--text-main);
            font-size: 2.3rem;
            font-weight: 800;
            margin-bottom: 0.45rem;
            overflow-wrap: anywhere;
            line-height: 1.1;
        }

        .hero-caption {
            color: var(--text-muted);
            font-size: 1rem;
            line-height: 1.6;
            margin-bottom: 0;
            overflow-wrap: anywhere;
            max-width: 42rem;
        }

        .hero-shell {
            display: grid;
            grid-template-columns: 164px minmax(0, 1fr);
            gap: 1.5rem;
            align-items: center;
        }

        .hero-shell-no-logo {
            grid-template-columns: 1fr;
        }

        .hero-brand-mark {
            width: 164px;
            height: 164px;
            border-radius: 28px;
            display: flex;
            align-items: center;
            justify-content: center;
            background:
                linear-gradient(180deg, rgba(23, 36, 61, 0.96), rgba(11, 19, 35, 0.92)),
                radial-gradient(circle at top, rgba(249, 115, 22, 0.12), transparent 52%);
            border: 1px solid rgba(148, 163, 184, 0.14);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
        }

        .hero-brand-mark img {
            width: 118px;
            height: 118px;
            object-fit: cover;
            border-radius: 22px;
            display: block;
        }

        .hero-copy {
            min-width: 0;
        }

        .hero-meta-row {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            flex-wrap: wrap;
            margin-bottom: 0.45rem;
        }

        .hero-status-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.35rem 0.72rem;
            border-radius: 999px;
            border: 1px solid rgba(249, 115, 22, 0.2);
            background: rgba(249, 115, 22, 0.08);
            color: #fdba74;
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.02em;
        }

        .hero-footnote {
            margin-top: 1rem;
            color: var(--text-soft);
            font-size: 0.9rem;
            line-height: 1.5;
        }

        .sidebar-intro {
            padding: 0.95rem 1rem;
            border-radius: 20px;
            border: 1px solid rgba(148, 163, 184, 0.12);
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.82), rgba(9, 17, 31, 0.78));
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
            margin-bottom: 1rem;
        }

        .sidebar-intro-title {
            color: var(--text-main);
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }

        .sidebar-intro-copy {
            color: var(--text-muted);
            font-size: 0.9rem;
            line-height: 1.5;
            margin: 0;
        }

        .sidebar-dataset-chip {
            margin-top: 0.85rem;
            padding: 0.85rem 0.95rem;
            border-radius: 18px;
            border: 1px solid rgba(148, 163, 184, 0.12);
            background: rgba(15, 23, 42, 0.58);
        }

        .sidebar-dataset-chip span {
            display: block;
            color: var(--text-muted);
            font-size: 0.73rem;
            text-transform: uppercase;
            letter-spacing: 0.09em;
            margin-bottom: 0.28rem;
        }

        .sidebar-dataset-chip strong {
            display: block;
            color: var(--text-main);
            font-size: 0.98rem;
            line-height: 1.35;
            margin-bottom: 0.2rem;
        }

        .sidebar-dataset-chip small {
            display: block;
            color: var(--text-muted);
            font-size: 0.82rem;
            line-height: 1.45;
        }

        .insight-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.9rem;
            margin-top: 1rem;
        }

        .insight-card {
            min-width: 0;
            border-radius: 20px;
            border: 1px solid rgba(148, 163, 184, 0.12);
            background: linear-gradient(180deg, rgba(16, 24, 44, 0.92), rgba(10, 18, 33, 0.86));
            padding: 1rem 1rem 0.95rem;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
        }

        .insight-card-label {
            color: #fdba74;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-weight: 700;
            margin-bottom: 0.45rem;
        }

        .insight-card-text {
            color: var(--text-soft);
            font-size: 0.95rem;
            line-height: 1.6;
            margin: 0;
        }

        .chart-note {
            margin-top: 0.85rem;
            padding: 0.8rem 0.9rem;
            border-radius: 16px;
            border: 1px solid rgba(249, 115, 22, 0.14);
            background: rgba(249, 115, 22, 0.07);
            color: var(--text-soft);
            font-size: 0.92rem;
            line-height: 1.55;
        }

        .benchmark-type {
            display: inline-flex;
            align-items: center;
            padding: 0.22rem 0.6rem;
            border-radius: 999px;
            background: rgba(249, 115, 22, 0.16);
            color: #fdba74;
            font-size: 0.74rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .benchmark-query {
            color: var(--text-main);
            font-size: 1rem;
            line-height: 1.55;
            margin: 0.8rem 0 1rem;
            min-height: 3rem;
            overflow-wrap: anywhere;
        }

        .loading-shell {
            display: flex;
            justify-content: center;
            padding: 1rem 0 0.25rem;
        }

        .loading-card {
            width: 100%;
            max-width: 420px;
            text-align: center;
            padding: 1.5rem 1.35rem;
            border-radius: 28px;
            border: 1px solid rgba(148, 163, 184, 0.18);
            background: linear-gradient(180deg, rgba(12, 24, 42, 0.94), rgba(8, 17, 31, 0.9));
            box-shadow: 0 22px 44px rgba(0, 0, 0, 0.22);
        }

        .loading-media-shell {
            width: 112px;
            height: 112px;
            border-radius: 999px;
            overflow: hidden;
            margin: 0 auto;
            border: 1px solid rgba(148, 163, 184, 0.18);
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.18);
            background: rgba(8, 17, 31, 0.9);
        }

        .loading-logo,
        .loading-video {
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
        }

        .loading-logo {
            animation: aidssist-spin 2.6s linear infinite;
        }

        .loading-title {
            color: var(--text-main);
            font-size: 1.05rem;
            font-weight: 700;
            margin-top: 1rem;
        }

        .loading-caption {
            color: var(--text-muted);
            font-size: 0.95rem;
            line-height: 1.6;
            margin: 0.45rem 0 0;
        }

        div[data-testid="stDataFrame"] {
            border-radius: 18px;
            overflow: hidden;
        }

        section[data-testid="stFileUploader"],
        div[data-testid="stFileUploader"] {
            width: 100%;
        }

        div[data-testid="stFileUploaderDropzone"] {
            border-radius: 22px;
            padding: 1rem 1.05rem;
            min-height: 5.8rem;
            border: 1px dashed rgba(148, 163, 184, 0.2);
            background: linear-gradient(180deg, rgba(17, 25, 44, 0.92), rgba(12, 20, 35, 0.88));
        }

        section[data-testid="stFileUploader"] button,
        div[data-testid="stFileUploader"] button {
            width: 100%;
            min-height: 3rem;
            white-space: normal;
            touch-action: manipulation;
            -webkit-tap-highlight-color: transparent;
        }

        button[data-baseweb="tab"] {
            border-radius: 14px;
            background: rgba(15, 23, 42, 0.68);
            border: 1px solid rgba(148, 163, 184, 0.08);
            color: var(--text-main);
            margin-right: 0.35rem;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }

        button[data-baseweb="tab"]:hover {
            border-color: rgba(249, 115, 22, 0.35);
            transform: translateY(-1px);
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(135deg, rgba(249, 115, 22, 0.22), rgba(251, 146, 60, 0.12));
            border-color: rgba(249, 115, 22, 0.4);
        }

        @supports not ((backdrop-filter: blur(1px)) or (-webkit-backdrop-filter: blur(1px))) {
            [data-testid="stHeader"] {
                background: rgba(7, 16, 29, 0.96);
            }
        }

        @media (max-width: 1080px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
                max-width: 100%;
            }

            .hero-title {
                font-size: 2rem;
            }

            .hero-shell {
                grid-template-columns: 136px minmax(0, 1fr);
                gap: 1.2rem;
            }

            .hero-brand-mark {
                width: 136px;
                height: 136px;
            }

            .hero-brand-mark img {
                width: 100px;
                height: 100px;
            }
        }

        @media (max-width: 768px) {
            .block-container {
                padding-top: 1rem;
                padding-left: 0.85rem;
                padding-right: 0.85rem;
                padding-bottom: 1.8rem;
            }

            .page-top-spacer {
                height: max(1.9rem, env(safe-area-inset-top));
            }

            .hero-title {
                font-size: 1.8rem;
            }

            .section-title {
                font-size: 1.35rem;
            }

            .hero-caption,
            .section-caption,
            .benchmark-query {
                font-size: 0.95rem;
            }

            div[data-testid="stHorizontalBlock"] {
                flex-wrap: wrap;
                gap: 0.85rem;
            }

            .hero-shell {
                grid-template-columns: 1fr;
                gap: 1rem;
            }

            .insight-grid {
                grid-template-columns: 1fr;
            }

            .hero-brand-mark {
                width: 112px;
                height: 112px;
                border-radius: 22px;
            }

            .hero-brand-mark img {
                width: 84px;
                height: 84px;
                border-radius: 16px;
            }

            .hero-footnote {
                margin-top: 0.8rem;
            }

            div[data-testid="column"] {
                width: 100% !important;
                flex: 1 1 100% !important;
            }

            div[data-testid="stMetric"] {
                min-height: 7.2rem;
            }

            .loading-media-shell {
                width: 88px;
                height: 88px;
            }
        }

        @media (max-width: 520px) {
            .block-container {
                padding-left: 0.7rem;
                padding-right: 0.7rem;
            }

            .page-top-spacer {
                height: max(1.15rem, env(safe-area-inset-top));
            }

            [data-testid="stHeader"] {
                padding-left: 0.35rem;
                padding-right: 0.35rem;
            }

            .hero-title {
                font-size: 1.62rem;
            }

            .hero-caption,
            .section-caption,
            .sidebar-intro-copy,
            .insight-card-text,
            .chart-note {
                font-size: 0.92rem;
            }

            div[data-testid="stFileUploaderDropzone"] {
                padding: 0.75rem 0.8rem;
            }

            div[data-testid="stButton"] > button,
            div[data-testid="stFormSubmitButton"] > button,
            section[data-testid="stFileUploader"] button,
            div[data-testid="stFileUploader"] button,
            div[data-testid="stTextInput"] input,
            div[data-testid="stTextArea"] textarea,
            div[data-testid="stNumberInput"] input,
            div[data-baseweb="select"] > div {
                min-height: 3rem;
                font-size: 16px;
            }
        }

        @media (prefers-reduced-motion: reduce) {
            .loading-logo {
                animation: none;
            }

            div[data-testid="stButton"] > button,
            div[data-testid="stFormSubmitButton"] > button {
                transition: none;
            }
        }

        @supports (-webkit-touch-callout: none) {
            [data-testid="stHeader"] {
                background: rgba(7, 16, 29, 0.92);
            }

            .block-container {
                padding-left: max(0.9rem, env(safe-area-inset-left));
                padding-right: max(0.9rem, env(safe-area-inset-right));
                padding-bottom: max(2rem, env(safe-area-inset-bottom));
            }

            section[data-testid="stSidebar"] {
                -webkit-overflow-scrolling: touch;
            }

            div[data-testid="stTextArea"] textarea {
                min-height: 7.6rem;
            }
        }

        @keyframes aidssist-spin {
            from {
                transform: rotate(0deg);
            }
            to {
                transform: rotate(360deg);
            }
        }

        @keyframes aidssist-fade-up {
            from {
                opacity: 0;
                transform: translateY(14px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_app_header():
    logo_markup = ""
    hero_shell_class = "hero-shell hero-shell-no-logo"
    if LOGO_PATH.exists():
        logo_data_uri = get_image_data_uri(str(LOGO_PATH))
        logo_markup = (
            '<div class="hero-brand-mark">'
            f'<img src="{logo_data_uri}" alt="Aidssist logo" />'
            "</div>"
        )
        hero_shell_class = "hero-shell"

    with st.container(border=True):
        st.markdown(
            f"""
            <div class="{hero_shell_class}">
                {logo_markup}
                <div class="hero-copy">
                    <div class="hero-meta-row">
                        <div class="section-kicker">AI Data Workspace</div>
                        <div class="hero-status-pill">Guided workflow</div>
                    </div>
                    <div class="hero-title">Aidssist</div>
                    <p class="hero-caption">Upload a dataset, validate quality, explore the signal, and ask decision-focused questions in one clean AI workspace.</p>
                    <div class="hero-footnote">Designed for a polished desktop experience and touch-friendly use on iPhone and iPad.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_section_heading(title, caption, kicker=None):
    if kicker:
        st.markdown(f'<div class="section-kicker">{escape(kicker)}</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="section-title">{escape(title)}</div>', unsafe_allow_html=True)
    st.markdown(f'<p class="section-caption">{escape(caption)}</p>', unsafe_allow_html=True)


def render_insight_cards(title, caption, insights, kicker=None):
    insight_items = [item for item in insights if item]
    if not insight_items:
        return

    with st.container(border=True):
        render_section_heading(title, caption, kicker=kicker)
        card_markup = "".join(
            (
                '<div class="insight-card">'
                f'<div class="insight-card-label">Insight {index + 1}</div>'
                f'<p class="insight-card-text">{escape(insight_text)}</p>'
                "</div>"
            )
            for index, insight_text in enumerate(insight_items)
        )
        st.markdown(f'<div class="insight-grid">{card_markup}</div>', unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def get_image_data_uri(image_path):
    image_file = Path(image_path)
    mime_type = "image/png"
    if image_file.suffix.lower() == ".jpg" or image_file.suffix.lower() == ".jpeg":
        mime_type = "image/jpeg"

    encoded_bytes = base64.b64encode(image_file.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded_bytes}"


@st.cache_data(show_spinner=False)
def get_video_data_uri(video_path):
    video_file = Path(video_path)
    encoded_bytes = base64.b64encode(video_file.read_bytes()).decode("ascii")
    return f"data:video/mp4;base64,{encoded_bytes}"


def render_loading_indicator(container, message):
    if not LOADING_VIDEO_PATH.exists() and not LOGO_PATH.exists():
        container.info(message)
        return

    safe_message = escape(message)
    loading_media = ""

    if LOADING_VIDEO_PATH.exists():
        video_data_uri = get_video_data_uri(str(LOADING_VIDEO_PATH))
        loading_media = (
            f'<div class="loading-media-shell">'
            f'<video class="loading-video" autoplay loop muted playsinline>'
            f'<source src="{video_data_uri}" type="video/mp4" />'
            f"</video>"
            f"</div>"
        )
    elif LOGO_PATH.exists():
        logo_data_uri = get_image_data_uri(str(LOGO_PATH))
        loading_media = (
            f'<div class="loading-media-shell">'
            f'<img src="{logo_data_uri}" alt="Aidssist loading logo" class="loading-logo" />'
            f"</div>"
        )

    container.markdown(
        f"""
        <div class="loading-shell">
            <div class="loading-card">
                {loading_media}
                <div class="loading-title">{safe_message}</div>
                <p class="loading-caption">Building your prompt, testing the generated code, and preparing the final response.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result(result):
    if isinstance(result, pd.DataFrame):
        visible_rows = max(1, min(len(result), 12))
        height = min(540, 90 + visible_rows * 36)
        st.dataframe(result, use_container_width=True, height=height)
        return

    if isinstance(result, pd.Series):
        visible_rows = max(1, min(len(result), 12))
        height = min(540, 90 + visible_rows * 36)
        st.dataframe(
            result.to_frame(name=result.name or "value"),
            use_container_width=True,
            height=height,
        )
        return

    if hasattr(result, "figure") and result.figure is not None:
        st.pyplot(result.figure)
        return

    if hasattr(result, "savefig"):
        st.pyplot(result)
        return

    if isinstance(result, (list, dict)):
        st.json(result)
        return

    st.write(str(result))


@st.cache_data(show_spinner=False)
def load_uploaded_dataframe(file_name, file_bytes):
    del file_name
    return pd.read_csv(BytesIO(file_bytes), low_memory=False)


@st.cache_data(show_spinner=False)
def get_cached_dataset_profile(dataset_name, dataset_key, profile_schema_version, df):
    del profile_schema_version
    return profile_dataset(df, dataset_name=dataset_name, dataset_key=dataset_key)


def ensure_dataset_profile(profile, df, dataset_name, dataset_key):
    if profile is None:
        return profile_dataset(df, dataset_name=dataset_name, dataset_key=dataset_key)

    missing_fields = [
        field_name
        for field_name in DATASET_PROFILE_REQUIRED_FIELDS
        if not hasattr(profile, field_name)
    ]
    if missing_fields:
        return profile_dataset(df, dataset_name=dataset_name, dataset_key=dataset_key)

    return profile


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def figure_to_png_bytes(figure) -> bytes:
    buffer = BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    return buffer.getvalue()


def build_result_download_payload(result):
    result_profile = profile_analysis_result(result)

    if hasattr(result, "figure") and result.figure is not None:
        return (
            "analysis_result_chart.png",
            figure_to_png_bytes(result.figure),
            "image/png",
            "Download analysis chart PNG",
        )

    if hasattr(result, "savefig"):
        return (
            "analysis_result_chart.png",
            figure_to_png_bytes(result),
            "image/png",
            "Download analysis chart PNG",
        )

    if result_profile.table is not None:
        return (
            "analysis_result.csv",
            dataframe_to_csv_bytes(result_profile.table),
            "text/csv",
            "Download analysis result CSV",
        )

    if isinstance(result, (dict, list)):
        return (
            "analysis_result.json",
            json.dumps(result, indent=2, default=str).encode("utf-8"),
            "application/json",
            "Download analysis result JSON",
        )

    return (
        "analysis_result.txt",
        str(result).encode("utf-8"),
        "text/plain",
        "Download analysis result text",
    )


def build_chart_download_payloads(chart_spec: ChartSpec | None, file_stem: str, data_label: str):
    if chart_spec is None:
        return []

    payloads = [
        (
            data_label,
            dataframe_to_csv_bytes(chart_spec.data),
            f"{file_stem}_chart_data.csv",
            "text/csv",
        )
    ]

    try:
        chart_json = build_chart(chart_spec).to_json(indent=2).encode("utf-8")
    except Exception:
        chart_json = None

    if chart_json:
        payloads.append(
            (
                "Download chart spec JSON",
                chart_json,
                f"{file_stem}_chart_spec.json",
                "application/json",
            )
        )

    return payloads


def build_chart(chart_spec: ChartSpec):
    chart_color = chart_spec.color
    base_chart = alt.Chart(chart_spec.data)

    def finalize_chart(chart):
        return (
            chart.properties(height=300, title=chart_spec.title)
            .configure_view(stroke=None)
            .configure_axis(
                labelColor="#CBD5E1",
                titleColor="#E2E8F0",
                gridColor="rgba(148, 163, 184, 0.12)",
                domainColor="rgba(148, 163, 184, 0.12)",
                tickColor="rgba(148, 163, 184, 0.12)",
            )
            .configure_title(color="#F8FAFC", fontSize=18, anchor="start")
        )

    if chart_spec.kind == "line":
        line_chart = base_chart.mark_line(
            interpolate="monotone",
            strokeCap="round",
            strokeJoin="round",
            strokeWidth=3,
            color=chart_color or "#FB923C",
        ).encode(
            x=alt.X(f"{chart_spec.x}:T", title=chart_spec.x_title),
            y=alt.Y(f"{chart_spec.y}:Q", title=chart_spec.y_title),
            tooltip=[
                alt.Tooltip(f"{chart_spec.x}:T", title=chart_spec.x_title),
                alt.Tooltip(f"{chart_spec.y}:Q", title=chart_spec.y_title),
            ],
        )

        if LINE_PEAK_FIELD in chart_spec.data.columns:
            peak_points = base_chart.transform_filter(
                alt.FieldEqualPredicate(field=LINE_PEAK_FIELD, equal="Peak")
            ).mark_circle(size=110, color="#F8FAFC", stroke="#FB923C", strokeWidth=3).encode(
                x=alt.X(f"{chart_spec.x}:T", title=chart_spec.x_title),
                y=alt.Y(f"{chart_spec.y}:Q", title=chart_spec.y_title),
                tooltip=[
                    alt.Tooltip(f"{chart_spec.x}:T", title=chart_spec.x_title),
                    alt.Tooltip(f"{chart_spec.y}:Q", title=chart_spec.y_title),
                ],
            )
            return finalize_chart(line_chart + peak_points)

        point_layer = base_chart.mark_circle(
            filled=True,
            size=60,
            color=chart_color or "#FB923C",
        ).encode(
            x=alt.X(f"{chart_spec.x}:T", title=chart_spec.x_title),
            y=alt.Y(f"{chart_spec.y}:Q", title=chart_spec.y_title),
            tooltip=[
                alt.Tooltip(f"{chart_spec.x}:T", title=chart_spec.x_title),
                alt.Tooltip(f"{chart_spec.y}:Q", title=chart_spec.y_title),
            ],
        )
        return finalize_chart(line_chart + point_layer)

    if chart_spec.kind == "bar":
        color_encoding = alt.value(chart_color or "#F97316")
        if BAR_HIGHLIGHT_FIELD in chart_spec.data.columns:
            color_encoding = alt.Color(
                f"{BAR_HIGHLIGHT_FIELD}:N",
                scale=alt.Scale(
                    domain=["Top segment", "Other"],
                    range=["#F97316", "#475569"],
                ),
                legend=None,
            )

        return finalize_chart(
            base_chart.mark_bar(
                cornerRadiusTopLeft=8,
                cornerRadiusTopRight=8,
            ).encode(
                x=alt.X(
                    f"{chart_spec.x}:N",
                    title=chart_spec.x_title,
                    sort="-y",
                    axis=alt.Axis(labelAngle=-20),
                ),
                y=alt.Y(f"{chart_spec.y}:Q", title=chart_spec.y_title),
                color=color_encoding,
                tooltip=[
                    alt.Tooltip(f"{chart_spec.x}:N", title=chart_spec.x_title),
                    alt.Tooltip(f"{chart_spec.y}:Q", title=chart_spec.y_title),
                ],
            )
        )

    histogram_chart = base_chart.mark_bar(
        cornerRadiusTopLeft=8,
        cornerRadiusTopRight=8,
        color=chart_color or "#22C55E",
    ).encode(
        x=alt.X(f"{chart_spec.x}:Q", bin=alt.Bin(maxbins=24), title=chart_spec.x_title),
        y=alt.Y("count():Q", title=chart_spec.y_title),
        tooltip=[
            alt.Tooltip(f"{chart_spec.x}:Q", bin=True, title=chart_spec.x_title),
            alt.Tooltip("count():Q", title=chart_spec.y_title),
        ],
    )

    numeric_series = pd.to_numeric(chart_spec.data[chart_spec.x], errors="coerce").dropna()
    if numeric_series.empty:
        return finalize_chart(histogram_chart)

    median_rule = alt.Chart(
        pd.DataFrame({"median": [float(numeric_series.median())]})
    ).mark_rule(color="#F59E0B", strokeDash=[6, 4], size=2).encode(
        x=alt.X("median:Q", title=chart_spec.x_title)
    )
    return finalize_chart(histogram_chart + median_rule)


def build_column_type_chart(profile):
    chart_data = profile.column_type_breakdown[profile.column_type_breakdown["count"] > 0]

    return (
        alt.Chart(chart_data)
        .mark_arc(innerRadius=66, outerRadius=110)
        .encode(
            theta=alt.Theta("count:Q"),
            color=alt.Color(
                "type:N",
                scale=alt.Scale(
                    domain=["Numeric", "Categorical", "Datetime"],
                    range=["#F97316", "#38BDF8", "#22C55E"],
                ),
                legend=alt.Legend(title=None, labelColor="#E2E8F0"),
            ),
            tooltip=[
                alt.Tooltip("type:N", title="Type"),
                alt.Tooltip("count:Q", title="Columns"),
            ],
        )
        .properties(height=300, title="Column type distribution")
        .configure_view(stroke=None)
        .configure_title(color="#F8FAFC", fontSize=18, anchor="start")
        .configure_legend(labelColor="#E2E8F0")
    )


def build_missing_values_chart(profile):
    return (
        alt.Chart(profile.missing_by_column)
        .mark_bar(cornerRadiusEnd=8, color="#F59E0B")
        .encode(
            y=alt.Y("column:N", sort="-x", title=None),
            x=alt.X("missing_count:Q", title="Missing cells"),
            tooltip=[
                alt.Tooltip("column:N", title="Column"),
                alt.Tooltip("missing_count:Q", title="Missing cells"),
            ],
        )
        .properties(height=300, title="Missing values by column")
        .configure_view(stroke=None)
        .configure_axis(
            labelColor="#CBD5E1",
            titleColor="#E2E8F0",
            gridColor="rgba(148, 163, 184, 0.12)",
            domainColor="rgba(148, 163, 184, 0.12)",
            tickColor="rgba(148, 163, 184, 0.12)",
        )
        .configure_title(color="#F8FAFC", fontSize=18, anchor="start")
    )


def render_chart_collection(charts, empty_message):
    if not charts:
        st.info(empty_message)
        return

    chart_columns = st.columns(min(2, len(charts)), gap="large")
    for index, chart_item in enumerate(charts):
        with chart_columns[index % len(chart_columns)]:
            with st.container(border=True):
                chart = build_chart(chart_item) if isinstance(chart_item, ChartSpec) else chart_item
                st.altair_chart(chart, use_container_width=True)


def render_dataset_dashboard(profile):
    overview_charts = list(getattr(profile, "overview_charts", []))
    numeric_charts = list(getattr(profile, "numeric_charts", []))
    categorical_charts = list(getattr(profile, "categorical_charts", []))
    datetime_charts = list(getattr(profile, "datetime_charts", []))

    with st.container(border=True):
        render_section_heading(
            "Dataset Overview",
            f"{profile.row_count:,} rows and {profile.column_count:,} columns are ready for exploration.",
            kicker="Upload Dashboard",
        )
        st.markdown(
            f'<p class="hero-caption"><strong>{escape(profile.dataset_name)}</strong> is now active. Aidssist profiled the file and prepared a first-pass view of structure, completeness, and content patterns.</p>',
            unsafe_allow_html=True,
        )

    metric_columns = st.columns(7)
    metric_columns[0].metric("Rows", f"{profile.row_count:,}")
    metric_columns[1].metric("Columns", f"{profile.column_count:,}")
    metric_columns[2].metric("Missing cells", f"{profile.missing_cell_count:,}")
    metric_columns[3].metric("Duplicate rows", f"{profile.duplicate_row_count:,}")
    metric_columns[4].metric("Numeric", f"{profile.numeric_column_count:,}")
    metric_columns[5].metric("Categorical", f"{profile.categorical_column_count:,}")
    metric_columns[6].metric("Datetime", f"{profile.datetime_column_count:,}")

    overview_tab, numeric_tab, categorical_tab, datetime_tab = st.tabs(
        ["Overview", "Numeric", "Categorical", "Time"]
    )

    with overview_tab:
        chart_items = [build_column_type_chart(profile)]
        if not profile.missing_by_column.empty:
            chart_items.append(build_missing_values_chart(profile))
        chart_items.extend(overview_charts)
        render_chart_collection(
            chart_items,
            "No overview charts are available for this dataset yet.",
        )

    with numeric_tab:
        render_chart_collection(
            numeric_charts,
            "No numeric columns were detected for numeric charting.",
        )

    with categorical_tab:
        render_chart_collection(
            categorical_charts,
            "No categorical columns were detected for category charting.",
        )

    with datetime_tab:
        render_chart_collection(
            datetime_charts,
            "No datetime columns were detected for time-series charting.",
        )
        if datetime_charts:
            with st.container(border=True):
                render_section_heading(
                    "Recent Activity Snapshot",
                    "Latest points from the primary time-series chart.",
                )
                st.dataframe(
                    datetime_charts[0].data.tail(7),
                    use_container_width=True,
                    hide_index=True,
                )


def filter_explore_dataframe(df, visible_columns, search_text):
    filtered_df = df.loc[:, visible_columns].copy()
    if not search_text.strip():
        return filtered_df

    search_frame = filtered_df.astype("string").fillna("")
    matches = search_frame.apply(
        lambda column: column.str.contains(search_text, case=False, regex=False)
    )
    return filtered_df.loc[matches.any(axis=1)]


def render_data_explorer(df, dataset_key):
    with st.container(border=True):
        render_section_heading(
            "Explore the Full CSV",
            "Search the uploaded file, choose visible columns, and inspect the full dataset with live charts already prepared above.",
        )

        visible_columns = st.multiselect(
            "Visible columns",
            options=list(df.columns),
            default=list(df.columns),
            key=f"visible_columns_{dataset_key}",
        )
        search_text = st.text_input(
            "Search rows across visible columns",
            key=f"search_rows_{dataset_key}",
            placeholder="Type text to filter matching rows...",
        )

        if not visible_columns:
            st.warning("Select at least one column to inspect the uploaded CSV.")
            return

        filtered_df = filter_explore_dataframe(df, visible_columns, search_text)
        summary_columns = st.columns(3)
        summary_columns[0].metric("Visible rows", f"{len(filtered_df):,}")
        summary_columns[1].metric("Visible columns", f"{len(visible_columns):,}")
        summary_columns[2].metric("Search active", "Yes" if search_text.strip() else "No")

        st.dataframe(filtered_df, use_container_width=True, height=460)
        st.download_button(
            "Download filtered CSV",
            data=dataframe_to_csv_bytes(filtered_df),
            file_name="filtered_dataset.csv",
            mime="text/csv",
            use_container_width=True,
        )


def render_column_explorer(df, profile, dataset_key):
    with st.container(border=True):
        render_section_heading(
            "Column Explorer",
            "Drill into any column to inspect its type, missingness, unique values, and best-fit visualization.",
        )

        selected_column = st.selectbox(
            "Choose a column",
            options=profile.data_dictionary["column"].tolist(),
            key=f"column_explorer_{dataset_key}",
        )
        column_insight = build_column_insight(df, selected_column)

        metric_columns = st.columns(4)
        metric_columns[0].metric("Semantic type", column_insight.semantic_type)
        metric_columns[1].metric("Non-null values", f"{column_insight.non_null_count:,}")
        metric_columns[2].metric("Missing", f"{column_insight.missing_count:,}")
        metric_columns[3].metric("Unique", f"{column_insight.unique_count:,}")

        detail_columns = st.columns([1.2, 1], gap="large")
        with detail_columns[0]:
            with st.container(border=True):
                render_section_heading(
                    "Best-fit Column Chart",
                    "Aidssist chooses the strongest default chart for the selected column.",
                )
                if column_insight.chart is not None:
                    st.altair_chart(build_chart(column_insight.chart), use_container_width=True)
                else:
                    st.info("This column does not have enough signal for a safe automatic chart.")

        with detail_columns[1]:
            with st.container(border=True):
                render_section_heading(
                    "Column Details",
                    "Profile summary and example values for the selected column.",
                )
                column_row = profile.data_dictionary.loc[
                    profile.data_dictionary["column"] == selected_column
                ]
                st.dataframe(column_row, use_container_width=True, hide_index=True)
                st.markdown("**Sample values**")
                if column_insight.sample_values:
                    st.write(", ".join(column_insight.sample_values))
                else:
                    st.write("No non-null sample values available.")


def render_benchmark_queries(test_cases, dataset_key):
    with st.container(border=True):
        render_section_heading(
            "Benchmark Queries",
            "Start from a benchmark prompt to explore the uploaded dataset, then refine it in the custom query box below.",
        )
        benchmark_columns = st.columns(len(test_cases))

        for index, (column, test_case) in enumerate(zip(benchmark_columns, test_cases)):
            with column:
                with st.container(border=True):
                    st.markdown(
                        f'<span class="benchmark-type">{escape(test_case["type"])}</span>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<p class="benchmark-query">{escape(test_case["query"])}</p>',
                        unsafe_allow_html=True,
                    )

                    if st.button(
                        "Use this query",
                        key=f"benchmark_{dataset_key}_{index}",
                        use_container_width=True,
                    ):
                        st.session_state.query_input = test_case["query"]


def render_query_form(api_ready, api_message, analysis_allowed):
    with st.container(border=True):
        render_section_heading(
            "Ask your data",
            "Type a business question or start from a benchmark query. Aidssist will validate the active dataset, execute the analysis pipeline, and return the result through the production runtime.",
        )

        if not api_ready:
            st.warning("Aidssist runtime setup is required before analysis can run.")
            st.caption(api_message)
        else:
            st.caption(api_message)
            if not analysis_allowed:
                st.info("Resolve blocking validation findings or acknowledge remaining warnings to enable analysis.")

        with st.form("analysis_form"):
            query = st.text_input(
                "Analysis question",
                key="query_input",
                placeholder="Example: Which segments are driving the biggest revenue swings over time?",
            )
            submitted = st.form_submit_button(
                "Run analysis",
                disabled=not api_ready or not analysis_allowed,
                use_container_width=True,
            )

        return submitted, query


def render_analysis_output(analysis, show_customization=False):
    result_profile = None if analysis["error"] else profile_analysis_result(analysis["result"])
    custom_chart_spec = None
    chart_takeaway = None
    contract = analysis.get("analysis_contract") or {}
    system_decision = contract.get("system_decision") or analysis.get("system_decision") or {}
    contract_intent = contract.get("intent")
    contract_confidence = contract.get("confidence") or analysis.get("confidence")
    contract_warnings = contract.get("warnings") or analysis.get("warnings") or []
    contract_recommendations = contract.get("recommendations") or analysis.get("recommendations") or []

    if isinstance(contract_warnings, str):
        contract_warnings = [contract_warnings]
    if isinstance(contract_recommendations, str):
        contract_recommendations = [contract_recommendations]

    st.markdown("")
    with st.container(border=True):
        render_section_heading(
            "Analysis Results",
            "The final answer leads with the insight and recommended action, while technical execution details stay tucked away below.",
            kicker="Executive View",
        )

    if analysis["error"]:
        with st.container(border=True):
            st.error(analysis["error"])
    else:
        if show_customization:
            result_profile, custom_chart_spec = render_chart_customization_panel(analysis)
        display_chart_spec = custom_chart_spec or (result_profile.chart if result_profile else None)
        chart_takeaway = build_chart_takeaway(display_chart_spec)
        with st.container(border=True):
            render_section_heading(
                "Key insight",
                "Start with the summary and recommended next move before diving into the supporting chart and raw output.",
            )
            metadata_line = []
            if contract_intent:
                metadata_line.append(f"Intent: `{contract_intent}`")
            if contract_confidence:
                metadata_line.append(f"Confidence: `{contract_confidence}`")
            if metadata_line:
                st.caption(" • ".join(metadata_line))
            if system_decision.get("selected_mode"):
                st.info(
                    f"System decision: {str(system_decision.get('selected_mode')).upper()} | "
                    f"{system_decision.get('reason') or 'No routing reason was provided.'}"
                )
                if system_decision.get("suggestion"):
                    st.caption(system_decision["suggestion"])
            if analysis.get("summary"):
                st.markdown("**Summary**")
                st.write(analysis["summary"])
            if analysis.get("insights"):
                st.markdown("**Ranked Decision Insights**")
                st.write(analysis["insights"])
            if analysis.get("business_decisions"):
                st.markdown("**Recommended Action**")
                st.write(analysis["business_decisions"])
            elif chart_takeaway:
                st.markdown("**Auto Takeaway**")
                st.write(chart_takeaway)
            if contract_recommendations:
                st.markdown("**Recommendations**")
                for recommendation in contract_recommendations:
                    st.write(f"- {recommendation}")
            if contract_warnings:
                st.markdown("**Warnings**")
                for warning_message in contract_warnings:
                    st.warning(warning_message)

        with st.container(border=True):
            render_section_heading(
                "Supporting chart",
                "Aidssist inferred the strongest visual for the returned analysis result.",
            )
            if display_chart_spec is not None:
                st.altair_chart(build_chart(display_chart_spec), use_container_width=True)
                if chart_takeaway:
                    st.markdown(
                        f'<div class="chart-note">{escape(chart_takeaway)}</div>',
                        unsafe_allow_html=True,
                    )
            elif result_profile and result_profile.metric_value:
                st.metric(result_profile.metric_label or "Result", result_profile.metric_value)
            elif result_profile and result_profile.text_value:
                st.write(result_profile.text_value)
            else:
                render_result(analysis["result"])

        with st.container(border=True):
            render_section_heading(
                "Result Details",
                "Inspect the returned table, metric, chart, or raw output generated by the analysis pipeline.",
            )
            render_result(analysis["result"])

    with st.container(border=True):
        render_section_heading(
            "Technical Details",
            "Expand these sections if you want to review the detected intent, validation result, generated code, and execution outcome.",
            kicker="Advanced",
        )

        with st.expander("Pipeline", expanded=False):
            st.markdown("**Intent Detection**")
            st.write(analysis.get("intent") or "No intent was detected.")
            st.markdown("**Query**")
            st.write(analysis["build_query"])
            st.markdown("**Validation**")
            st.write(analysis.get("module_validation") or "No module validation was produced.")
            st.markdown("**Plan**")
            st.write(analysis["build_plan"])
            st.markdown("**Generated Code**")
            if analysis["generated_code"]:
                st.code(analysis["generated_code"], language="python")
                st.download_button(
                    "Download generated analysis code",
                    data=analysis["generated_code"].encode("utf-8"),
                    file_name="analysis_pipeline.py",
                    mime="text/x-python",
                    use_container_width=True,
                )
            else:
                st.write("No code was generated.")

        with st.expander("Execution", expanded=False):
            st.text(f"Status: {analysis['test_status']}")
            if analysis.get("fix_status"):
                if analysis.get("fix_applied"):
                    st.info(analysis["fix_status"])
                else:
                    st.caption(analysis["fix_status"])
            if analysis["test_error"]:
                st.error(analysis["test_error"])
            else:
                st.success("Execution passed.")
                render_result(analysis["result"])

        if analysis.get("fix_applied") and analysis.get("fixed_code"):
            with st.expander("Automatic Fix", expanded=False):
                st.write(analysis.get("fix_status") or "Aidssist repaired the generated code before returning the result.")
                st.code(analysis["fixed_code"], language="python")


def render_export_center(df, profile, analysis):
    dataset_stem = Path(profile.dataset_name).stem

    with st.container(border=True):
        render_section_heading(
            "Export Center",
            "Download the raw dataset, filtered/visualized chart data, and the latest analysis result artifact.",
            kicker="Export",
        )

        base_download_columns = st.columns(2)
        with base_download_columns[0]:
            st.download_button(
                "Download raw dataset CSV",
                data=dataframe_to_csv_bytes(df),
                file_name=profile.dataset_name,
                mime="text/csv",
                use_container_width=True,
            )
        with base_download_columns[1]:
            st.download_button(
                "Download data dictionary CSV",
                data=dataframe_to_csv_bytes(profile.data_dictionary),
                file_name=f"{dataset_stem}_data_dictionary.csv",
                mime="text/csv",
                use_container_width=True,
            )

        overview_chart_payloads = build_chart_download_payloads(
            profile.content_chart,
            f"{dataset_stem}_overview",
            "Download overview chart data CSV",
        )
        if overview_chart_payloads:
            overview_columns = st.columns(len(overview_chart_payloads))
            for column, (label, data, file_name, mime) in zip(overview_columns, overview_chart_payloads):
                with column:
                    st.download_button(
                        label,
                        data=data,
                        file_name=file_name,
                        mime=mime,
                        use_container_width=True,
                    )

        if not analysis or analysis.get("error"):
            st.info("Run a successful analysis to unlock updated result downloads and visualized result data.")
            return

        result_profile = profile_analysis_result(analysis["result"])
        result_file_name, result_payload, result_mime, result_label = build_result_download_payload(
            analysis["result"]
        )
        result_downloads = [
            (
                result_label,
                result_payload,
                result_file_name,
                result_mime,
            )
        ]
        result_downloads.extend(
            build_chart_download_payloads(
                result_profile.chart,
                f"{dataset_stem}_analysis",
                "Download visualized result data CSV",
            )
        )

        download_columns = st.columns(len(result_downloads))
        for column, (label, data, file_name, mime) in zip(download_columns, result_downloads):
            with column:
                st.download_button(
                    label,
                    data=data,
                    file_name=file_name,
                    mime=mime,
                    use_container_width=True,
                )


def build_forecast_download_payloads(forecast_output, dataset_stem):
    payloads = []
    forecast_table = forecast_output.get("forecast_table")
    comparison_table = forecast_output.get("comparison_table")
    driver_importance_table = forecast_output.get("driver_importance_table")

    if isinstance(forecast_table, pd.DataFrame) and not forecast_table.empty:
        payloads.append(
            (
                "Download forecast CSV",
                dataframe_to_csv_bytes(forecast_table),
                f"{dataset_stem}_forecast.csv",
                "text/csv",
            )
        )
        payloads.append(
            (
                "Download forecast JSON",
                forecast_table.to_json(orient="records", date_format="iso", indent=2).encode("utf-8"),
                f"{dataset_stem}_forecast.json",
                "application/json",
            )
        )

    evaluation_payload = {
        "chosen_model": forecast_output.get("chosen_model"),
        "trend_status": forecast_output.get("trend_status"),
        "resolved_frequency": forecast_output.get("resolved_frequency"),
        "history_points": forecast_output.get("history_points"),
        "history_start": forecast_output.get("history_start"),
        "history_end": forecast_output.get("history_end"),
        "horizon": forecast_output.get("horizon"),
        "uncertainty_ratio": forecast_output.get("uncertainty_ratio"),
        "evaluation_metrics": forecast_output.get("evaluation_metrics"),
        "baseline_metrics": forecast_output.get("baseline_metrics"),
        "artifact_metadata": forecast_output.get("artifact_metadata"),
    }
    payloads.append(
        (
            "Download evaluation report",
            json.dumps(evaluation_payload, indent=2, default=str).encode("utf-8"),
            f"{dataset_stem}_forecast_evaluation.json",
            "application/json",
        )
    )

    payloads.append(
        (
            "Download recommendation summary",
            json.dumps(forecast_output.get("recommendations", []), indent=2, default=str).encode("utf-8"),
            f"{dataset_stem}_forecast_recommendations.json",
            "application/json",
        )
    )

    if isinstance(comparison_table, pd.DataFrame) and not comparison_table.empty:
        payloads.append(
            (
                "Download model comparison CSV",
                dataframe_to_csv_bytes(comparison_table),
                f"{dataset_stem}_forecast_model_comparison.csv",
                "text/csv",
            )
        )

    if isinstance(driver_importance_table, pd.DataFrame) and not driver_importance_table.empty:
        payloads.append(
            (
                "Download driver importance CSV",
                dataframe_to_csv_bytes(driver_importance_table),
                f"{dataset_stem}_forecast_drivers.csv",
                "text/csv",
            )
        )

    payloads.append(
        (
            "Download forecast narrative",
            str(forecast_output.get("summary") or "No forecast summary available.").encode("utf-8"),
            f"{dataset_stem}_forecast_summary.txt",
            "text/plain",
        )
    )

    return payloads


def initialize_app_state():
    for key, value in DEFAULT_EDITOR_STATE.items():
        st.session_state.setdefault(key, value)

    st.session_state.setdefault("analysis_output", None)
    st.session_state.setdefault("forecast_output", None)
    st.session_state.setdefault("loaded_source_state", None)
    st.session_state.setdefault("cleaning_preview_state", None)
    st.session_state.setdefault("active_dataset", None)
    st.session_state.setdefault("active_workflow_id", None)
    st.session_state.setdefault("active_workflow_version", None)
    st.session_state.setdefault("active_workflow_name", None)
    st.session_state.setdefault("saved_source_snapshot_path", None)
    st.session_state.setdefault("saved_source_file_name", None)
    st.session_state.setdefault("advanced_mode", False)
    st.session_state.setdefault("workflow_step", "Upload")
    st.session_state.setdefault("workflow_step_selector", "Upload")
    st.session_state.setdefault("workflow_step_selector_needs_sync", False)
    st.session_state.setdefault("cleaning_applied_source_fingerprint", None)
    st.session_state.setdefault("cleaning_applied_options_signature", None)
    st.session_state.setdefault("validation_acknowledged", False)
    st.session_state.setdefault("forecast_skipped", False)
    st.session_state.setdefault("forecast_defaults_source_fingerprint", None)
    st.session_state.setdefault("last_auto_loaded_source_signature", None)
    st.session_state.setdefault("selected_workflow_id", "")
    st.session_state.setdefault("editor_history", [])
    st.session_state.setdefault("editor_history_index", -1)
    st.session_state.setdefault("editor_history_suspended", False)
    st.session_state.setdefault("guided_flow_notice", None)
    st.session_state.setdefault("dataset_intelligence_cache", {})
    if st.session_state.editor_history_index < 0:
        record_editor_state()


def get_editor_snapshot():
    return {key: st.session_state.get(key) for key in EDITOR_STATE_KEYS}


def record_editor_state():
    if st.session_state.get("editor_history_suspended"):
        return

    snapshot = get_editor_snapshot()
    history = list(st.session_state.get("editor_history", []))
    history_index = int(st.session_state.get("editor_history_index", -1))

    if history_index >= 0 and history and history[history_index] == snapshot:
        return

    next_history = history[: history_index + 1]
    next_history.append(snapshot)
    st.session_state.editor_history = next_history[-50:]
    st.session_state.editor_history_index = len(st.session_state.editor_history) - 1


def apply_editor_snapshot(snapshot):
    st.session_state.editor_history_suspended = True
    try:
        for key, value in snapshot.items():
            st.session_state[key] = value
    finally:
        st.session_state.editor_history_suspended = False


def undo_editor_state():
    history_index = int(st.session_state.get("editor_history_index", -1))
    if history_index <= 0:
        return
    next_index = history_index - 1
    apply_editor_snapshot(st.session_state.editor_history[next_index])
    st.session_state.editor_history_index = next_index


def redo_editor_state():
    history = st.session_state.get("editor_history", [])
    history_index = int(st.session_state.get("editor_history_index", -1))
    if history_index >= len(history) - 1:
        return
    next_index = history_index + 1
    apply_editor_snapshot(history[next_index])
    st.session_state.editor_history_index = next_index


def _safe_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _safe_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def build_uploaded_file_signature(uploaded_source_file, source_type):
    if uploaded_source_file is None:
        return None
    return f"{source_type}:{uploaded_source_file.name}:{uploaded_source_file.size}"


def build_cleaning_options_from_state():
    return CleaningOptions(
        parse_dates=bool(st.session_state.parse_dates),
        coerce_numeric_text=bool(st.session_state.coerce_numeric_text),
        trim_strings=bool(st.session_state.trim_strings),
        drop_duplicates=bool(st.session_state.drop_duplicates_enabled),
        fill_numeric_nulls=str(st.session_state.fill_numeric_nulls),
        fill_text_nulls=str(st.session_state.fill_text_nulls),
        drop_null_rows_over=_safe_float(st.session_state.drop_null_rows_over, 1.0),
        drop_null_columns_over=_safe_float(st.session_state.drop_null_columns_over, 1.0),
    )


def get_current_cleaning_options_signature():
    return build_cleaning_options_signature(build_cleaning_options_from_state())


def build_forecast_config_from_state():
    return ForecastConfig(
        date_column=str(st.session_state.forecast_date_column or ""),
        target_column=str(st.session_state.forecast_target_column or ""),
        driver_columns=[
            str(column)
            for column in (st.session_state.forecast_driver_columns or [])
            if str(column or "").strip()
        ],
        aggregation_frequency=str(st.session_state.forecast_frequency or "auto"),
        horizon=str(st.session_state.forecast_horizon or "next_month"),
        model_strategy=str(st.session_state.forecast_model_strategy or "hybrid"),
        training_mode=str(st.session_state.forecast_training_mode or "auto"),
    )


def build_forecast_config_dict_from_state():
    return forecast_config_to_dict(build_forecast_config_from_state())


def get_current_forecast_config_signature():
    return build_forecast_config_signature(build_forecast_config_from_state())


def build_chart_preferences_from_state():
    return {
        "kind": str(st.session_state.chart_kind_pref),
        "x_column": str(st.session_state.chart_x_pref or ""),
        "y_column": str(st.session_state.chart_y_pref or ""),
        "aggregation": str(st.session_state.chart_aggregation_pref),
        "palette": str(st.session_state.chart_palette_pref),
        "title": str(st.session_state.chart_title_pref or ""),
        "x_title": str(st.session_state.chart_x_title_pref or ""),
        "y_title": str(st.session_state.chart_y_title_pref or ""),
    }


def build_chart_customization_from_state():
    preferences = build_chart_preferences_from_state()
    return ChartCustomization(
        kind=preferences["kind"],
        x_column=preferences["x_column"] or None,
        y_column=preferences["y_column"] or None,
        aggregation=preferences["aggregation"],
        palette=preferences["palette"],
        title=preferences["title"],
        x_title=preferences["x_title"],
        y_title=preferences["y_title"],
    )


def build_export_settings_from_state():
    return {
        "include_csv": bool(st.session_state.export_csv_enabled),
        "include_json": bool(st.session_state.export_json_enabled),
    }


def ensure_forecast_defaults(dataset_state):
    if dataset_state is None:
        return

    source_fingerprint = dataset_state.get("source_fingerprint")
    if not source_fingerprint:
        return

    if st.session_state.get("forecast_defaults_source_fingerprint") == source_fingerprint:
        return

    suggestions = get_forecast_mapping_options(dataset_state["df"]).get("suggestions", {})
    st.session_state.editor_history_suspended = True
    try:
        if not str(st.session_state.forecast_date_column or "").strip():
            st.session_state.forecast_date_column = suggestions.get("date_column", "")
        if not str(st.session_state.forecast_target_column or "").strip():
            st.session_state.forecast_target_column = suggestions.get("target_column", "")
        if not list(st.session_state.forecast_driver_columns or []):
            st.session_state.forecast_driver_columns = list(suggestions.get("driver_columns", []))
        if str(st.session_state.forecast_frequency or "auto") == "auto":
            st.session_state.forecast_frequency = suggestions.get("aggregation_frequency", "auto")
    finally:
        st.session_state.editor_history_suspended = False

    st.session_state.forecast_defaults_source_fingerprint = source_fingerprint
    record_editor_state()


def get_forecast_validation_for_active_dataset():
    active_dataset = st.session_state.get("active_dataset")
    if active_dataset is None:
        return None
    return validate_forecast_config(
        active_dataset["df"],
        build_forecast_config_from_state(),
    )


def _build_csv_source_config(uploaded_source_file):
    file_name = st.session_state.get("saved_source_file_name") or "dataset.csv"
    file_bytes = None
    if uploaded_source_file is not None:
        file_name = uploaded_source_file.name
        file_bytes = uploaded_source_file.getvalue()

    snapshot_path = st.session_state.get("saved_source_snapshot_path")
    if file_bytes is None and not snapshot_path:
        raise DataSourceError("Upload a CSV file or load a saved workflow snapshot first.")

    return CSVSourceConfig(
        file_name=file_name,
        file_bytes=file_bytes,
        snapshot_path=snapshot_path,
        delimiter=str(st.session_state.csv_delimiter or ","),
        encoding=str(st.session_state.csv_encoding or "utf-8"),
    )


def _build_excel_source_config(uploaded_source_file):
    file_name = st.session_state.get("saved_source_file_name") or "dataset.xlsx"
    file_bytes = None
    if uploaded_source_file is not None:
        file_name = uploaded_source_file.name
        file_bytes = uploaded_source_file.getvalue()

    snapshot_path = st.session_state.get("saved_source_snapshot_path")
    if file_bytes is None and not snapshot_path:
        raise DataSourceError("Upload an Excel file or load a saved workflow snapshot first.")

    return ExcelSourceConfig(
        file_name=file_name,
        file_bytes=file_bytes,
        snapshot_path=snapshot_path,
        sheet_name=str(st.session_state.excel_sheet_name or "0"),
    )


def _build_sql_source_config(source_kind):
    source_config = SQLSourceConfig(
        kind=source_kind,
        host=str(st.session_state.sql_host or ""),
        port=_safe_int(st.session_state.sql_port, 5432 if source_kind == "postgres" else 3306),
        database=str(st.session_state.sql_database or ""),
        username=str(st.session_state.sql_username or ""),
        password=str(st.session_state.sql_password or ""),
        table_name=str(st.session_state.sql_table_name or "").strip() or None,
        query=str(st.session_state.sql_query or "").strip() or None,
        limit=_safe_int(st.session_state.sql_limit, 1000),
    )
    issues = validate_sql_source_config(source_config)
    if issues:
        raise DataSourceError(" ".join(issues))
    return source_config


def build_source_config_from_state(uploaded_source_file):
    source_type = str(st.session_state.source_type)
    if source_type == "csv":
        return _build_csv_source_config(uploaded_source_file)
    if source_type == "excel":
        return _build_excel_source_config(uploaded_source_file)
    return _build_sql_source_config(source_type)


def persist_source_config_for_workflow(source_config):
    if isinstance(source_config, (CSVSourceConfig, ExcelSourceConfig)):
        source_config = persist_file_source_snapshot(source_config, SOURCE_SNAPSHOT_DIR)

    serialized_config = serialize_source_config(source_config)
    if "file_name" in serialized_config:
        st.session_state.saved_source_file_name = serialized_config.get("file_name")
        st.session_state.saved_source_snapshot_path = serialized_config.get("snapshot_path")
    return serialized_config


def load_dataset_from_source_config(source_config):
    loaded_data = load_dataframe_from_source(source_config)
    raw_findings = validate_dataframe(loaded_data.dataframe)
    cleaning_result = apply_cleaning_plan(loaded_data.dataframe, build_cleaning_options_from_state())
    cleaned_findings = validate_dataframe(cleaning_result.dataframe)

    cleaned_profile = get_cached_dataset_profile(
        loaded_data.dataset_name,
        loaded_data.dataset_key,
        DATASET_PROFILE_SCHEMA_VERSION,
        cleaning_result.dataframe,
    )
    cleaned_profile = ensure_dataset_profile(
        cleaned_profile,
        cleaning_result.dataframe,
        loaded_data.dataset_name,
        loaded_data.dataset_key,
    )

    return {
        "source_config": serialize_source_config(source_config),
        "source_label": loaded_data.source_label,
        "dataset_name": loaded_data.dataset_name,
        "dataset_key": loaded_data.dataset_key,
        "source_fingerprint": loaded_data.source_fingerprint,
        "raw_df": loaded_data.dataframe,
        "df": cleaning_result.dataframe,
        "raw_findings": raw_findings,
        "validation_findings": cleaned_findings,
        "cleaning_actions": cleaning_result.actions,
        "cleaning_report": dict(cleaning_result.report or {}),
        "profile": cleaned_profile,
    }


def set_active_dataset(dataset_state, *, workflow_id=None, workflow_version=None, workflow_name=None):
    dataset_state = dict(dataset_state)
    dataset_state["workflow_id"] = workflow_id
    dataset_state["workflow_version"] = workflow_version
    dataset_state["workflow_name"] = workflow_name
    st.session_state.active_dataset = dataset_state
    st.session_state.forecast_output = None
    st.session_state.forecast_skipped = False
    st.session_state.active_workflow_id = workflow_id
    st.session_state.active_workflow_version = workflow_version
    st.session_state.active_workflow_name = workflow_name
    if st.session_state.analysis_output and (
        st.session_state.analysis_output.get("dataset_key") != dataset_state["dataset_key"]
    ):
        st.session_state.analysis_output = None


def save_current_workflow(uploaded_source_file):
    source_config = build_source_config_from_state(uploaded_source_file)
    serialized_source_config = persist_source_config_for_workflow(source_config)
    workflow_definition = WORKFLOW_STORE.save_workflow(
        name=st.session_state.workflow_name or "Untitled workflow",
        source_config=serialized_source_config,
        cleaning_options=asdict(build_cleaning_options_from_state()),
        analysis_query=str(st.session_state.query_input or ""),
        forecast_config=build_forecast_config_dict_from_state(),
        chart_preferences=build_chart_preferences_from_state(),
        export_settings=build_export_settings_from_state(),
        workflow_id=st.session_state.active_workflow_id,
    )
    load_workflow_into_editor(workflow_definition)
    return workflow_definition


def workflow_label(workflow_definition):
    return f"{workflow_definition.name} (v{workflow_definition.version})"


def load_workflow_into_editor(workflow_definition):
    source_config = workflow_definition.source_config
    st.session_state.editor_history_suspended = True
    try:
        st.session_state.workflow_name = workflow_definition.name
        st.session_state.query_input = workflow_definition.analysis_query
        st.session_state.active_workflow_id = workflow_definition.workflow_id
        st.session_state.active_workflow_version = workflow_definition.version
        st.session_state.active_workflow_name = workflow_definition.name
        st.session_state.source_type = source_config.get("kind", "csv")
        st.session_state.saved_source_snapshot_path = source_config.get("snapshot_path")
        st.session_state.saved_source_file_name = source_config.get("file_name")

        st.session_state.csv_delimiter = source_config.get("delimiter", ",")
        st.session_state.csv_encoding = source_config.get("encoding", "utf-8")
        st.session_state.excel_sheet_name = str(source_config.get("sheet_name", "0"))
        st.session_state.sql_host = source_config.get("host", "localhost")
        st.session_state.sql_port = source_config.get(
            "port",
            5432 if source_config.get("kind") == "postgres" else 3306,
        )
        st.session_state.sql_database = source_config.get("database", "")
        st.session_state.sql_username = source_config.get("username", "")
        st.session_state.sql_password = source_config.get("password", "")
        st.session_state.sql_table_name = source_config.get("table_name", "") or ""
        st.session_state.sql_query = source_config.get("query", "") or ""
        st.session_state.sql_limit = source_config.get("limit", 1000)

        cleaning_options = workflow_definition.cleaning_options
        st.session_state.parse_dates = cleaning_options.get("parse_dates", True)
        st.session_state.coerce_numeric_text = cleaning_options.get("coerce_numeric_text", True)
        st.session_state.trim_strings = cleaning_options.get("trim_strings", True)
        st.session_state.drop_duplicates_enabled = cleaning_options.get("drop_duplicates", True)
        st.session_state.fill_numeric_nulls = cleaning_options.get("fill_numeric_nulls", "none")
        st.session_state.fill_text_nulls = cleaning_options.get("fill_text_nulls", "none")
        st.session_state.drop_null_rows_over = cleaning_options.get("drop_null_rows_over", 1.0)
        st.session_state.drop_null_columns_over = cleaning_options.get("drop_null_columns_over", 1.0)

        forecast_config = workflow_definition.forecast_config
        st.session_state.forecast_date_column = forecast_config.get("date_column", "")
        st.session_state.forecast_target_column = forecast_config.get("target_column", "")
        st.session_state.forecast_driver_columns = list(forecast_config.get("driver_columns", []))
        st.session_state.forecast_frequency = forecast_config.get("aggregation_frequency", "auto")
        st.session_state.forecast_horizon = forecast_config.get("horizon", "next_month")
        st.session_state.forecast_model_strategy = forecast_config.get("model_strategy", "hybrid")
        st.session_state.forecast_training_mode = forecast_config.get("training_mode", "auto")

        chart_preferences = workflow_definition.chart_preferences
        st.session_state.chart_kind_pref = chart_preferences.get("kind", "auto")
        st.session_state.chart_x_pref = chart_preferences.get("x_column", "")
        st.session_state.chart_y_pref = chart_preferences.get("y_column", "")
        st.session_state.chart_aggregation_pref = chart_preferences.get("aggregation", "sum")
        st.session_state.chart_palette_pref = chart_preferences.get("palette", "orange")
        st.session_state.chart_title_pref = chart_preferences.get("title", "")
        st.session_state.chart_x_title_pref = chart_preferences.get("x_title", "")
        st.session_state.chart_y_title_pref = chart_preferences.get("y_title", "")

        export_settings = workflow_definition.export_settings
        st.session_state.export_csv_enabled = export_settings.get("include_csv", True)
        st.session_state.export_json_enabled = export_settings.get("include_json", True)
    finally:
        st.session_state.editor_history_suspended = False

    record_editor_state()


def run_saved_workflow(workflow_definition):
    source_config = deserialize_source_config(workflow_definition.source_config)
    loaded_source_state = build_loaded_source_state(source_config)
    set_loaded_source_state(
        loaded_source_state,
        workflow_id=workflow_definition.workflow_id,
        workflow_version=workflow_definition.version,
        workflow_name=workflow_definition.name,
    )
    promote_cleaning_preview_to_active_dataset()
    forecast_config = forecast_config_from_dict(workflow_definition.forecast_config)
    if forecast_config.date_column and forecast_config.target_column:
        run_forecast_for_active_dataset(forecast_config)
    if workflow_definition.analysis_query.strip():
        run_analysis_for_active_dataset(workflow_definition.analysis_query.strip())


def _should_block_analysis(dataset_state):
    findings = dataset_state.get("validation_findings", [])
    if has_blocking_findings(findings):
        return True
    warning_count, error_count = summarize_findings(findings)
    return error_count > 0 or (warning_count > 0 and not st.session_state.get("validation_acknowledged", False))


def _build_workflow_context(dataset_state):
    forecast_output = st.session_state.get("forecast_output") or {}
    return {
        "workflow_id": dataset_state.get("workflow_id"),
        "workflow_version": dataset_state.get("workflow_version"),
        "workflow_name": dataset_state.get("workflow_name"),
        "source_label": dataset_state.get("source_label"),
        "source_fingerprint": dataset_state.get("source_fingerprint"),
        "cleaning_actions": list(dataset_state.get("cleaning_actions", [])),
        "cleaning_report": dict(dataset_state.get("cleaning_report") or {}),
        "validation_findings": [finding_to_dict(finding) for finding in dataset_state.get("validation_findings", [])],
        "source_config": dataset_state.get("source_config", {}),
        "forecast_config": build_forecast_config_dict_from_state(),
        "forecast_summary": forecast_output.get("summary"),
        "forecast_recommendations": forecast_output.get("recommendations", []),
        "forecast_model": forecast_output.get("chosen_model"),
        "forecast_artifact_metadata": forecast_output.get("artifact_metadata"),
        "chart_preferences": build_chart_preferences_from_state(),
        "export_settings": build_export_settings_from_state(),
    }


def record_analysis_run(analysis_output, dataset_state):
    generated_code = analysis_output.get("generated_code")
    export_artifacts = []
    if st.session_state.export_csv_enabled:
        export_artifacts.append("csv")
    if st.session_state.export_json_enabled:
        export_artifacts.append("json")
    if generated_code:
        export_artifacts.append("python")

    run_record = WORKFLOW_STORE.build_run_record(
        workflow_id=dataset_state.get("workflow_id"),
        workflow_version=dataset_state.get("workflow_version"),
        workflow_name=dataset_state.get("workflow_name"),
        source_fingerprint=dataset_state.get("source_fingerprint"),
        source_label=dataset_state.get("source_label"),
        validation_findings=[finding_to_dict(finding) for finding in dataset_state.get("validation_findings", [])],
        cleaning_actions=list(dataset_state.get("cleaning_actions", [])),
        generated_code=generated_code,
        final_status=analysis_output.get("test_status") or "UNKNOWN",
        error_message=analysis_output.get("error"),
        export_artifacts=export_artifacts,
        analysis_query=analysis_output.get("query") or "",
        result_summary=analysis_output.get("summary"),
    )
    WORKFLOW_STORE.record_run(run_record)


def record_forecast_run(forecast_output, dataset_state):
    artifact_key, artifact_metadata = persist_forecast_artifact(
        forecast_output,
        workflow_id=dataset_state.get("workflow_id"),
        workflow_version=dataset_state.get("workflow_version"),
        source_fingerprint=dataset_state.get("source_fingerprint"),
        dataset_name=dataset_state.get("dataset_name") or "dataset.csv",
    )
    forecast_record = WORKFLOW_STORE.build_forecast_artifact_record(
        workflow_id=dataset_state.get("workflow_id"),
        workflow_version=dataset_state.get("workflow_version"),
        workflow_name=dataset_state.get("workflow_name"),
        source_fingerprint=dataset_state.get("source_fingerprint"),
        source_label=dataset_state.get("source_label"),
        target_column=artifact_metadata.get("target_column") or forecast_output.get("config", {}).get("target_column"),
        horizon=artifact_metadata.get("horizon") or forecast_output.get("horizon"),
        model_name=artifact_metadata.get("model_name") or forecast_output.get("chosen_model"),
        training_mode=forecast_output.get("config", {}).get("training_mode") or "auto",
        status=artifact_metadata.get("status") or forecast_output.get("status") or "UNKNOWN",
        artifact_key=artifact_key,
        forecast_config=forecast_output.get("config", {}),
        evaluation_metrics=artifact_metadata.get("evaluation_metrics") or forecast_output.get("evaluation_metrics", {}),
        recommendation_payload=artifact_metadata.get("recommendations") or forecast_output.get("recommendations", []),
        summary=artifact_metadata.get("summary") or forecast_output.get("summary"),
    )
    WORKFLOW_STORE.record_forecast_artifact(forecast_record)
    artifact_metadata["artifact_id"] = forecast_record.artifact_id
    forecast_output["artifact_metadata"] = artifact_metadata


def run_analysis_for_active_dataset(query):
    dataset_state = st.session_state.get("active_dataset")
    if dataset_state is None:
        raise RuntimeError("Load a dataset before running analysis.")

    workflow_context = _build_workflow_context(dataset_state)
    if analysis_service_enabled():
        analysis_output = run_remote_analysis(
            dataset_state,
            query,
            workflow_context=workflow_context,
        )
    else:
        analysis_output = run_builder_pipeline(
            query,
            dataset_state["df"],
            workflow_context=workflow_context,
        )
        record_analysis_run(analysis_output, dataset_state)

    analysis_output["dataset_key"] = dataset_state["dataset_key"]
    st.session_state.analysis_output = analysis_output
    return analysis_output


def normalize_cleaning_report(cleaning_report):
    payload = dict(cleaning_report or {})
    before = dict(payload.get("before") or {})
    after = dict(payload.get("after") or {})
    outlier_columns = {
        str(column_name): int(count or 0)
        for column_name, count in dict(payload.get("outlier_columns") or {}).items()
    }

    return {
        "quality_score": float(payload.get("quality_score") or 0.0),
        "missing_handled": int(payload.get("missing_handled") or 0),
        "duplicates_removed": int(payload.get("duplicates_removed") or 0),
        "outliers_detected": int(payload.get("outliers_detected") or 0),
        "issues": [str(item) for item in payload.get("issues", []) if str(item).strip()],
        "actions": [str(item) for item in payload.get("actions", []) if str(item).strip()],
        "columns_dropped": [str(item) for item in payload.get("columns_dropped", []) if str(item).strip()],
        "type_conversions": {
            str(column_name): {
                "from": str(dict(details or {}).get("from") or ""),
                "to": str(dict(details or {}).get("to") or ""),
            }
            for column_name, details in dict(payload.get("type_conversions") or {}).items()
        },
        "outlier_columns": outlier_columns,
        "before": {
            "row_count": int(before.get("row_count") or 0),
            "column_count": int(before.get("column_count") or 0),
            "missing_cells": int(before.get("missing_cells") or 0),
            "duplicate_rows": int(before.get("duplicate_rows") or 0),
        },
        "after": {
            "row_count": int(after.get("row_count") or 0),
            "column_count": int(after.get("column_count") or 0),
            "missing_cells": int(after.get("missing_cells") or 0),
            "duplicate_rows": int(after.get("duplicate_rows") or 0),
        },
    }


def render_cleaning_report_panel(cleaning_report):
    report = normalize_cleaning_report(cleaning_report)
    if not any(
        (
            report["quality_score"],
            report["missing_handled"],
            report["duplicates_removed"],
            report["outliers_detected"],
            report["before"]["row_count"],
            report["after"]["row_count"],
        )
    ):
        return

    with st.container(border=True):
        render_section_heading(
            "Cleaning report",
            "Real preprocessing. ✅Reliable model. ✅ Trust layer strong. ✅",
            kicker="Trust Layer",
        )
        metrics = st.columns(4)
        metrics[0].metric("Data quality score", f"{report['quality_score']:.2f}")
        metrics[1].metric("Missing fixed", f"{report['missing_handled']:,}")
        metrics[2].metric("Duplicates removed", f"{report['duplicates_removed']:,}")
        metrics[3].metric("Outlier alerts", f"{report['outliers_detected']:,}")

        comparison_columns = st.columns(2, gap="large")
        with comparison_columns[0]:
            st.markdown("**Before cleaning**")
            st.write(f"Rows: {report['before']['row_count']:,}")
            st.write(f"Columns: {report['before']['column_count']:,}")
            st.write(f"Missing cells: {report['before']['missing_cells']:,}")
            st.write(f"Duplicate rows: {report['before']['duplicate_rows']:,}")
        with comparison_columns[1]:
            st.markdown("**After cleaning**")
            st.write(f"Rows: {report['after']['row_count']:,}")
            st.write(f"Columns: {report['after']['column_count']:,}")
            st.write(f"Missing cells: {report['after']['missing_cells']:,}")
            st.write(f"Duplicate rows: {report['after']['duplicate_rows']:,}")

        outlier_alerts = [
            f"{column_name}: {count:,} potential outliers"
            for column_name, count in sorted(report["outlier_columns"].items())
            if count > 0
        ]
        if outlier_alerts:
            st.markdown("**Outlier alerts**")
            for alert in outlier_alerts:
                st.warning(alert)
        else:
            st.success("No numeric outlier alerts were detected after preprocessing.")

        if report["columns_dropped"]:
            st.caption("Dropped columns: " + ", ".join(report["columns_dropped"]))

        if report["type_conversions"]:
            with st.expander("Type conversions", expanded=False):
                for column_name, details in report["type_conversions"].items():
                    st.write(f"{column_name}: {details['from']} -> {details['to']}")

        if report["issues"]:
            with st.expander("Remaining cleaning issues", expanded=False):
                for issue in report["issues"]:
                    st.write(f"- {issue}")


def render_validation_summary(dataset_state):
    raw_findings = dataset_state.get("raw_findings", [])
    cleaned_findings = dataset_state.get("validation_findings", [])
    warning_count, error_count = summarize_findings(cleaned_findings)

    with st.container(border=True):
        render_section_heading(
            "Validation and Cleaning",
            "Aidssist validates the loaded dataset before analysis, applies the configured cleaning rules, and records the resulting warnings or blockers.",
            kicker="Quality Gate",
        )
        metrics = st.columns(4)
        metrics[0].metric("Warnings", f"{warning_count}")
        metrics[1].metric("Errors", f"{error_count}")
        metrics[2].metric("Cleaning actions", f"{len(dataset_state.get('cleaning_actions', []))}")
        metrics[3].metric("Source fingerprint", dataset_state.get("source_fingerprint", "N/A"))

        st.caption(dataset_state.get("source_label", ""))
        render_cleaning_report_panel(dataset_state.get("cleaning_report"))

        if dataset_state.get("cleaning_actions"):
            st.markdown("**Applied cleaning actions**")
            for action in dataset_state["cleaning_actions"]:
                st.write(f"- {action}")

        if raw_findings:
            with st.expander("Raw dataset findings", expanded=False):
                for finding in raw_findings:
                    label = "Error" if finding.severity == "error" else "Warning"
                    st.write(f"**{label} - {finding.category}**")
                    st.write(finding.message)
                    if finding.suggested_fix:
                        st.caption(f"Suggested fix: {finding.suggested_fix}")

        if cleaned_findings:
            with st.expander("Active dataset findings", expanded=True):
                for finding in cleaned_findings:
                    label = "Error" if finding.severity == "error" else "Warning"
                    if finding.severity == "error":
                        st.error(f"{label}: {finding.message}")
                    else:
                        st.warning(f"{label}: {finding.message}")
                    if finding.suggested_fix:
                        st.caption(f"Suggested fix: {finding.suggested_fix}")
        else:
            st.success("No validation findings remain after cleaning.")

        if error_count == 0 and warning_count > 0:
            st.checkbox(
                "I understand the remaining warnings and want to allow analysis anyway.",
                key="validation_acknowledged",
            )
        elif error_count > 0:
            st.session_state.validation_acknowledged = False


def render_chart_customization_panel(analysis):
    result_profile = profile_analysis_result(analysis["result"])
    if result_profile.table is None or result_profile.table.empty:
        return result_profile, None

    options = get_chart_customization_options(result_profile.table)
    chart_customization = build_chart_customization_from_state()
    custom_chart_spec = build_custom_result_chart(result_profile.table, chart_customization)

    with st.container(border=True):
        render_section_heading(
            "Chart Customization",
            "Override the automatic result chart with your preferred visual, grouping, and palette settings.",
            kicker="Visualization",
        )
        control_columns = st.columns(3)
        control_columns[0].selectbox(
            "Chart type",
            options=options["kinds"],
            key="chart_kind_pref",
            on_change=record_editor_state,
        )
        control_columns[1].selectbox(
            "X axis",
            options=[""] + options["columns"],
            key="chart_x_pref",
            on_change=record_editor_state,
        )
        control_columns[2].selectbox(
            "Y axis",
            options=[""] + options["numeric_columns"],
            key="chart_y_pref",
            on_change=record_editor_state,
        )

        detail_columns = st.columns(3)
        detail_columns[0].selectbox(
            "Aggregation",
            options=options["aggregations"],
            key="chart_aggregation_pref",
            on_change=record_editor_state,
        )
        detail_columns[1].selectbox(
            "Palette",
            options=options["palettes"],
            key="chart_palette_pref",
            on_change=record_editor_state,
        )
        detail_columns[2].text_input(
            "Chart title",
            key="chart_title_pref",
            on_change=record_editor_state,
            placeholder="Leave blank to auto-generate",
        )

        title_columns = st.columns(2)
        title_columns[0].text_input(
            "X-axis title",
            key="chart_x_title_pref",
            on_change=record_editor_state,
        )
        title_columns[1].text_input(
            "Y-axis title",
            key="chart_y_title_pref",
            on_change=record_editor_state,
        )

        if custom_chart_spec is not None:
            st.altair_chart(build_chart(custom_chart_spec), use_container_width=True)
        elif str(st.session_state.chart_kind_pref) != "auto":
            st.info("The current customization does not produce a compatible chart for this result shape.")

    return result_profile, custom_chart_spec


def render_workflow_run_history(selected_workflow_id=None):
    runs = WORKFLOW_STORE.list_runs(limit=15, workflow_id=selected_workflow_id or None)
    with st.container(border=True):
        render_section_heading(
            "Workflow Run History",
            "Recent runs capture the workflow version, source fingerprint, and final status for reproducibility.",
            kicker="Audit Trail",
        )
        if not runs:
            st.info("No workflow runs have been recorded yet.")
            return

        history_rows = [
            {
                "run_id": run.run_id[:8],
                "workflow": run.workflow_name or "Ad hoc",
                "version": run.workflow_version or "-",
                "status": run.final_status,
                "source_fingerprint": run.source_fingerprint,
                "created_at": run.created_at,
                "analysis_query": run.analysis_query,
            }
            for run in runs
        ]
        st.dataframe(pd.DataFrame(history_rows), use_container_width=True, hide_index=True)


def infer_source_type_from_upload(uploaded_source_file):
    if uploaded_source_file is None:
        return "csv"

    suffix = Path(uploaded_source_file.name).suffix.lower()
    if suffix == ".xlsx":
        return "excel"
    return "csv"


def step_kicker(step_name):
    return f"Step {WORKFLOW_STEPS.index(step_name) + 1} of {len(WORKFLOW_STEPS)}"


def build_loaded_source_state(source_config):
    if isinstance(source_config, (CSVSourceConfig, ExcelSourceConfig)):
        source_config = persist_file_source_snapshot(source_config, SOURCE_SNAPSHOT_DIR)

    loaded_data = load_dataframe_from_source(source_config)
    raw_findings = validate_dataframe(loaded_data.dataframe)

    return {
        "source_config": serialize_source_config(source_config),
        "source_label": loaded_data.source_label,
        "dataset_name": loaded_data.dataset_name,
        "dataset_key": loaded_data.dataset_key,
        "source_fingerprint": loaded_data.source_fingerprint,
        "raw_df": loaded_data.dataframe,
        "raw_findings": raw_findings,
    }


def build_cleaning_preview_state(loaded_source_state):
    if loaded_source_state is None:
        return None

    cleaning_options = build_cleaning_options_from_state()
    options_signature = build_cleaning_options_signature(cleaning_options)
    cleaning_result = apply_cleaning_plan(
        loaded_source_state["raw_df"],
        cleaning_options,
    )
    cleaned_findings = validate_dataframe(cleaning_result.dataframe)
    cleaned_profile = get_cached_dataset_profile(
        loaded_source_state["dataset_name"],
        loaded_source_state["dataset_key"],
        DATASET_PROFILE_SCHEMA_VERSION,
        cleaning_result.dataframe,
    )
    cleaned_profile = ensure_dataset_profile(
        cleaned_profile,
        cleaning_result.dataframe,
        loaded_source_state["dataset_name"],
        loaded_source_state["dataset_key"],
    )

    return {
        "df": cleaning_result.dataframe,
        "validation_findings": cleaned_findings,
        "cleaning_actions": cleaning_result.actions,
        "cleaning_report": dict(cleaning_result.report or {}),
        "profile": cleaned_profile,
        "cleaning_options_signature": options_signature,
        "_source_fingerprint": loaded_source_state.get("source_fingerprint"),
        "_options_signature": options_signature,
    }


def set_loaded_source_state(loaded_source_state, *, workflow_id=None, workflow_version=None, workflow_name=None):
    prepared_state = dict(loaded_source_state)
    prepared_state["workflow_id"] = workflow_id
    prepared_state["workflow_version"] = workflow_version
    prepared_state["workflow_name"] = workflow_name

    st.session_state.loaded_source_state = prepared_state
    st.session_state.cleaning_preview_state = build_cleaning_preview_state(prepared_state)
    st.session_state.active_dataset = None
    st.session_state.forecast_output = None
    st.session_state.analysis_output = None
    st.session_state.validation_acknowledged = False
    st.session_state.forecast_skipped = False
    st.session_state.forecast_defaults_source_fingerprint = None
    st.session_state.cleaning_applied_source_fingerprint = None
    st.session_state.cleaning_applied_options_signature = None
    st.session_state.active_workflow_id = workflow_id
    st.session_state.active_workflow_version = workflow_version
    st.session_state.active_workflow_name = workflow_name

    source_config = prepared_state.get("source_config", {})
    st.session_state.saved_source_snapshot_path = source_config.get("snapshot_path")
    st.session_state.saved_source_file_name = source_config.get("file_name")


def refresh_cleaning_preview_state():
    loaded_source_state = st.session_state.get("loaded_source_state")
    if loaded_source_state is None:
        st.session_state.cleaning_preview_state = None
        return None

    options_signature = get_current_cleaning_options_signature()
    preview_state = st.session_state.get("cleaning_preview_state")
    if should_reuse_cleaning_preview(
        preview_state,
        loaded_source_state.get("source_fingerprint"),
        options_signature,
    ):
        return preview_state

    preview_state = build_cleaning_preview_state(loaded_source_state)
    st.session_state.cleaning_preview_state = preview_state
    return preview_state


def promote_cleaning_preview_to_active_dataset():
    loaded_source_state = st.session_state.get("loaded_source_state")
    if loaded_source_state is None:
        raise RuntimeError("Load a dataset before applying cleaning.")

    preview_state = refresh_cleaning_preview_state()
    if preview_state is None:
        raise RuntimeError("Cleaning preview is unavailable.")

    dataset_state = {
        "source_config": loaded_source_state["source_config"],
        "source_label": loaded_source_state["source_label"],
        "dataset_name": loaded_source_state["dataset_name"],
        "dataset_key": loaded_source_state["dataset_key"],
        "source_fingerprint": loaded_source_state["source_fingerprint"],
        "raw_df": loaded_source_state["raw_df"],
        "df": preview_state["df"],
        "raw_findings": loaded_source_state["raw_findings"],
        "validation_findings": preview_state["validation_findings"],
        "cleaning_actions": preview_state["cleaning_actions"],
        "cleaning_report": preview_state.get("cleaning_report"),
        "profile": preview_state["profile"],
        "cleaning_options_signature": preview_state["cleaning_options_signature"],
    }

    set_active_dataset(
        dataset_state,
        workflow_id=loaded_source_state.get("workflow_id"),
        workflow_version=loaded_source_state.get("workflow_version"),
        workflow_name=loaded_source_state.get("workflow_name"),
    )
    st.session_state.cleaning_applied_source_fingerprint = loaded_source_state["source_fingerprint"]
    st.session_state.cleaning_applied_options_signature = preview_state["cleaning_options_signature"]
    st.session_state.validation_acknowledged = False
    return dataset_state


def has_cleaned_dataset():
    return is_cleaned_dataset_current(
        active_dataset=st.session_state.get("active_dataset"),
        loaded_source_state=st.session_state.get("loaded_source_state"),
        current_options_signature=get_current_cleaning_options_signature(),
        applied_source_fingerprint=st.session_state.get("cleaning_applied_source_fingerprint"),
        applied_options_signature=st.session_state.get("cleaning_applied_options_signature"),
    )


def has_successful_forecast():
    return is_forecast_result_current(
        forecast_output=st.session_state.get("forecast_output"),
        active_dataset=st.session_state.get("active_dataset"),
        current_config_signature=get_current_forecast_config_signature(),
    )


def has_successful_analysis():
    active_dataset = st.session_state.get("active_dataset")
    analysis = st.session_state.get("analysis_output")
    if active_dataset is None or analysis is None:
        return False
    return (
        analysis.get("dataset_key") == active_dataset.get("dataset_key")
        and not analysis.get("error")
    )


def get_dataset_intelligence_snapshot(dataset_state):
    if dataset_state is None:
        return None

    cache = st.session_state.setdefault("dataset_intelligence_cache", {})
    cache_key = str(
        dataset_state.get("source_fingerprint")
        or dataset_state.get("dataset_key")
        or "active-dataset"
    )
    cached_snapshot = cache.get(cache_key)
    if cached_snapshot is not None:
        return cached_snapshot

    dataframe = dataset_state["df"]
    intelligence = detect_dataset_type(dataframe)
    guided_mode = decide_analysis_mode(dataframe, "predict future values")
    snapshot = {
        "intelligence": intelligence,
        "guided_mode": {
            "selected_mode": str(guided_mode.get("mode") or "analysis"),
            "reason": str(guided_mode.get("reason") or ""),
            "confidence": float(guided_mode.get("confidence") or 0.0),
        },
    }
    cache[cache_key] = snapshot
    return snapshot


def dataset_supports_forecast(dataset_state):
    snapshot = get_dataset_intelligence_snapshot(dataset_state)
    if snapshot is None:
        return False
    return (
        snapshot["guided_mode"]["selected_mode"] == "forecast"
        and bool((snapshot.get("intelligence") or {}).get("numeric_columns"))
    )


def run_forecast_for_active_dataset(config: ForecastConfig | dict | None = None):
    dataset_state = st.session_state.get("active_dataset")
    if dataset_state is None:
        raise RuntimeError("Apply cleaning before running a forecast.")
    if not dataset_supports_forecast(dataset_state):
        raise RuntimeError("No valid time column detected. Forecasting requires time-based data.")

    forecast_config = config if isinstance(config, ForecastConfig) else forecast_config_from_dict(config or build_forecast_config_dict_from_state())
    validation = validate_forecast_config(dataset_state["df"], forecast_config)
    if not validation.is_valid:
        raise RuntimeError(" ".join(validation.errors))

    workflow_context = _build_workflow_context(dataset_state)
    forecast_config_payload = forecast_config_to_dict(forecast_config)
    use_remote = analysis_service_enabled() and forecast_config.training_mode != "local"

    if use_remote:
        forecast_output = run_remote_forecast(
            dataset_state,
            forecast_config_payload,
            workflow_context=workflow_context,
        )
    else:
        forecast_output = run_forecast_pipeline(
            dataset_state["df"],
            forecast_config_payload,
            workflow_context=workflow_context,
        )
        record_forecast_run(forecast_output, dataset_state)

    forecast_output["dataset_key"] = dataset_state["dataset_key"]
    forecast_output["source_fingerprint"] = dataset_state["source_fingerprint"]
    forecast_output["forecast_config_signature"] = build_forecast_config_signature(forecast_config)
    st.session_state.forecast_output = forecast_output
    st.session_state.forecast_skipped = False
    return forecast_output


def get_step_states():
    active_dataset = st.session_state.get("active_dataset")
    can_forecast = bool(active_dataset is not None and not _should_block_analysis(active_dataset))
    forecast_current = has_successful_forecast()
    forecast_skipped = bool(
        active_dataset is not None
        and (
            st.session_state.get("forecast_skipped")
            or not dataset_supports_forecast(active_dataset)
        )
    )
    return build_step_states(
        has_loaded_source=st.session_state.get("loaded_source_state") is not None,
        has_cleaned_dataset=has_cleaned_dataset(),
        can_forecast=can_forecast,
        has_successful_forecast=forecast_current,
        forecast_skipped=forecast_skipped,
        can_analyze=can_forecast,
        has_exportable_output=has_successful_analysis() or forecast_current,
    )


def go_to_step(step_name):
    st.session_state.workflow_step = step_name
    st.session_state.workflow_step_selector_needs_sync = True
    st.rerun()


def render_step_header(step_name, title, caption):
    with st.container(border=True):
        render_section_heading(
            title,
            caption,
            kicker=step_kicker(step_name),
        )


def build_missing_values_takeaway(profile):
    if profile.missing_by_column.empty:
        return None

    top_missing = profile.missing_by_column.iloc[0]
    column_name = str(top_missing["column"])
    missing_count = int(top_missing["missing_count"])
    share_pct = 0.0
    if profile.missing_cell_count:
        share_pct = (missing_count / profile.missing_cell_count) * 100

    return (
        f"{column_name} is the main data-quality pressure point with {missing_count:,} missing cells, "
        f"or {share_pct:.1f}% of all remaining missing values."
    )


def build_explore_observations(profile):
    observations = [
        f"The dataset contains {profile.numeric_column_count} numeric, {profile.categorical_column_count} categorical, and {profile.datetime_column_count} datetime columns.",
    ]

    if profile.missing_cell_count:
        observations.append(
            build_missing_values_takeaway(profile)
            or f"{profile.missing_cell_count:,} missing cells remain after cleaning."
        )

    if profile.duplicate_row_count:
        observations.append(
            f"{profile.duplicate_row_count:,} duplicate rows are still present, which can inflate counts and mislead trend summaries."
        )

    chart_takeaway = build_chart_takeaway(profile.content_chart)
    if chart_takeaway:
        observations.append(chart_takeaway)

    return observations[:3]


def build_guided_explore_charts(profile):
    chart_items = []
    seen_titles = set()

    if not profile.missing_by_column.empty:
        chart_items.append(
            {
                "chart": build_missing_values_chart(profile),
                "takeaway": build_missing_values_takeaway(profile),
            }
        )

    for chart_spec in list(getattr(profile, "overview_charts", [])) + [profile.content_chart]:
        if chart_spec is None:
            continue
        if chart_spec.title in seen_titles:
            continue
        chart_items.append(
            {
                "chart": build_chart(chart_spec),
                "takeaway": build_chart_takeaway(chart_spec),
            }
        )
        seen_titles.add(chart_spec.title)
        if len(chart_items) >= 3:
            break

    return chart_items[:3]


def render_grouped_findings(findings):
    errors = [finding for finding in findings if finding.severity == "error"]
    warnings = [finding for finding in findings if finding.severity == "warning"]

    if errors:
        st.markdown("**Needs attention**")
        for finding in errors:
            st.error(finding.message)
            if finding.suggested_fix:
                st.caption(f"Suggested fix: {finding.suggested_fix}")

    if warnings:
        st.markdown("**Warnings**")
        for finding in warnings:
            st.warning(finding.message)
            if finding.suggested_fix:
                st.caption(f"Suggested fix: {finding.suggested_fix}")

    if not errors and not warnings:
        st.success("No validation findings remain after the recommended cleaning preview.")


def render_advanced_cleaning_controls():
    st.caption(
        "Preview updates automatically when advanced cleaning settings change. Core preprocessing still validates types, imputes missing values, and removes exact duplicates before analysis."
    )

    cleaning_columns = st.columns(4, gap="medium")
    cleaning_columns[0].checkbox("Parse dates", key="parse_dates", on_change=record_editor_state)
    cleaning_columns[1].checkbox(
        "Coerce numeric text",
        key="coerce_numeric_text",
        on_change=record_editor_state,
    )
    cleaning_columns[2].checkbox(
        "Trim strings",
        key="trim_strings",
        on_change=record_editor_state,
    )
    cleaning_columns[3].checkbox(
        "Drop duplicates",
        key="drop_duplicates_enabled",
        on_change=record_editor_state,
    )

    fill_columns = st.columns(4, gap="medium")
    fill_columns[0].selectbox(
        "Fill numeric nulls",
        options=["none", "mean", "median"],
        key="fill_numeric_nulls",
        on_change=record_editor_state,
    )
    fill_columns[1].selectbox(
        "Fill text nulls",
        options=["none", "missing"],
        key="fill_text_nulls",
        on_change=record_editor_state,
    )
    fill_columns[2].slider(
        "Drop rows with missing ratio >= ",
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        key="drop_null_rows_over",
        on_change=record_editor_state,
    )
    fill_columns[3].slider(
        "Drop columns with missing ratio >= ",
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        key="drop_null_columns_over",
        on_change=record_editor_state,
    )


def render_sidebar_navigation():
    step_states = get_step_states()
    current_step = st.session_state.get("workflow_step", "Upload")
    if st.session_state.get("workflow_step_selector_needs_sync"):
        st.session_state.workflow_step_selector = current_step
        st.session_state.workflow_step_selector_needs_sync = False
    elif st.session_state.get("workflow_step_selector") not in WORKFLOW_STEPS:
        st.session_state.workflow_step_selector = current_step

    unlocked_steps = sum(1 for step_state in step_states.values() if step_state.accessible)
    st.sidebar.markdown(
        """
        <div class="sidebar-intro">
            <div class="section-kicker">Guided Flow</div>
            <div class="sidebar-intro-title">One clear next step</div>
            <p class="sidebar-intro-copy">Move from upload to export without the clutter of a full control panel.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.caption(f"{unlocked_steps} of {len(WORKFLOW_STEPS)} steps unlocked")
    requested_step = st.sidebar.radio(
        "Workflow",
        options=list(WORKFLOW_STEPS),
        key="workflow_step_selector",
        format_func=lambda step_name: build_step_label(step_states[step_name]),
    )
    st.sidebar.toggle("Advanced Mode", key="advanced_mode")

    resolved_step, locked_reason = resolve_active_step(requested_step, step_states)
    if resolved_step != requested_step:
        st.sidebar.info(locked_reason or "Complete the previous step first.")
        st.session_state.workflow_step = resolved_step
        st.session_state.workflow_step_selector_needs_sync = True
        st.rerun()
    else:
        st.session_state.workflow_step = resolved_step

    loaded_source_state = st.session_state.get("loaded_source_state")
    if loaded_source_state is not None:
        st.sidebar.markdown(
            f"""
            <div class="sidebar-dataset-chip">
                <span>Active dataset</span>
                <strong>{escape(loaded_source_state["dataset_name"])}</strong>
                <small>{escape(loaded_source_state["source_label"])}</small>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return resolved_step, bool(st.session_state.advanced_mode)


def render_advanced_upload_controls(uploaded_source_file, workflow_definitions):
    with st.expander("Advanced Settings", expanded=False):
        st.markdown("**Saved workflows**")
        workflow_options = {"": None}
        for workflow_definition in workflow_definitions:
            workflow_options[workflow_label(workflow_definition)] = workflow_definition

        selected_workflow_label = st.selectbox(
            "Saved workflows",
            options=list(workflow_options.keys()),
            key="selected_workflow_label",
        )
        selected_workflow = workflow_options[selected_workflow_label]

        workflow_columns = st.columns(3, gap="medium")
        load_selected = workflow_columns[0].button(
            "Load workflow",
            disabled=selected_workflow is None,
            use_container_width=True,
        )
        rerun_selected = workflow_columns[1].button(
            "Rerun workflow",
            disabled=selected_workflow is None,
            use_container_width=True,
        )
        save_workflow_pressed = workflow_columns[2].button(
            "Save current setup",
            use_container_width=True,
        )

        st.text_input(
            "Workflow name",
            key="workflow_name",
            placeholder="Example: Monthly Sales Health Check",
            on_change=record_editor_state,
        )

        st.markdown("**Advanced source configuration**")
        st.selectbox(
            "Advanced source type",
            options=["csv", "excel", "postgres", "mysql"],
            key="source_type",
            on_change=record_editor_state,
        )

        reload_pressed = False
        if st.session_state.source_type == "csv":
            st.text_input("Delimiter", key="csv_delimiter", on_change=record_editor_state)
            st.text_input("Encoding", key="csv_encoding", on_change=record_editor_state)
            reload_pressed = st.button("Reload source with CSV settings", use_container_width=True)
        elif st.session_state.source_type == "excel":
            st.text_input("Sheet name or index", key="excel_sheet_name", on_change=record_editor_state)
            reload_pressed = st.button("Reload source with Excel settings", use_container_width=True)
        else:
            sql_col_one, sql_col_two = st.columns(2, gap="medium")
            sql_col_one.text_input("Host", key="sql_host", on_change=record_editor_state)
            sql_col_two.number_input(
                "Port",
                key="sql_port",
                min_value=1,
                step=1,
                on_change=record_editor_state,
            )
            sql_col_one.text_input("Database", key="sql_database", on_change=record_editor_state)
            sql_col_two.text_input("Username", key="sql_username", on_change=record_editor_state)
            st.text_input("Password", key="sql_password", type="password", on_change=record_editor_state)
            sql_mode_columns = st.columns(2, gap="medium")
            sql_mode_columns[0].text_input(
                "Table name",
                key="sql_table_name",
                on_change=record_editor_state,
                placeholder="orders",
            )
            sql_mode_columns[1].number_input(
                "Preview row limit",
                key="sql_limit",
                min_value=1,
                step=100,
                on_change=record_editor_state,
            )
            st.text_area(
                "Custom SELECT query",
                key="sql_query",
                on_change=record_editor_state,
                placeholder="SELECT * FROM orders WHERE order_date >= CURRENT_DATE - INTERVAL '90 days'",
            )
            reload_pressed = st.button("Connect SQL source", use_container_width=True)

        if load_selected and selected_workflow is not None:
            loading_placeholder = st.empty()
            try:
                render_loading_indicator(loading_placeholder, "Loading saved workflow...")
                load_workflow_into_editor(selected_workflow)
                preview_state = build_loaded_source_state(
                    deserialize_source_config(selected_workflow.source_config)
                )
                set_loaded_source_state(
                    preview_state,
                    workflow_id=selected_workflow.workflow_id,
                    workflow_version=selected_workflow.version,
                    workflow_name=selected_workflow.name,
                )
                loading_placeholder.empty()
                st.success(f"Loaded workflow '{selected_workflow.name}' into the guided flow.")
                go_to_step("Clean")
            except (DataSourceError, ValueError, RuntimeError) as error:
                loading_placeholder.empty()
                st.error(str(error))

        if rerun_selected and selected_workflow is not None:
            loading_placeholder = st.empty()
            try:
                render_loading_indicator(loading_placeholder, "Rerunning saved workflow...")
                load_workflow_into_editor(selected_workflow)
                run_saved_workflow(selected_workflow)
                loading_placeholder.empty()
                st.success(f"Reran workflow '{selected_workflow.name}' version {selected_workflow.version}.")
                go_to_step("Analyze")
            except (DataSourceError, RuntimeError, ValueError) as error:
                loading_placeholder.empty()
                st.error(str(error))

        if save_workflow_pressed:
            try:
                saved_workflow = save_current_workflow(uploaded_source_file)
                st.success(f"Saved workflow '{saved_workflow.name}' as version {saved_workflow.version}.")
            except (DataSourceError, ValueError) as error:
                st.error(str(error))

        if reload_pressed:
            loading_placeholder = st.empty()
            try:
                render_loading_indicator(loading_placeholder, "Refreshing source preview...")
                source_config = build_source_config_from_state(uploaded_source_file)
                preview_state = build_loaded_source_state(source_config)
                set_loaded_source_state(
                    preview_state,
                    workflow_id=st.session_state.active_workflow_id,
                    workflow_version=st.session_state.active_workflow_version,
                    workflow_name=st.session_state.active_workflow_name,
                )
                loading_placeholder.empty()
                st.success(f"Loaded {preview_state['dataset_name']} from {preview_state['source_label']}.")
            except (DataSourceError, ValueError) as error:
                loading_placeholder.empty()
                st.error(str(error))

        active_workflow_id = st.session_state.get("active_workflow_id")
        if active_workflow_id:
            render_workflow_run_history(active_workflow_id)


def render_upload_step(advanced_mode, workflow_definitions):
    render_step_header(
        "Upload",
        "Upload your dataset",
        "Start with a CSV or Excel file. Advanced connections and workflow controls stay available, but out of the way.",
    )

    uploaded_source_file = st.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx"],
        key="guided_uploaded_source_file",
    )

    if uploaded_source_file is not None:
        inferred_source_type = infer_source_type_from_upload(uploaded_source_file)
        st.session_state.source_type = inferred_source_type
        upload_signature = build_uploaded_file_signature(uploaded_source_file, inferred_source_type)

        if st.session_state.get("last_auto_loaded_source_signature") != upload_signature:
            loading_placeholder = st.empty()
            try:
                render_loading_indicator(loading_placeholder, "Loading your uploaded dataset...")
                if inferred_source_type == "excel":
                    source_config = ExcelSourceConfig(
                        file_name=uploaded_source_file.name,
                        file_bytes=uploaded_source_file.getvalue(),
                        sheet_name=str(st.session_state.excel_sheet_name or "0"),
                    )
                else:
                    source_config = CSVSourceConfig(
                        file_name=uploaded_source_file.name,
                        file_bytes=uploaded_source_file.getvalue(),
                        delimiter=str(st.session_state.csv_delimiter or ","),
                        encoding=str(st.session_state.csv_encoding or "utf-8"),
                    )

                preview_state = build_loaded_source_state(source_config)
                set_loaded_source_state(preview_state)
                st.session_state.last_auto_loaded_source_signature = upload_signature
                loading_placeholder.empty()
                st.success(f"Loaded {preview_state['dataset_name']} from {preview_state['source_label']}.")
            except DataSourceError as error:
                loading_placeholder.empty()
                st.error(str(error))
    else:
        st.session_state.last_auto_loaded_source_signature = None

    loaded_source_state = st.session_state.get("loaded_source_state")
    if loaded_source_state is not None:
        summary_columns = st.columns(3)
        summary_columns[0].metric("Rows", f"{len(loaded_source_state['raw_df']):,}")
        summary_columns[1].metric("Columns", f"{len(loaded_source_state['raw_df'].columns):,}")
        summary_columns[2].metric("Source", loaded_source_state["source_label"])

        with st.container(border=True):
            render_section_heading(
                "Preview",
                "A quick look at the loaded raw dataset before any cleaning is applied.",
            )
            st.dataframe(
                loaded_source_state["raw_df"].head(10),
                use_container_width=True,
                height=420,
            )

        if st.button("Continue to Clean", type="primary", use_container_width=True):
            go_to_step("Clean")
    else:
        with st.container(border=True):
            render_section_heading(
                "Ready to begin",
                "Upload a CSV or Excel file to unlock validation, cleaning, and AI-guided analysis.",
            )

    if advanced_mode:
        render_advanced_upload_controls(uploaded_source_file, workflow_definitions)

    return uploaded_source_file


def render_clean_step(advanced_mode):
    render_step_header(
        "Clean",
        "Clean and validate",
        "Aidssist previews safe fixes first, then applies them only when you confirm.",
    )

    loaded_source_state = st.session_state.get("loaded_source_state")
    if loaded_source_state is None:
        st.info("Load a dataset in the Upload step to preview cleaning recommendations.")
        return

    preview_state = refresh_cleaning_preview_state()
    warning_count, error_count = summarize_findings(preview_state["validation_findings"])
    issue_count = warning_count + error_count
    current_options_signature = preview_state["cleaning_options_signature"]
    cleaning_report = normalize_cleaning_report(preview_state.get("cleaning_report"))

    metrics = st.columns(4)
    metrics[0].metric("Data quality score", f"{cleaning_report['quality_score']:.2f}")
    metrics[1].metric("Missing fixed", f"{cleaning_report['missing_handled']:,}")
    metrics[2].metric("Duplicates removed", f"{cleaning_report['duplicates_removed']:,}")
    metrics[3].metric("Issues", f"{issue_count:,}")

    render_cleaning_report_panel(preview_state.get("cleaning_report"))

    with st.container(border=True):
        render_section_heading(
            "Suggested fixes",
            "These are the safe cleaning actions Aidssist is ready to apply to the active source.",
        )
        if preview_state["cleaning_actions"]:
            for action in preview_state["cleaning_actions"]:
                st.write(f"- {action}")
        else:
            st.write("No automatic cleaning changes are required with the current defaults.")

    with st.container(border=True):
        render_section_heading(
            "Validation findings",
            "Review what still needs attention before the cleaned dataset becomes your active workspace.",
        )
        render_grouped_findings(preview_state["validation_findings"])

    if advanced_mode:
        with st.expander("Advanced cleaning settings", expanded=False):
            render_advanced_cleaning_controls()
        if loaded_source_state.get("raw_findings"):
            with st.expander("Raw dataset findings", expanded=False):
                render_grouped_findings(loaded_source_state["raw_findings"])

    apply_pressed = st.button("Apply Cleaning and Continue", type="primary", use_container_width=True)
    if apply_pressed:
        loading_placeholder = st.empty()
        try:
            render_loading_indicator(loading_placeholder, "Applying cleaning and opening the explore step...")
            dataset_state = promote_cleaning_preview_to_active_dataset()
            st.session_state.guided_flow_notice = f"Applied cleaning to {dataset_state['dataset_name']}."
            loading_placeholder.empty()
            go_to_step("Explore")
        except RuntimeError as error:
            loading_placeholder.empty()
            st.error(str(error))

    active_dataset = st.session_state.get("active_dataset")
    cleaned_current_source = is_cleaned_dataset_current(
        active_dataset=active_dataset,
        loaded_source_state=loaded_source_state,
        current_options_signature=current_options_signature,
        applied_source_fingerprint=st.session_state.get("cleaning_applied_source_fingerprint"),
        applied_options_signature=st.session_state.get("cleaning_applied_options_signature"),
    )
    source_matches_active_dataset = (
        active_dataset is not None
        and active_dataset.get("source_fingerprint") == loaded_source_state.get("source_fingerprint")
    )
    if source_matches_active_dataset and not cleaned_current_source:
        st.info("Cleaning settings changed after the last apply. Apply cleaning again to refresh the active dataset.")

    if cleaned_current_source:
        active_warning_count, active_error_count = summarize_findings(active_dataset["validation_findings"])
        if active_error_count == 0 and active_warning_count > 0:
            st.checkbox(
                "I understand the remaining warnings and still want to allow analysis later.",
                key="validation_acknowledged",
            )

        continue_disabled = False
        if st.button("Continue to Explore", disabled=continue_disabled, use_container_width=True):
            go_to_step("Explore")


def render_explore_step(advanced_mode):
    render_step_header(
        "Explore",
        "Explore the cleaned dataset",
        "Start with the most useful overview signals, then drill deeper only if you need more detail.",
    )

    active_dataset = st.session_state.get("active_dataset")
    if active_dataset is None:
        st.info("Apply cleaning in the Clean step to unlock exploration.")
        return

    guided_flow_notice = st.session_state.pop("guided_flow_notice", None)
    if guided_flow_notice:
        st.success(guided_flow_notice)

    profile = active_dataset["profile"]
    metrics = st.columns(4)
    metrics[0].metric("Rows", f"{profile.row_count:,}")
    metrics[1].metric("Columns", f"{profile.column_count:,}")
    metrics[2].metric("Missing cells", f"{profile.missing_cell_count:,}")
    metrics[3].metric("Duplicate rows", f"{profile.duplicate_row_count:,}")

    render_insight_cards(
        "Instant insights",
        "Aidssist surfaces the highest-signal observations first so you can decide where to look next.",
        build_explore_observations(profile),
        kicker="Decision view",
    )

    chart_items = build_guided_explore_charts(profile)
    if chart_items:
        chart_columns = st.columns(len(chart_items), gap="large")
        for column, chart_item in zip(chart_columns, chart_items):
            with column:
                with st.container(border=True):
                    st.altair_chart(chart_item["chart"], use_container_width=True)
                    if chart_item.get("takeaway"):
                        st.markdown(
                            f'<div class="chart-note">{escape(chart_item["takeaway"])}</div>',
                            unsafe_allow_html=True,
                        )

    with st.container(border=True):
        render_section_heading(
            "Preview table",
            "A lightweight look at the cleaned dataset you will analyze.",
        )
        st.dataframe(active_dataset["df"].head(15), use_container_width=True, height=420)

    analyze_locked = _should_block_analysis(active_dataset)
    forecast_supported = dataset_supports_forecast(active_dataset)
    if analyze_locked:
        st.info("Review the Clean step before analysis. Remaining blockers or unacknowledged warnings still need attention.")
    continue_label = "Continue to Forecast" if forecast_supported else "Continue to Analyze"
    if st.button(continue_label, disabled=analyze_locked, type="primary", use_container_width=True):
        if forecast_supported:
            go_to_step("Forecast")
        else:
            st.session_state.forecast_skipped = True
            st.session_state.guided_flow_notice = "Aidssist detected a non-time dataset, so analysis will continue without the forecast step."
            go_to_step("Analyze")

    if advanced_mode:
        with st.expander("Advanced exploration", expanded=False):
            render_dataset_dashboard(profile)
            render_data_explorer(active_dataset["df"], active_dataset["dataset_key"])
            render_column_explorer(active_dataset["df"], profile, active_dataset["dataset_key"])


def build_forecast_chart(forecast_output):
    chart_records = pd.DataFrame(forecast_output.get("chart_records", []))
    if chart_records.empty:
        return None

    chart_records = chart_records.copy()
    chart_records["date"] = pd.to_datetime(chart_records["date"], errors="coerce")
    chart_records = chart_records.dropna(subset=["date"])
    if chart_records.empty:
        return None

    historical = chart_records[chart_records["series"] == "Historical"]
    forecast = chart_records[chart_records["series"] == "Forecast"]

    history_line = (
        alt.Chart(historical)
        .mark_line(strokeWidth=3, color="#38BDF8")
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("value:Q", title="Value"),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("value:Q", title="Historical"),
            ],
        )
    )
    forecast_line = (
        alt.Chart(forecast)
        .mark_line(strokeWidth=3, strokeDash=[6, 4], color="#FB923C")
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("value:Q", title="Value"),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("value:Q", title="Forecast"),
                alt.Tooltip("lower_bound:Q", title="Lower bound"),
                alt.Tooltip("upper_bound:Q", title="Upper bound"),
            ],
        )
    )
    confidence_band = (
        alt.Chart(forecast)
        .mark_area(opacity=0.16, color="#FB923C")
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("lower_bound:Q", title="Value"),
            y2="upper_bound:Q",
        )
    )
    points = (
        alt.Chart(forecast)
        .mark_circle(size=64, color="#FDBA74")
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("value:Q", title="Value"),
        )
    )

    return (
        (confidence_band + history_line + forecast_line + points)
        .properties(height=320, title="Historical trend and forecast")
        .configure_view(stroke=None)
        .configure_axis(
            labelColor="#CBD5E1",
            titleColor="#E2E8F0",
            gridColor="rgba(148, 163, 184, 0.12)",
            domainColor="rgba(148, 163, 184, 0.12)",
            tickColor="rgba(148, 163, 184, 0.12)",
        )
        .configure_title(color="#F8FAFC", fontSize=18, anchor="start")
    )


def render_forecast_output(forecast_output):
    metric_columns = st.columns(4)
    metric_columns[0].metric("Chosen model", str(forecast_output.get("chosen_model", "N/A")).replace("_", " ").title())
    metric_columns[1].metric("Trend", str(forecast_output.get("trend_status", "stable")).replace("_", " ").title())
    metric_columns[2].metric("Frequency", FREQUENCY_LABELS.get(forecast_output.get("resolved_frequency"), "N/A"))
    metric_columns[3].metric("History points", f"{int(forecast_output.get('history_points', 0)):,}")

    with st.container(border=True):
        render_section_heading(
            "Forecast summary",
            "Aidssist projects the selected KPI and turns the model output into ranked business recommendations.",
            kicker="Forecast",
        )
        st.write(forecast_output.get("summary") or "No forecast summary is available.")

    chart = build_forecast_chart(forecast_output)
    summary_columns = st.columns([1.45, 1], gap="large")
    with summary_columns[0]:
        with st.container(border=True):
            render_section_heading(
                "Forecast outlook",
                "Historical performance, projected values, and the uncertainty band for the selected horizon.",
            )
            if chart is not None:
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("Forecast chart data is unavailable for this run.")

    with summary_columns[1]:
        with st.container(border=True):
            render_section_heading(
                "Risk view",
                "Confidence band width and backtesting metrics help you judge how much to trust the forecast.",
            )
            evaluation_metrics = forecast_output.get("evaluation_metrics", {})
            st.metric("MAPE", f"{float(evaluation_metrics.get('mape', 0.0)):.1f}%")
            st.metric("RMSE", f"{float(evaluation_metrics.get('rmse', 0.0)):.2f}")
            st.metric("Directional accuracy", f"{float(evaluation_metrics.get('directional_accuracy', 0.0)):.1f}%")
            st.metric("Uncertainty ratio", f"{float(forecast_output.get('uncertainty_ratio', 0.0)):.0%}")

    detail_columns = st.columns(2, gap="large")
    with detail_columns[0]:
        with st.container(border=True):
            render_section_heading(
                "Model comparison",
                "Aidssist compares baselines and stronger models with rolling backtests before selecting the forecast.",
            )
            comparison_table = forecast_output.get("comparison_table")
            if isinstance(comparison_table, pd.DataFrame) and not comparison_table.empty:
                st.dataframe(
                    comparison_table,
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("Model-comparison data is unavailable for this forecast.")
    with detail_columns[1]:
        with st.container(border=True):
            render_section_heading(
                "Driver importance",
                "The top mapped drivers indicate which levers most strongly shape the forecast.",
            )
            driver_importance_table = forecast_output.get("driver_importance_table")
            if isinstance(driver_importance_table, pd.DataFrame) and not driver_importance_table.empty:
                st.dataframe(
                    driver_importance_table,
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No driver-importance table is available for this run.")

    with st.container(border=True):
        render_section_heading(
            "Decision recommendations",
            "These are advisory recommendations for humans to review, not automated actions.",
        )
        recommendations = list(forecast_output.get("recommendations", []))
        if not recommendations:
            st.info("No structured recommendations were generated for this forecast.")
        else:
            for recommendation in recommendations:
                st.markdown(
                    f"**{escape(str(recommendation.get('title', 'Recommendation')))}**"
                )
                st.write(recommendation.get("recommended_action") or "")
                st.caption(
                    f"{recommendation.get('category', 'general')} | "
                    f"impact: {recommendation.get('impact_direction', 'review')} | "
                    f"confidence: {recommendation.get('confidence', 'medium')}"
                )
                if recommendation.get("rationale"):
                    st.write(recommendation["rationale"])

        with st.expander("Forecast technical details", expanded=False):
            st.markdown("**Forecast table**")
            forecast_table = forecast_output.get("forecast_table")
            if isinstance(forecast_table, pd.DataFrame) and not forecast_table.empty:
                st.dataframe(
                    forecast_table,
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No forecast table is available for this run.")
            st.markdown("**Artifact metadata**")
            st.json(forecast_output.get("artifact_metadata") or {})


def render_forecast_history(selected_workflow_id=None, source_fingerprint=None):
    artifacts = WORKFLOW_STORE.list_forecast_artifacts(
        limit=10,
        workflow_id=selected_workflow_id or None,
        source_fingerprint=source_fingerprint or None,
    )
    with st.container(border=True):
        render_section_heading(
            "Forecast Run History",
            "Saved forecast runs keep the model choice, target, horizon, and artifact key for reproducibility.",
            kicker="Forecast Audit",
        )
        if not artifacts:
            st.info("No forecast runs have been recorded yet.")
            return

        history_rows = [
            {
                "created_at": artifact.created_at,
                "workflow": artifact.workflow_name or "Ad hoc",
                "version": artifact.workflow_version or "-",
                "target": artifact.target_column,
                "horizon": artifact.horizon,
                "model": artifact.model_name,
                "status": artifact.status,
            }
            for artifact in artifacts
        ]
        st.dataframe(pd.DataFrame(history_rows), use_container_width=True, hide_index=True)


def render_forecast_step(advanced_mode):
    render_step_header(
        "Forecast",
        "Forecast future outcomes",
        "Map one primary KPI, validate the time series strictly, and generate decision recommendations from a deterministic forecasting engine.",
    )

    active_dataset = st.session_state.get("active_dataset")
    if active_dataset is None:
        st.info("Apply cleaning and explore the dataset first to unlock forecasting.")
        return

    routing_snapshot = get_dataset_intelligence_snapshot(active_dataset)
    if routing_snapshot is not None and not dataset_supports_forecast(active_dataset):
        intelligence = routing_snapshot["intelligence"]
        guided_mode = routing_snapshot["guided_mode"]
        with st.container(border=True):
            render_section_heading(
                "System decision",
                "Aidssist detected that this dataset should bypass the forecast workflow and move directly into the safer analysis path.",
                kicker="Auto route",
            )
            selected_mode = str(guided_mode.get("selected_mode") or "analysis").upper()
            reason_text = str(guided_mode.get("reason") or "").strip()
            if not intelligence.get("numeric_columns"):
                reason_text = "Aidssist could not find a usable numeric KPI candidate for safe forecasting."
            st.markdown(f"**Selected mode:** `{selected_mode}`")
            if reason_text:
                st.write(reason_text)
            st.caption(
                "Datetime columns: "
                + (", ".join(intelligence.get("datetime_columns") or []) or "none")
                + " | Numeric columns: "
                + str(len(intelligence.get("numeric_columns") or []))
                + " | Categorical columns: "
                + str(len(intelligence.get("categorical_columns") or []))
            )
            st.info("Forecast mapping is hidden because the dataset is missing either a reliable time-series route or a usable numeric KPI.")

        st.session_state.forecast_skipped = True
        action_columns = st.columns(2, gap="medium")
        if action_columns[0].button("Continue to Analyze", type="primary", use_container_width=True):
            st.session_state.guided_flow_notice = "Forecast was skipped automatically for this dataset."
            go_to_step("Analyze")
        if action_columns[1].button("Back to Explore", use_container_width=True):
            go_to_step("Explore")
        return

    ensure_forecast_defaults(active_dataset)
    mapping_options = get_forecast_mapping_options(active_dataset["df"])

    date_options = [""] + list(dict.fromkeys(
        list(mapping_options.get("date_columns", [])) + [str(st.session_state.forecast_date_column or "")]
    ))
    target_options = [""] + list(dict.fromkeys(
        list(mapping_options.get("target_columns", [])) + [str(st.session_state.forecast_target_column or "")]
    ))
    driver_options = [
        column
        for column in dict.fromkeys(
            list(mapping_options.get("driver_columns", [])) + list(st.session_state.forecast_driver_columns or [])
        )
        if column
    ]

    with st.container(border=True):
        render_section_heading(
            "Forecast mapping",
            "Choose the date column, one primary KPI, optional business drivers, and the horizon you want Aidssist to project.",
        )
        control_columns = st.columns(3, gap="medium")
        control_columns[0].selectbox(
            "Date column",
            options=date_options,
            key="forecast_date_column",
            on_change=record_editor_state,
        )
        control_columns[1].selectbox(
            "Primary target metric",
            options=target_options,
            key="forecast_target_column",
            on_change=record_editor_state,
        )
        control_columns[2].selectbox(
            "Forecast horizon",
            options=[item["value"] for item in mapping_options.get("horizon_options", [])],
            format_func=lambda value: HORIZON_LABELS.get(value, value.replace("_", " ").title()),
            key="forecast_horizon",
            on_change=record_editor_state,
        )

        detail_columns = st.columns(3, gap="medium")
        detail_columns[0].multiselect(
            "Optional drivers",
            options=driver_options,
            key="forecast_driver_columns",
            on_change=record_editor_state,
        )
        detail_columns[1].selectbox(
            "Aggregation frequency",
            options=[item["value"] for item in mapping_options.get("frequency_options", [])],
            format_func=lambda value: FREQUENCY_LABELS.get(value, value),
            key="forecast_frequency",
            on_change=record_editor_state,
        )
        detail_columns[2].selectbox(
            "Model strategy",
            options=mapping_options.get("model_strategy_options", []),
            key="forecast_model_strategy",
            on_change=record_editor_state,
        )

        mode_columns = st.columns(2, gap="medium")
        mode_columns[0].selectbox(
            "Training mode",
            options=mapping_options.get("training_mode_options", []),
            key="forecast_training_mode",
            on_change=record_editor_state,
        )
        mode_columns[1].caption(
            "Auto uses the background runtime when `AIDSSIST_API_URL` is configured, otherwise it runs locally."
        )

    validation = get_forecast_validation_for_active_dataset()
    if validation is not None:
        metric_columns = st.columns(4)
        metric_columns[0].metric("Resolved frequency", FREQUENCY_LABELS.get(validation.resolved_frequency, "Pending"))
        metric_columns[1].metric("History points", f"{validation.history_points:,}")
        metric_columns[2].metric("Minimum needed", f"{validation.minimum_history_points:,}")
        metric_columns[3].metric("Date range", f"{validation.date_start or 'N/A'} to {validation.date_end or 'N/A'}")

        with st.container(border=True):
            render_section_heading(
                "Forecast validation gate",
                "Aidssist blocks training until the mapped date, target, history length, and horizon are all safe for forecasting.",
            )
            if validation.errors:
                st.markdown("**Blocking issues**")
                for error_message in validation.errors:
                    st.error(error_message)
            if validation.warnings:
                st.markdown("**Warnings**")
                for warning_message in validation.warnings:
                    st.warning(warning_message)
            if validation.compatible_horizons:
                st.caption(
                    "Compatible horizons: "
                    + ", ".join(HORIZON_LABELS.get(item, item) for item in validation.compatible_horizons)
                )
            if validation.is_valid and not validation.warnings:
                st.success("The forecast mapping passed validation and is ready to run.")

    forecast_current = has_successful_forecast()
    forecast_output = st.session_state.get("forecast_output")
    if forecast_output and not forecast_current and forecast_output.get("dataset_key") == active_dataset.get("dataset_key"):
        st.info("Forecast settings changed after the last run. Rerun the forecast to refresh the projection and recommendations.")

    action_columns = st.columns(2, gap="medium")
    run_forecast_pressed = action_columns[0].button(
        "Run Forecast",
        type="primary",
        use_container_width=True,
    )
    skip_forecast_pressed = action_columns[1].button(
        "Skip Forecast for Now",
        use_container_width=True,
    )

    if skip_forecast_pressed:
        st.session_state.forecast_skipped = True
        st.session_state.guided_flow_notice = "Forecast skipped. You can still return here later."
        go_to_step("Analyze")

    if run_forecast_pressed:
        if validation is None or not validation.is_valid:
            st.warning("Resolve the forecast mapping errors before running the model.")
        else:
            loading_placeholder = st.empty()
            try:
                render_loading_indicator(loading_placeholder, "Training the forecast engine and ranking business recommendations...")
                forecast_output = run_forecast_for_active_dataset()
                loading_placeholder.empty()
                st.success("Forecast ready.")
            except RuntimeError as error:
                loading_placeholder.empty()
                st.session_state.forecast_output = None
                st.error(str(error))
                return

    if forecast_current and st.session_state.get("forecast_output") is not None:
        render_forecast_output(st.session_state.forecast_output)
        if st.button("Continue to Analyze", type="primary", use_container_width=True):
            go_to_step("Analyze")
    elif st.session_state.get("forecast_skipped"):
        st.info("Forecast is currently skipped for this dataset. You can continue to analysis or come back here later.")
        if st.button("Continue to Analyze", type="primary", use_container_width=True):
            go_to_step("Analyze")

    if advanced_mode:
        with st.expander("Advanced forecast history", expanded=False):
            render_forecast_history(
                selected_workflow_id=active_dataset.get("workflow_id"),
                source_fingerprint=active_dataset.get("source_fingerprint"),
            )


def render_analyze_step(api_ready, api_message, advanced_mode):
    render_step_header(
        "Analyze",
        "Ask your data",
        "This is the core experience: ask a business question, get the insight, supporting chart, and recommended action.",
    )

    active_dataset = st.session_state.get("active_dataset")
    if active_dataset is None:
        st.info("Apply cleaning and explore the dataset first to unlock analysis.")
        return

    forecast_skipped = bool(
        st.session_state.get("forecast_skipped")
        or not dataset_supports_forecast(active_dataset)
    )
    if not has_successful_forecast() and not forecast_skipped:
        st.info("Complete the Forecast step or skip it for now before running analysis.")
        return

    forecast_output = st.session_state.get("forecast_output")
    if has_successful_forecast() and forecast_output is not None:
        st.caption(
            f"Forecast context: {forecast_output.get('summary') or 'Forecast completed successfully.'}"
        )
    elif forecast_skipped:
        st.caption("Forecast is skipped for this dataset. Analysis will run directly on the cleaned data.")

    if not api_ready:
        st.warning("Aidssist runtime setup is required before analysis can run.")
        st.caption(api_message)
    else:
        st.caption(api_message)

    test_cases = get_default_test_cases()
    suggestion_columns = st.columns(len(test_cases), gap="medium")
    for index, (column, test_case) in enumerate(zip(suggestion_columns, test_cases)):
        with column:
            if st.button(
                test_case["query"],
                key=f"guided_query_{active_dataset['dataset_key']}_{index}",
                use_container_width=True,
            ):
                st.session_state.query_input = test_case["query"]
                st.rerun()

    hero_columns = st.columns([1, 2.2, 1])
    with hero_columns[1]:
        with st.container(border=True):
            st.text_area(
                "Ask your data",
                key="query_input",
                height=140,
                placeholder="Example: Which segments are driving the biggest revenue swings over time?",
            )
            run_pressed = st.button(
                "Run Analysis",
                disabled=not api_ready or _should_block_analysis(active_dataset),
                type="primary",
                use_container_width=True,
            )

    if _should_block_analysis(active_dataset):
        st.info("Return to the Clean step to resolve blockers or acknowledge warnings before running analysis.")
        run_pressed = False

    if run_pressed:
        query = str(st.session_state.query_input or "").strip()
        if not query:
            st.warning("Enter a question before running the analysis.")
        else:
            loading_placeholder = st.empty()
            try:
                render_loading_indicator(loading_placeholder, "Analyzing your dataset...")
                analysis_output = run_analysis_for_active_dataset(query)
            except RuntimeError as error:
                loading_placeholder.empty()
                st.session_state.analysis_output = None
                st.error(str(error))
                return

            loading_placeholder.empty()
            analysis_output["dataset_key"] = active_dataset["dataset_key"]
            st.session_state.analysis_output = analysis_output
            if analysis_output.get("error"):
                st.error("Analysis finished with an error.")
            else:
                st.success("Analysis ready.")

    analysis = st.session_state.get("analysis_output")
    if analysis and analysis.get("dataset_key") == active_dataset["dataset_key"]:
        render_analysis_output(analysis, show_customization=advanced_mode)


def render_export_step(advanced_mode):
    render_step_header(
        "Export",
        "Export results",
        "Download the cleaned dataset plus the strongest artifacts from the latest forecast and analysis runs.",
    )

    active_dataset = st.session_state.get("active_dataset")
    analysis = st.session_state.get("analysis_output")
    forecast_output = st.session_state.get("forecast_output")
    forecast_current = has_successful_forecast()
    analysis_current = (
        active_dataset is not None
        and analysis is not None
        and analysis.get("dataset_key") == active_dataset.get("dataset_key")
        and not analysis.get("error")
    )
    if active_dataset is None or (not forecast_current and not analysis_current):
        st.info("Run a forecast or a successful analysis to unlock exports.")
        return

    profile = active_dataset["profile"]
    dataset_stem = Path(profile.dataset_name).stem
    primary_columns = st.columns([1.4, 1], gap="large")
    with primary_columns[0]:
        with st.container(border=True):
            render_section_heading(
                "Cleaned dataset",
                "Export the cleaned dataframe currently powering forecasting, exploration, and analysis.",
            )
            st.download_button(
                "Download cleaned dataset CSV",
                data=dataframe_to_csv_bytes(active_dataset["df"]),
                file_name=f"{dataset_stem}_cleaned.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True,
            )

    with primary_columns[1]:
        with st.container(border=True):
            render_section_heading(
                "Data dictionary",
                "Keep a lightweight schema reference alongside your exported artifacts.",
            )
            st.download_button(
                "Download data dictionary CSV",
                data=dataframe_to_csv_bytes(profile.data_dictionary),
                file_name=f"{dataset_stem}_data_dictionary.csv",
                mime="text/csv",
                use_container_width=True,
            )

    if forecast_current and forecast_output is not None:
        with st.container(border=True):
            render_section_heading(
                "Forecast artifacts",
                "Export the forecast table, evaluation report, recommendations, and model metadata from the latest ML run.",
            )
            forecast_payloads = build_forecast_download_payloads(forecast_output, dataset_stem)
            forecast_columns = st.columns(min(3, len(forecast_payloads)), gap="medium")
            for index, (label, data, file_name, mime) in enumerate(forecast_payloads):
                with forecast_columns[index % len(forecast_columns)]:
                    st.download_button(
                        label,
                        data=data,
                        file_name=file_name,
                        mime=mime,
                        use_container_width=True,
                    )

    if analysis_current:
        result_profile = profile_analysis_result(analysis["result"])
        result_file_name, result_payload, result_mime, result_label = build_result_download_payload(
            analysis["result"]
        )
        summary_text = analysis.get("summary") or analysis.get("insights") or "No summary was produced."
        chart_payloads = build_chart_download_payloads(
            result_profile.chart,
            f"{dataset_stem}_analysis",
            "Download chart data CSV",
        )

        secondary_columns = st.columns(2, gap="large")
        with secondary_columns[0]:
            with st.container(border=True):
                render_section_heading(
                    "Analysis result",
                    "The primary artifact from the latest successful analysis run.",
                )
                st.download_button(
                    result_label,
                    data=result_payload,
                    file_name=result_file_name,
                    mime=result_mime,
                    use_container_width=True,
                )
                st.download_button(
                    "Download analysis summary",
                    data=summary_text.encode("utf-8"),
                    file_name=f"{dataset_stem}_analysis_summary.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

        with secondary_columns[1]:
            with st.container(border=True):
                render_section_heading(
                    "Analysis chart data",
                    "Download the data behind the inferred result chart when available.",
                )
                if chart_payloads:
                    chart_label, chart_data, chart_file_name, chart_mime = chart_payloads[0]
                    st.download_button(
                        chart_label,
                        data=chart_data,
                        file_name=chart_file_name,
                        mime=chart_mime,
                        use_container_width=True,
                    )
                else:
                    st.info("No chart-data download is available for this result shape.")

    if advanced_mode:
        with st.expander("Advanced exports", expanded=False):
            render_export_center(active_dataset["df"], profile, analysis if analysis_current else None)
            if active_dataset.get("workflow_id"):
                render_workflow_run_history(active_dataset.get("workflow_id"))
                render_forecast_history(
                    selected_workflow_id=active_dataset.get("workflow_id"),
                    source_fingerprint=active_dataset.get("source_fingerprint"),
                )


apply_custom_styles()
initialize_app_state()
st.markdown('<div class="page-top-spacer"></div>', unsafe_allow_html=True)
render_app_header()

api_ready, api_message = get_runtime_configuration_status(get_provider_configuration_status)
if st.sidebar.toggle("Companion Console", key="companion_console_mode"):
    render_companion_console(
        workflow_store=WORKFLOW_STORE,
        api_ready=api_ready,
        api_message=api_message,
    )
    st.stop()

workflow_definitions = WORKFLOW_STORE.list_workflows()
current_step, advanced_mode = render_sidebar_navigation()

if current_step == "Upload":
    render_upload_step(advanced_mode, workflow_definitions)
elif current_step == "Clean":
    render_clean_step(advanced_mode)
elif current_step == "Explore":
    render_explore_step(advanced_mode)
elif current_step == "Forecast":
    render_forecast_step(advanced_mode)
elif current_step == "Analyze":
    render_analyze_step(api_ready, api_message, advanced_mode)
else:
    render_export_step(advanced_mode)
