from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd


FIXTURE_PATH = Path(__file__).resolve().parent.parent / "load_tests" / "fixtures" / "sales.csv"
DEMO_QUERIES = [
    "Predict next month sales",
    "Top 5 products",
    "Why revenue dropped",
]
DEMO_TIME_FILTER = "last_quarter"


def _load_demo_dataframe() -> pd.DataFrame:
    dataframe = pd.read_csv(FIXTURE_PATH)
    if "order_date" in dataframe.columns:
        dataframe["order_date"] = pd.to_datetime(dataframe["order_date"], errors="coerce", format="mixed")
    return dataframe


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    serializable = frame.copy()
    for column in serializable.columns:
        if pd.api.types.is_datetime64_any_dtype(serializable[column]):
            serializable[column] = serializable[column].dt.strftime("%Y-%m-%d")
    return serializable.where(pd.notna(serializable), None).to_dict(orient="records")


def _demo_dataset_payload(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "metadata": {
            "file_name": FIXTURE_PATH.name,
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "columns": [str(column) for column in df.columns.tolist()],
            "time_column": "order_date",
            "target_column": "sales",
            "sample_domain": "retail_sales",
        },
        "rows": _records(df),
    }


def _base_demo_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily_sales = df.groupby("order_date", dropna=False, as_index=False)["sales"].sum().sort_values("order_date")
    region_sales = df.groupby("region", as_index=False)["sales"].sum().sort_values("sales", ascending=False)
    product_sales = df.groupby("product", as_index=False)["sales"].sum().sort_values("sales", ascending=False)
    return daily_sales, region_sales, product_sales


def _dashboard_layout() -> dict[str, Any]:
    return {"type": "grid", "columns": 12, "row_height": 132, "mode": "demo_ready"}


def _build_analysis_dashboard(df: pd.DataFrame) -> dict[str, Any]:
    daily_sales, region_sales, product_sales = _base_demo_metrics(df)
    top_region = str(region_sales.iloc[0]["region"]) if not region_sales.empty else "n/a"
    top_product = str(product_sales.iloc[0]["product"]) if not product_sales.empty else "n/a"
    return {
        "charts": [
            {
                "type": "line",
                "purpose": "trend",
                "x": "order_date",
                "y": "sales",
                "time_column": "order_date",
                "title": "Daily sales trend",
                "rows": _records(daily_sales),
            },
            {
                "type": "bar",
                "purpose": "regional_mix",
                "x": "region",
                "y": "sales",
                "title": "Revenue by region",
                "rows": _records(region_sales),
            },
            {
                "type": "pie",
                "purpose": "product_mix",
                "x": "product",
                "y": "sales",
                "title": "Revenue by product",
                "rows": _records(product_sales),
            },
        ],
        "filters": ["current_month", "last_month", "last_quarter", "last_year"],
        "kpis": [
            {"metric": "total_sales", "value": int(df["sales"].sum())},
            {"metric": "average_order_value", "value": round(float(df["sales"].mean()), 2)},
            {"metric": "top_region", "value": top_region},
            {"metric": "top_product", "value": top_product},
        ],
        "layout": _dashboard_layout(),
        "drilldown_ready": True,
        "time_column": "order_date",
        "active_filter": DEMO_TIME_FILTER,
        "visualization_type": "line",
    }


def _build_forecast_dashboard(df: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    daily_sales, region_sales, _ = _base_demo_metrics(df)
    values = daily_sales["sales"].astype(float).reset_index(drop=True)
    indexes = pd.Series(range(len(values)), dtype="float64")
    slope = float(values.diff().fillna(0).tail(5).mean()) if len(values) > 1 else 0.0
    intercept = float(values.iloc[0]) if not values.empty else 0.0

    future_dates = pd.date_range(daily_sales["order_date"].max() + pd.Timedelta(days=1), periods=7, freq="D")
    history_rows = [
        {"date": row["order_date"], "value": row["sales"], "series": "Actual"}
        for row in _records(daily_sales)
    ]
    predicted_history = intercept + (indexes * slope)
    residual = (values - predicted_history).abs()
    band = float(max(residual.mean(), 85.0)) if not residual.empty else 85.0

    forecast_points: list[dict[str, Any]] = []
    combined_rows = list(history_rows)
    for offset, date_value in enumerate(future_dates, start=len(values)):
        projected = max(0.0, intercept + (offset * slope))
        lower_bound = max(0.0, projected - band)
        upper_bound = projected + band
        point = {
            "date": date_value.strftime("%Y-%m-%d"),
            "value": round(projected, 2),
            "lower_bound": round(lower_bound, 2),
            "upper_bound": round(upper_bound, 2),
        }
        forecast_points.append(point)
        combined_rows.append({"date": point["date"], "value": point["value"], "series": "Forecast"})

    dashboard = {
        "charts": [
            {
                "type": "line",
                "purpose": "forecast",
                "x": "date",
                "y": "value",
                "time_column": "date",
                "title": "Actual vs projected sales",
                "rows": combined_rows,
            },
            {
                "type": "bar",
                "purpose": "regional_mix",
                "x": "region",
                "y": "sales",
                "title": "Current regional run-rate",
                "rows": _records(region_sales),
            },
        ],
        "filters": ["current_month", "last_month", "last_quarter", "last_year"],
        "kpis": [
            {"metric": "next_7_day_projection", "value": round(sum(item["value"] for item in forecast_points), 2)},
            {"metric": "trend_direction", "value": "rising" if slope >= 0 else "softening"},
            {"metric": "forecast_confidence", "value": "medium"},
            {"metric": "history_points", "value": int(len(daily_sales))},
        ],
        "layout": _dashboard_layout(),
        "drilldown_ready": True,
        "time_column": "date",
        "active_filter": DEMO_TIME_FILTER,
        "visualization_type": "line",
    }
    return dashboard, combined_rows, forecast_points


def _analysis_suggestions() -> list[dict[str, Any]]:
    return [
        {
            "title": "Forecast next month sales",
            "prompt": "Forecast next month sales and summarize the most likely trend.",
            "action_type": "forecast",
            "category": "forecasting",
            "goal": "Show forward-looking planning value",
            "rationale": "This pairs the historical trend with an executive-friendly forecast view.",
            "score": 10,
            "rank": 1,
        },
        {
            "title": "Explain regional performance gaps",
            "prompt": "Explain which region is underperforming and the most likely reasons.",
            "action_type": "analysis",
            "category": "root_cause",
            "goal": "Reveal where the follow-up story lives",
            "rationale": "Recruiters can immediately see how the system moves from KPI tracking to diagnosis.",
            "score": 9,
            "rank": 2,
        },
        {
            "title": "Review product mix",
            "prompt": "Show which product contributes the most revenue and how to protect growth.",
            "action_type": "analysis",
            "category": "product_mix",
            "goal": "Connect insight to action",
            "rationale": "The product distribution is balanced enough to make the recommendation story easy to follow.",
            "score": 8,
            "rank": 3,
        },
    ]


def _build_analysis_output(df: pd.DataFrame) -> dict[str, Any]:
    _, region_sales, product_sales = _base_demo_metrics(df)
    top_region = region_sales.iloc[0]
    top_product = product_sales.iloc[0]
    summary = (
        f"Sample revenue reached {int(df['sales'].sum()):,} across {len(df):,} orders. "
        f"{top_region['region']} leads all regions with {int(top_region['sales']):,} in sales, "
        f"while {top_product['product']} is the highest-contributing product line."
    )
    insights = [
        f"{top_region['region']} is the strongest market, contributing {round(float(top_region['sales']) / float(df['sales'].sum()) * 100, 1)}% of total revenue.",
        f"{top_product['product']} is the best-performing product family with {int(top_product['sales']):,} in sales.",
        "Daily sales trend is steady with upward momentum in the closing third of the sample period.",
    ]
    recommendations = [
        f"Prioritize demand capture in {top_region['region']} where the current sample shows the strongest momentum.",
        f"Feature {top_product['product']} in the demo narrative as the clearest revenue driver.",
        "Use the forecast view as the natural next step to turn historical insight into a planning story.",
    ]
    return {
        "summary": summary,
        "confidence": "high",
        "warnings": [],
        "dashboard": _build_analysis_dashboard(df),
        "suggestions": _analysis_suggestions(),
        "recommended_next_step": "Open the forecast output to extend the story from observation to planning.",
        "suggested_questions": [
            "Which region should we scale first?",
            "How concentrated is revenue in the top product?",
            "What should the next forecast window be?",
        ],
        "active_filter": DEMO_TIME_FILTER,
        "visualization_type": "line",
        "analysis_contract": {
            "intent": "dashboard",
            "code": "Precomputed public demo payload from the bundled sales sample.",
            "result_summary": summary,
            "insights": insights,
            "recommendations": recommendations,
            "confidence": "high",
            "warnings": [],
            "analysis_mode": "dashboard",
            "tool_used": "PYTHON",
            "active_filter": DEMO_TIME_FILTER,
            "visualization_type": "line",
        },
        "result": {"records": _records(region_sales.rename(columns={"sales": "regional_sales"}))},
    }


def _build_root_cause_output(df: pd.DataFrame) -> dict[str, Any]:
    daily_sales, region_sales, product_sales = _base_demo_metrics(df)
    lowest_region = region_sales.iloc[-1]
    highest_region = region_sales.iloc[0]
    top_product = product_sales.iloc[0]
    gap = int(highest_region["sales"] - lowest_region["sales"])
    root_cause_dashboard = {
        "charts": [
            {
                "type": "bar",
                "purpose": "regional_gap",
                "x": "region",
                "y": "sales",
                "title": "Regional performance gap",
                "rows": _records(region_sales),
            },
            {
                "type": "bar",
                "purpose": "product_mix",
                "x": "product",
                "y": "sales",
                "title": "Product contribution",
                "rows": _records(product_sales),
            },
            {
                "type": "line",
                "purpose": "trend",
                "x": "order_date",
                "y": "sales",
                "time_column": "order_date",
                "title": "Momentum before intervention",
                "rows": _records(daily_sales),
            },
        ],
        "filters": ["current_month", "last_month", "last_quarter", "last_year"],
        "kpis": [
            {"metric": "largest_region_gap", "value": gap},
            {"metric": "best_region", "value": str(highest_region["region"])},
            {"metric": "softest_region", "value": str(lowest_region["region"])},
            {"metric": "strongest_product", "value": str(top_product["product"])},
        ],
        "layout": _dashboard_layout(),
        "drilldown_ready": True,
        "time_column": "order_date",
        "active_filter": DEMO_TIME_FILTER,
        "visualization_type": "bar",
    }
    summary = (
        f"The clearest performance gap is regional: {lowest_region['region']} trails {highest_region['region']} "
        f"by {gap:,} in sales. Product mix is comparatively healthy, with {top_product['product']} giving the system "
        "a stable anchor for recommendations."
    )
    return {
        "summary": summary,
        "confidence": "high",
        "warnings": [],
        "dashboard": root_cause_dashboard,
        "suggestions": _analysis_suggestions(),
        "recommended_next_step": f"Use {top_product['product']} as the baseline offer and investigate why {lowest_region['region']} is lagging.",
        "suggested_questions": [
            f"Why is {lowest_region['region']} behind the rest of the market?",
            f"Can {top_product['product']} be used to lift {lowest_region['region']} performance?",
        ],
        "active_filter": DEMO_TIME_FILTER,
        "visualization_type": "bar",
        "analysis_contract": {
            "intent": "root_cause",
            "code": "Precomputed public demo payload from the bundled sales sample.",
            "result_summary": summary,
            "insights": [
                f"{lowest_region['region']} is the weakest region in the sample.",
                f"{top_product['product']} remains the most resilient product line.",
                "The gap story is easy to explain in a recruiter demo because it maps directly to an action plan.",
            ],
            "recommendations": [
                f"Audit channel execution in {lowest_region['region']}.",
                f"Package {top_product['product']} into the recovery plan.",
                "Use the forecast output to quantify the upside of correcting the lagging region.",
            ],
            "confidence": "high",
            "warnings": [],
            "analysis_mode": "dashboard",
            "tool_used": "PYTHON",
            "active_filter": DEMO_TIME_FILTER,
            "visualization_type": "bar",
        },
        "result": {"records": _records(region_sales)},
    }


def _build_forecast_output(df: pd.DataFrame) -> dict[str, Any]:
    dashboard, combined_rows, forecast_points = _build_forecast_dashboard(df)
    projected_total = round(sum(item["value"] for item in forecast_points), 2)
    summary = (
        f"Projected sales for the next 7 days total {projected_total:,.2f}. "
        "The short-term trend is rising, so the best demo narrative is to pair the forecast with regional demand planning."
    )
    return {
        "status": "PASSED",
        "summary": summary,
        "dashboard": dashboard,
        "context": {
            "dataset_type": "retail_sales",
            "domain": "retail",
            "is_time_series": True,
            "time_columns": ["order_date"],
            "primary_metrics": ["sales"],
            "categorical_features": ["region", "product", "customer"],
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "column_summary": [],
            "data_types": {column: str(dtype) for column, dtype in df.dtypes.items()},
            "missing_values": {column: int(df[column].isna().sum()) for column in df.columns},
        },
        "suggestions": _analysis_suggestions(),
        "recommended_next_step": "Use the root-cause view to explain which segment deserves the first operational response.",
        "suggested_questions": [
            "What is driving the projected increase?",
            "Which region should absorb extra inventory first?",
            "How should we describe forecast confidence in the demo?",
        ],
        "auto_config": {
            "date_column": "order_date",
            "target": "sales",
            "frequency": "Daily",
            "frequency_code": "D",
            "data_points": int(len(df)),
            "horizon": "next_month",
            "horizon_label": "Next month",
            "model_strategy": "hybrid",
            "training_mode": "local",
            "confidence": "medium",
            "confidence_score": 0.82,
            "date_confidence": {"label": "high", "score": 0.99},
            "target_confidence": {"label": "high", "score": 0.96},
        },
        "forecast_eligibility": {
            "allowed": True,
            "reason": None,
            "detected_time_column": "order_date",
            "suggestions": [],
        },
        "forecast_metadata": {
            "time_column": "order_date",
            "data_points": int(len(df)),
            "frequency": "Daily",
            "filled_missing_timestamps": 0,
        },
        "forecast": {"sales": forecast_points},
        "time_series": combined_rows,
        "confidence": {"score": 0.82, "label": "medium", "warnings": []},
        "recommendations": [
            {
                "category": "inventory",
                "priority": 1,
                "priority_label": "High",
                "title": "Stage inventory ahead of the rising trend",
                "recommended_action": "Increase readiness for the strongest regions first while monitoring Gamma demand.",
                "rationale": "The projection remains directionally positive across the next forecast window.",
                "impact_direction": "up",
                "expected_impact": "Protect revenue during the next cycle.",
                "confidence": "medium",
                "risk_level": "low",
                "decision_id": "demo-forecast-1",
            }
        ],
        "chosen_model": "trend_projection_demo",
        "trend_status": "rising",
        "history_points": int(len(df)),
        "resolved_frequency": "D",
        "active_filter": DEMO_TIME_FILTER,
        "visualization_type": "line",
    }


def _demo_stats(df: pd.DataFrame) -> list[dict[str, Any]]:
    _, region_sales, product_sales = _base_demo_metrics(df)
    return [
        {
            "label": "Sample rows",
            "value": int(len(df)),
            "detail": "Bundled retail sales dataset for zero-friction demos.",
        },
        {
            "label": "Revenue tracked",
            "value": int(df["sales"].sum()),
            "detail": "Preloaded KPI cards and charts.",
        },
        {
            "label": "Markets covered",
            "value": int(region_sales["region"].nunique()),
            "detail": "North, South, East, and West.",
        },
        {
            "label": "Products compared",
            "value": int(product_sales["product"].nunique()),
            "detail": "Alpha, Beta, and Gamma.",
        },
    ]


def _demo_flow() -> list[dict[str, Any]]:
    return [
        {
            "title": "Load sample dataset",
            "description": "Open a clean retail dataset instantly without signup, upload time, or external credentials.",
        },
        {
            "title": "Show analysis dashboard",
            "description": "Start with a recruiter-friendly KPI and chart view that explains the signal in seconds.",
        },
        {
            "title": "Pivot into forecast and root cause",
            "description": "Demonstrate how the product moves from observation to planning and diagnosis in one flow.",
        },
    ]


@lru_cache(maxsize=1)
def get_demo_payload() -> dict[str, Any]:
    df = _load_demo_dataframe()
    analysis_output = _build_analysis_output(df)
    forecast_output = _build_forecast_output(df)
    root_cause_output = _build_root_cause_output(df)

    outputs = [
        {
            "query": DEMO_QUERIES[0],
            "intent": "forecast",
            "output": forecast_output,
        },
        {
            "query": DEMO_QUERIES[1],
            "intent": "analysis",
            "output": analysis_output,
        },
        {
            "query": DEMO_QUERIES[2],
            "intent": "root_cause",
            "output": root_cause_output,
        },
    ]

    dataset = _demo_dataset_payload(df)
    return {
        "dataset": dataset,
        "datasets": [dataset],
        "queries": list(DEMO_QUERIES),
        "outputs": outputs,
        "dashboard": analysis_output["dashboard"],
        "stats": _demo_stats(df),
        "flow": _demo_flow(),
        "suggestions": _analysis_suggestions(),
    }
