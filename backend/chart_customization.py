from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pandas.api.types import is_numeric_dtype

from backend.dashboard_helpers import ChartSpec, infer_datetime_columns


PALETTE_COLORS = {
    "orange": "#F97316",
    "blue": "#38BDF8",
    "green": "#22C55E",
}


@dataclass
class ChartCustomization:
    kind: str = "auto"
    x_column: str | None = None
    y_column: str | None = None
    aggregation: str = "sum"
    palette: str = "orange"
    title: str = ""
    x_title: str = ""
    y_title: str = ""


def get_palette_color(palette: str | None) -> str:
    return PALETTE_COLORS.get(str(palette or "").lower(), PALETTE_COLORS["orange"])


def get_chart_customization_options(table: pd.DataFrame) -> dict:
    datetime_columns = infer_datetime_columns(table)
    numeric_columns = [
        str(column_name)
        for column_name in table.columns
        if is_numeric_dtype(table[column_name])
    ]
    categorical_columns = [str(column_name) for column_name in table.columns if str(column_name) not in numeric_columns]
    return {
        "columns": [str(column_name) for column_name in table.columns],
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": list(datetime_columns.keys()),
        "kinds": ["auto", "bar", "line", "histogram", "table"],
        "aggregations": ["sum", "mean", "count"],
        "palettes": list(PALETTE_COLORS.keys()),
    }


def _aggregate_chart_data(
    table: pd.DataFrame,
    *,
    x_column: str,
    y_column: str | None,
    aggregation: str,
) -> pd.DataFrame:
    if aggregation == "count" or not y_column:
        grouped = table.groupby(x_column, dropna=False).size().reset_index(name="Rows")
        return grouped

    numeric_values = pd.to_numeric(table[y_column], errors="coerce")
    chart_data = pd.DataFrame({x_column: table[x_column], y_column: numeric_values}).dropna()
    if chart_data.empty:
        return chart_data

    grouped = (
        chart_data.groupby(x_column, dropna=False)[y_column]
        .mean()
        .reset_index()
        if aggregation == "mean"
        else chart_data.groupby(x_column, dropna=False)[y_column].sum().reset_index()
    )
    return grouped


def build_custom_result_chart(table: pd.DataFrame, customization: ChartCustomization) -> ChartSpec | None:
    if table.empty or customization.kind in {"auto", "table"}:
        return None

    palette_color = get_palette_color(customization.palette)
    x_column = customization.x_column or (str(table.columns[0]) if len(table.columns) else None)
    y_column = customization.y_column or None
    if not x_column or x_column not in table.columns:
        return None

    title = customization.title or ""
    x_title = customization.x_title or x_column
    y_title = customization.y_title or (y_column or "Rows")

    if customization.kind == "histogram":
        if not y_column:
            y_column = x_column
        if y_column not in table.columns or not is_numeric_dtype(table[y_column]):
            return None
        chart_data = pd.DataFrame({y_column: pd.to_numeric(table[y_column], errors="coerce")}).dropna()
        if chart_data.empty:
            return None
        return ChartSpec(
            kind="histogram",
            title=title or f"Distribution of {y_column}",
            data=chart_data,
            x=y_column,
            x_title=x_title if customization.x_title else y_column,
            y_title=y_title if customization.y_title else "Rows",
            color=palette_color,
        )

    chart_data = _aggregate_chart_data(
        table,
        x_column=x_column,
        y_column=y_column,
        aggregation=customization.aggregation,
    )
    if chart_data.empty:
        return None

    effective_y = "Rows" if customization.aggregation == "count" or not y_column else y_column
    return ChartSpec(
        kind=customization.kind,
        title=title or f"{effective_y} by {x_column}",
        data=chart_data,
        x=x_column,
        y=effective_y,
        x_title=x_title,
        y_title=y_title,
        color=palette_color,
    )
