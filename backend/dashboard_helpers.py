from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from numbers import Number

import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype


MAX_MISSING_COLUMNS = 8
MAX_CATEGORICAL_UNIQUES = 12
MAX_CATEGORY_VALUES = 10
MAX_CHARTS_PER_SECTION = 2
MAX_SAMPLE_VALUES = 3
MAX_RESULT_CATEGORY_VALUES = 12
MIN_DATETIME_SUCCESS_RATIO = 0.6
BAR_HIGHLIGHT_FIELD = "__aidssist_highlight"
LINE_PEAK_FIELD = "__aidssist_peak"


@dataclass
class ChartSpec:
    kind: str
    title: str
    data: pd.DataFrame
    x: str
    y: str | None = None
    x_title: str | None = None
    y_title: str | None = None
    color: str | None = None


@dataclass
class ColumnInsight:
    name: str
    semantic_type: str
    dtype: str
    non_null_count: int
    missing_count: int
    missing_ratio: float
    unique_count: int
    sample_values: list[str]
    chart: ChartSpec | None


@dataclass
class ResultProfile:
    table: pd.DataFrame | None
    chart: ChartSpec | None
    metric_label: str | None = None
    metric_value: str | None = None
    text_value: str | None = None


@dataclass
class DatasetProfile:
    dataset_name: str
    dataset_key: str
    row_count: int
    column_count: int
    missing_cell_count: int
    duplicate_row_count: int
    numeric_column_count: int
    categorical_column_count: int
    datetime_column_count: int
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    datetime_columns: list[str] = field(default_factory=list)
    column_type_breakdown: pd.DataFrame = field(default_factory=pd.DataFrame)
    missing_by_column: pd.DataFrame = field(default_factory=pd.DataFrame)
    overview_charts: list[ChartSpec] = field(default_factory=list)
    numeric_charts: list[ChartSpec] = field(default_factory=list)
    categorical_charts: list[ChartSpec] = field(default_factory=list)
    datetime_charts: list[ChartSpec] = field(default_factory=list)
    data_dictionary: pd.DataFrame = field(default_factory=pd.DataFrame)
    content_chart: ChartSpec | None = None


def build_dataset_key(file_name: str, file_size: int, file_bytes: bytes) -> str:
    content_digest = hashlib.sha256(file_bytes).hexdigest()[:16]
    return f"{file_name}:{file_size}:{content_digest}"


def _coerce_datetime_series(series: pd.Series) -> pd.Series | None:
    non_null_series = series.dropna()
    if non_null_series.empty:
        return None

    if is_datetime64_any_dtype(series):
        parsed = pd.to_datetime(series, errors="coerce")
    elif is_bool_dtype(series):
        return None
    elif is_numeric_dtype(series):
        return None
    else:
        parsed = pd.to_datetime(
            series.astype("string"),
            errors="coerce",
            format="mixed",
        )

    parsed_non_null = parsed.dropna()
    if parsed_non_null.empty:
        return None

    success_ratio = len(parsed_non_null) / len(non_null_series)
    if success_ratio < MIN_DATETIME_SUCCESS_RATIO:
        return None

    return parsed


def infer_datetime_columns(df: pd.DataFrame) -> dict[str, pd.Series]:
    datetime_columns: dict[str, pd.Series] = {}

    for column in df.columns:
        column_name = str(column)
        parsed_series = _coerce_datetime_series(df[column])
        if parsed_series is not None:
            datetime_columns[column_name] = parsed_series

    return datetime_columns


def classify_columns(
    df: pd.DataFrame, datetime_columns: dict[str, pd.Series] | None = None
) -> tuple[list[str], list[str], list[str]]:
    datetime_columns = datetime_columns or {}
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    datetime_column_names: list[str] = []

    for column in df.columns:
        column_name = str(column)

        if column_name in datetime_columns:
            datetime_column_names.append(column_name)
        elif is_numeric_dtype(df[column]) and not is_bool_dtype(df[column]):
            numeric_columns.append(column_name)
        else:
            categorical_columns.append(column_name)

    return numeric_columns, categorical_columns, datetime_column_names


def build_column_type_breakdown(
    numeric_columns: list[str], categorical_columns: list[str], datetime_columns: list[str]
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "type": ["Numeric", "Categorical", "Datetime"],
            "count": [
                len(numeric_columns),
                len(categorical_columns),
                len(datetime_columns),
            ],
        }
    )


def build_missing_by_column(df: pd.DataFrame) -> pd.DataFrame:
    missing_series = df.isna().sum().sort_values(ascending=False)
    missing_series = missing_series[missing_series > 0].head(MAX_MISSING_COLUMNS)

    return missing_series.rename_axis("column").reset_index(name="missing_count")


def _clean_categorical_series(series: pd.Series) -> pd.Series:
    return (
        series.dropna()
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
    )


def _sample_values(series: pd.Series) -> list[str]:
    cleaned_values = []
    for value in series.dropna().tolist():
        text_value = str(value).strip()
        if not text_value or text_value in cleaned_values:
            continue
        cleaned_values.append(text_value)
        if len(cleaned_values) >= MAX_SAMPLE_VALUES:
            break

    return cleaned_values


def _annotate_top_bar(data: pd.DataFrame, value_column: str) -> pd.DataFrame:
    chart_data = data.copy()
    numeric_values = pd.to_numeric(chart_data[value_column], errors="coerce")
    if numeric_values.isna().all():
        return chart_data

    top_value = numeric_values.max()
    chart_data[BAR_HIGHLIGHT_FIELD] = [
        "Top segment" if value == top_value else "Other"
        for value in numeric_values
    ]
    return chart_data


def _annotate_peak_point(data: pd.DataFrame, value_column: str) -> pd.DataFrame:
    chart_data = data.copy()
    numeric_values = pd.to_numeric(chart_data[value_column], errors="coerce")
    if numeric_values.isna().all():
        return chart_data

    peak_value = numeric_values.max()
    chart_data[LINE_PEAK_FIELD] = [
        "Peak" if value == peak_value else "Series"
        for value in numeric_values
    ]
    return chart_data


def _build_datetime_count_chart(column_name: str, datetime_series: pd.Series) -> ChartSpec | None:
    grouped_counts = (
        datetime_series.dropna()
        .dt.floor("D")
        .value_counts()
        .sort_index()
        .rename_axis(column_name)
        .reset_index(name="Rows")
    )

    if len(grouped_counts) < 2:
        return None

    grouped_counts = _annotate_peak_point(grouped_counts, "Rows")
    return ChartSpec(
        kind="line",
        title=f"Records over time by {column_name}",
        data=grouped_counts,
        x=column_name,
        y="Rows",
        x_title=column_name,
        y_title="Rows",
    )


def _build_numeric_over_time_chart(
    df: pd.DataFrame,
    datetime_column_name: str,
    datetime_series: pd.Series,
    numeric_column_name: str,
) -> ChartSpec | None:
    numeric_series = pd.to_numeric(df[numeric_column_name], errors="coerce")
    chart_data = pd.DataFrame(
        {
            datetime_column_name: datetime_series,
            numeric_column_name: numeric_series,
        }
    ).dropna()

    if chart_data.empty:
        return None

    grouped_data = (
        chart_data.assign(
            **{datetime_column_name: chart_data[datetime_column_name].dt.floor("D")}
        )
        .groupby(datetime_column_name, dropna=False)[numeric_column_name]
        .sum()
        .reset_index()
    )

    if len(grouped_data) < 2:
        return None

    grouped_data = _annotate_peak_point(grouped_data, numeric_column_name)
    return ChartSpec(
        kind="line",
        title=f"{numeric_column_name} over time by {datetime_column_name}",
        data=grouped_data,
        x=datetime_column_name,
        y=numeric_column_name,
        x_title=datetime_column_name,
        y_title=numeric_column_name,
    )


def _rank_numeric_columns(df: pd.DataFrame, numeric_columns: list[str]) -> list[str]:
    ranked_columns = []

    for index, column_name in enumerate(numeric_columns):
        non_null_count = int(df[column_name].notna().sum())
        if non_null_count > 0:
            ranked_columns.append((non_null_count, -index, column_name))

    ranked_columns.sort(reverse=True)
    return [column_name for _, _, column_name in ranked_columns]


def _build_numeric_histogram_chart(df: pd.DataFrame, column_name: str) -> ChartSpec | None:
    numeric_data = pd.to_numeric(df[column_name], errors="coerce").dropna()
    if numeric_data.empty:
        return None

    return ChartSpec(
        kind="histogram",
        title=f"Distribution of {column_name}",
        data=pd.DataFrame({column_name: numeric_data}),
        x=column_name,
        x_title=column_name,
        y_title="Rows",
    )


def _build_categorical_count_chart(df: pd.DataFrame, column_name: str) -> ChartSpec | None:
    categorical_series = _clean_categorical_series(df[column_name])
    unique_count = categorical_series.nunique(dropna=True)

    if unique_count == 0:
        return None

    category_counts = (
        categorical_series.value_counts()
        .head(MAX_CATEGORY_VALUES)
        .rename_axis(column_name)
        .reset_index(name="Rows")
    )

    if category_counts.empty:
        return None

    category_counts = _annotate_top_bar(category_counts, "Rows")
    return ChartSpec(
        kind="bar",
        title=f"Top values in {column_name}",
        data=category_counts,
        x=column_name,
        y="Rows",
        x_title=column_name,
        y_title="Rows",
    )


def build_datetime_charts(
    df: pd.DataFrame,
    datetime_columns: dict[str, pd.Series],
    numeric_columns: list[str],
) -> list[ChartSpec]:
    charts: list[ChartSpec] = []
    ranked_numeric_columns = _rank_numeric_columns(df, numeric_columns)

    for datetime_column_name, datetime_series in datetime_columns.items():
        count_chart = _build_datetime_count_chart(datetime_column_name, datetime_series)
        if count_chart is not None:
            charts.append(count_chart)

        if ranked_numeric_columns:
            numeric_chart = _build_numeric_over_time_chart(
                df,
                datetime_column_name,
                datetime_series,
                ranked_numeric_columns[0],
            )
            if numeric_chart is not None:
                charts.append(numeric_chart)

        if charts:
            break

    return charts[:MAX_CHARTS_PER_SECTION]


def build_categorical_charts(df: pd.DataFrame, categorical_columns: list[str]) -> list[ChartSpec]:
    charts: list[ChartSpec] = []

    for column_name in categorical_columns:
        chart = _build_categorical_count_chart(df, column_name)
        if chart is not None:
            charts.append(chart)

        if len(charts) >= MAX_CHARTS_PER_SECTION:
            break

    return charts


def build_numeric_charts(df: pd.DataFrame, numeric_columns: list[str]) -> list[ChartSpec]:
    charts: list[ChartSpec] = []

    for column_name in _rank_numeric_columns(df, numeric_columns):
        chart = _build_numeric_histogram_chart(df, column_name)
        if chart is not None:
            charts.append(chart)

        if len(charts) >= MAX_CHARTS_PER_SECTION:
            break

    return charts


def build_overview_charts(
    datetime_charts: list[ChartSpec],
    categorical_charts: list[ChartSpec],
    numeric_charts: list[ChartSpec],
) -> list[ChartSpec]:
    overview_charts: list[ChartSpec] = []

    for chart_group in (datetime_charts, categorical_charts, numeric_charts):
        if chart_group:
            overview_charts.append(chart_group[0])

    return overview_charts[:MAX_CHARTS_PER_SECTION]


def build_data_dictionary(
    df: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
    datetime_columns: list[str],
) -> pd.DataFrame:
    column_type_map = {}
    column_type_map.update({column_name: "Numeric" for column_name in numeric_columns})
    column_type_map.update({column_name: "Categorical" for column_name in categorical_columns})
    column_type_map.update({column_name: "Datetime" for column_name in datetime_columns})

    rows = []
    for column in df.columns:
        column_name = str(column)
        series = df[column]
        non_null_count = int(series.notna().sum())
        missing_count = int(series.isna().sum())
        rows.append(
            {
                "column": column_name,
                "semantic_type": column_type_map.get(column_name, "Categorical"),
                "dtype": str(series.dtype),
                "non_null_count": non_null_count,
                "missing_count": missing_count,
                "missing_pct": round((missing_count / len(df)) * 100, 2) if len(df) else 0.0,
                "unique_count": int(series.nunique(dropna=True)),
                "sample_values": ", ".join(_sample_values(series)) or "N/A",
            }
        )

    return pd.DataFrame(rows)


def select_content_chart(
    df: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
    datetime_columns: dict[str, pd.Series],
) -> ChartSpec | None:
    overview_charts = build_overview_charts(
        build_datetime_charts(df, datetime_columns, numeric_columns),
        build_categorical_charts(df, categorical_columns),
        build_numeric_charts(df, numeric_columns),
    )
    return overview_charts[0] if overview_charts else None


def build_column_insight(
    df: pd.DataFrame,
    column_name: str,
    datetime_columns: dict[str, pd.Series] | None = None,
) -> ColumnInsight:
    datetime_columns = datetime_columns or infer_datetime_columns(df)
    numeric_columns, categorical_columns, datetime_column_names = classify_columns(df, datetime_columns)
    data_dictionary = build_data_dictionary(
        df,
        numeric_columns,
        categorical_columns,
        datetime_column_names,
    )

    column_name = str(column_name)
    column_row = data_dictionary.loc[data_dictionary["column"] == column_name]
    if column_row.empty:
        raise KeyError(f"Column '{column_name}' not found")

    series = df[column_name]
    semantic_type = column_row.iloc[0]["semantic_type"]
    chart: ChartSpec | None = None

    if semantic_type == "Datetime":
        chart = _build_datetime_count_chart(column_name, datetime_columns[column_name])
    elif semantic_type == "Numeric":
        chart = _build_numeric_histogram_chart(df, column_name)
    else:
        chart = _build_categorical_count_chart(df, column_name)

    return ColumnInsight(
        name=column_name,
        semantic_type=semantic_type,
        dtype=str(series.dtype),
        non_null_count=int(series.notna().sum()),
        missing_count=int(series.isna().sum()),
        missing_ratio=(float(series.isna().sum()) / len(df)) if len(df) else 0.0,
        unique_count=int(series.nunique(dropna=True)),
        sample_values=_sample_values(series),
        chart=chart,
    )


def _has_default_range_index(index: pd.Index) -> bool:
    return isinstance(index, pd.RangeIndex) and index.start == 0 and index.step == 1


def coerce_result_to_table(result) -> pd.DataFrame | None:
    if isinstance(result, pd.DataFrame):
        table = result.copy()
        if not _has_default_range_index(table.index) or any(name is not None for name in table.index.names):
            table = table.reset_index()
        table.columns = [str(column) for column in table.columns]
        return table

    if isinstance(result, pd.Series):
        series_name = str(result.name) if result.name is not None else "value"
        index_name = str(result.index.name) if result.index.name is not None else "index"
        table = result.rename(series_name).rename_axis(index_name).reset_index()
        table.columns = [str(column) for column in table.columns]
        return table

    if isinstance(result, dict):
        if not result:
            return pd.DataFrame(columns=["key", "value"])

        scalar_values_only = all(
            not isinstance(value, (dict, list, tuple, set, pd.DataFrame, pd.Series))
            for value in result.values()
        )
        if scalar_values_only:
            return pd.DataFrame(
                {
                    "key": [str(key) for key in result.keys()],
                    "value": list(result.values()),
                }
            )

        try:
            return pd.DataFrame(result)
        except Exception:
            return None

    if isinstance(result, list):
        if not result:
            return pd.DataFrame()

        if all(isinstance(item, dict) for item in result):
            return pd.DataFrame(result)

        if all(not isinstance(item, (dict, list, tuple, set, pd.DataFrame, pd.Series)) for item in result):
            return pd.DataFrame({"value": result})

    return None


def _format_metric_value(value) -> str:
    if pd.isna(value):
        return "N/A"

    if isinstance(value, Number) and not isinstance(value, bool):
        numeric_value = float(value)
        if numeric_value.is_integer():
            return f"{int(numeric_value):,}"
        return f"{numeric_value:,.2f}"

    return str(value)


def _format_chart_axis_value(value) -> str:
    if isinstance(value, pd.Timestamp):
        return value.strftime("%b %d, %Y")
    return _format_metric_value(value)


def build_chart_takeaway(chart_spec: ChartSpec | None) -> str | None:
    if chart_spec is None:
        return None

    if chart_spec.kind == "bar" and chart_spec.y:
        if chart_spec.x not in chart_spec.data.columns or chart_spec.y not in chart_spec.data.columns:
            return None

        chart_data = chart_spec.data[[chart_spec.x, chart_spec.y]].copy()
        chart_data[chart_spec.y] = pd.to_numeric(chart_data[chart_spec.y], errors="coerce")
        chart_data = chart_data.dropna(subset=[chart_spec.y])
        if chart_data.empty:
            return None

        top_row = chart_data.sort_values(chart_spec.y, ascending=False).iloc[0]
        total_value = float(chart_data[chart_spec.y].sum())
        share_text = ""
        if total_value > 0:
            share_pct = (float(top_row[chart_spec.y]) / total_value) * 100
            share_text = f", contributing {share_pct:.1f}% of the displayed total"

        return (
            f"{top_row[chart_spec.x]} leads with {_format_metric_value(top_row[chart_spec.y])} "
            f"{chart_spec.y_title or chart_spec.y}{share_text}."
        )

    if chart_spec.kind == "line" and chart_spec.y:
        if chart_spec.x not in chart_spec.data.columns or chart_spec.y not in chart_spec.data.columns:
            return None

        chart_data = chart_spec.data[[chart_spec.x, chart_spec.y]].copy()
        chart_data[chart_spec.x] = pd.to_datetime(chart_data[chart_spec.x], errors="coerce")
        chart_data[chart_spec.y] = pd.to_numeric(chart_data[chart_spec.y], errors="coerce")
        chart_data = chart_data.dropna(subset=[chart_spec.x, chart_spec.y]).sort_values(chart_spec.x)
        if chart_data.empty:
            return None

        peak_row = chart_data.loc[chart_data[chart_spec.y].idxmax()]
        start_value = float(chart_data.iloc[0][chart_spec.y])
        end_value = float(chart_data.iloc[-1][chart_spec.y])
        trend_delta = end_value - start_value
        if abs(trend_delta) < 1e-9:
            trend_text = "and finishes flat versus the starting point"
        elif trend_delta > 0:
            trend_text = "and finishes above the starting point"
        else:
            trend_text = "and finishes below the starting point"

        return (
            f"{chart_spec.y_title or chart_spec.y} reach a peak on "
            f"{_format_chart_axis_value(peak_row[chart_spec.x])} at "
            f"{_format_metric_value(peak_row[chart_spec.y])} {trend_text}."
        )

    if chart_spec.kind == "histogram":
        if chart_spec.x not in chart_spec.data.columns:
            return None

        numeric_series = pd.to_numeric(chart_spec.data[chart_spec.x], errors="coerce").dropna()
        if numeric_series.empty:
            return None
        if len(numeric_series) == 1:
            return (
                f"{chart_spec.x_title or chart_spec.x} currently has a single observed value of "
                f"{_format_metric_value(numeric_series.iloc[0])}."
            )

        median_value = numeric_series.median()
        lower_quartile = numeric_series.quantile(0.25)
        upper_quartile = numeric_series.quantile(0.75)
        return (
            f"{chart_spec.x_title or chart_spec.x} centers around {_format_metric_value(median_value)}, "
            f"with the middle 50% of values falling between "
            f"{_format_metric_value(lower_quartile)} and {_format_metric_value(upper_quartile)}."
        )

    return None


def _build_result_line_chart(
    table: pd.DataFrame,
    datetime_column_name: str,
    numeric_column_name: str,
    datetime_columns: dict[str, pd.Series],
) -> ChartSpec | None:
    chart_data = pd.DataFrame(
        {
            datetime_column_name: datetime_columns[datetime_column_name],
            numeric_column_name: pd.to_numeric(table[numeric_column_name], errors="coerce"),
        }
    ).dropna()

    if chart_data.empty:
        return None

    grouped_data = (
        chart_data.assign(
            **{datetime_column_name: chart_data[datetime_column_name].dt.floor("D")}
        )
        .groupby(datetime_column_name, dropna=False)[numeric_column_name]
        .sum()
        .reset_index()
    )

    if len(grouped_data) < 2:
        return None

    return ChartSpec(
        kind="line",
        title=f"{numeric_column_name} over time",
        data=grouped_data,
        x=datetime_column_name,
        y=numeric_column_name,
        x_title=datetime_column_name,
        y_title=numeric_column_name,
    )


def _build_result_bar_chart(table: pd.DataFrame, categorical_column_name: str, numeric_column_name: str) -> ChartSpec | None:
    chart_data = table[[categorical_column_name, numeric_column_name]].copy()
    chart_data[categorical_column_name] = (
        chart_data[categorical_column_name]
        .astype("string")
        .fillna("Missing")
        .str.strip()
        .replace("", "Missing")
    )
    chart_data[numeric_column_name] = pd.to_numeric(chart_data[numeric_column_name], errors="coerce")
    chart_data = chart_data.dropna(subset=[numeric_column_name])

    if chart_data.empty:
        return None

    grouped_data = (
        chart_data.groupby(categorical_column_name, dropna=False)[numeric_column_name]
        .sum()
        .reset_index()
        .sort_values(numeric_column_name, ascending=False)
        .head(MAX_RESULT_CATEGORY_VALUES)
    )

    if grouped_data.empty:
        return None

    grouped_data = _annotate_top_bar(grouped_data, numeric_column_name)
    return ChartSpec(
        kind="bar",
        title=f"{numeric_column_name} by {categorical_column_name}",
        data=grouped_data,
        x=categorical_column_name,
        y=numeric_column_name,
        x_title=categorical_column_name,
        y_title=numeric_column_name,
    )


def infer_result_chart(table: pd.DataFrame) -> ChartSpec | None:
    if table.empty:
        return None

    datetime_columns = infer_datetime_columns(table)
    numeric_columns, categorical_columns, datetime_column_names = classify_columns(
        table,
        datetime_columns,
    )

    if datetime_column_names and numeric_columns:
        line_chart = _build_result_line_chart(
            table,
            datetime_column_names[0],
            numeric_columns[0],
            datetime_columns,
        )
        if line_chart is not None:
            return line_chart

    if categorical_columns and numeric_columns:
        bar_chart = _build_result_bar_chart(
            table,
            categorical_columns[0],
            numeric_columns[0],
        )
        if bar_chart is not None:
            return bar_chart

    if len(numeric_columns) == 1:
        return _build_numeric_histogram_chart(table, numeric_columns[0])

    return None


def profile_analysis_result(result) -> ResultProfile:
    if hasattr(result, "figure") or hasattr(result, "savefig"):
        return ResultProfile(table=None, chart=None)

    if isinstance(result, Number) and not isinstance(result, bool):
        return ResultProfile(
            table=None,
            chart=None,
            metric_label="Analysis result",
            metric_value=_format_metric_value(result),
            text_value=str(result),
        )

    if isinstance(result, str):
        return ResultProfile(table=None, chart=None, text_value=result)

    table = coerce_result_to_table(result)
    if table is None:
        return ResultProfile(table=None, chart=None, text_value=str(result))

    metric_label = None
    metric_value = None
    if table.shape == (1, 1):
        metric_label = str(table.columns[0])
        metric_value = _format_metric_value(table.iloc[0, 0])

    return ResultProfile(
        table=table,
        chart=infer_result_chart(table),
        metric_label=metric_label,
        metric_value=metric_value,
    )


def profile_dataset(
    df: pd.DataFrame, dataset_name: str, dataset_key: str | None = None
) -> DatasetProfile:
    datetime_columns = infer_datetime_columns(df)
    numeric_columns, categorical_columns, datetime_column_names = classify_columns(
        df,
        datetime_columns,
    )
    datetime_charts = build_datetime_charts(df, datetime_columns, numeric_columns)
    categorical_charts = build_categorical_charts(df, categorical_columns)
    numeric_charts = build_numeric_charts(df, numeric_columns)
    overview_charts = build_overview_charts(
        datetime_charts,
        categorical_charts,
        numeric_charts,
    )

    return DatasetProfile(
        dataset_name=dataset_name,
        dataset_key=dataset_key or dataset_name,
        row_count=len(df),
        column_count=len(df.columns),
        missing_cell_count=int(df.isna().sum().sum()),
        duplicate_row_count=int(df.duplicated().sum()),
        numeric_column_count=len(numeric_columns),
        categorical_column_count=len(categorical_columns),
        datetime_column_count=len(datetime_column_names),
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        datetime_columns=datetime_column_names,
        column_type_breakdown=build_column_type_breakdown(
            numeric_columns,
            categorical_columns,
            datetime_column_names,
        ),
        missing_by_column=build_missing_by_column(df),
        overview_charts=overview_charts,
        numeric_charts=numeric_charts,
        categorical_charts=categorical_charts,
        datetime_charts=datetime_charts,
        data_dictionary=build_data_dictionary(
            df,
            numeric_columns,
            categorical_columns,
            datetime_column_names,
        ),
        content_chart=overview_charts[0] if overview_charts else None,
    )
