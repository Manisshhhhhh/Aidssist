from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any

import duckdb
import pandas as pd


TYPE_KEYWORDS = {
    "sales": {
        "sales",
        "revenue",
        "orders",
        "order_id",
        "customer",
        "customers",
        "units",
        "quantity",
        "sku",
        "product",
        "region",
    },
    "finance": {
        "balance",
        "expense",
        "expenses",
        "ledger",
        "cashflow",
        "cash",
        "invoice",
        "payment",
        "profit",
        "loss",
        "margin",
        "budget",
        "gl",
    },
    "marketing": {
        "campaign",
        "channel",
        "ad",
        "ads",
        "impressions",
        "clicks",
        "ctr",
        "conversion",
        "conversions",
        "lead",
        "leads",
        "utm",
        "spend",
    },
}


def quote_ident(name: str) -> str:
    return '"' + str(name or "").replace('"', '""') + '"'


def _normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def _normalized_words(*values: str) -> set[str]:
    words: set[str] = set()
    for value in values:
        words.update(word for word in re.split(r"[^a-z0-9]+", str(value or "").lower()) if word)
    return words


def _name_similarity(left: str, right: str) -> float:
    left_token = _normalize_token(left)
    right_token = _normalize_token(right)
    if not left_token or not right_token:
        return 0.0
    if left_token == right_token:
        return 1.0
    if left_token.endswith(right_token) or right_token.endswith(left_token):
        return 0.9
    if left_token in right_token or right_token in left_token:
        return 0.75
    left_words = _normalized_words(left)
    right_words = _normalized_words(right)
    if not left_words or not right_words:
        return 0.0
    overlap = len(left_words & right_words)
    union = len(left_words | right_words)
    return overlap / union if union else 0.0


def _column_semantic_type(sql_type: str, sample: pd.Series) -> str:
    sql_type_upper = str(sql_type or "").upper()
    if any(token in sql_type_upper for token in ("BIGINT", "INT", "HUGEINT", "UBIGINT")):
        return "int"
    if any(token in sql_type_upper for token in ("DOUBLE", "FLOAT", "DECIMAL", "REAL", "NUMERIC")):
        return "float"
    if any(token in sql_type_upper for token in ("DATE", "TIMESTAMP", "TIME")):
        return "date"

    non_null = sample.dropna().astype(str)
    if not non_null.empty:
        parsed = pd.to_datetime(non_null.head(100), errors="coerce", format="mixed")
        if parsed.notna().mean() >= 0.75:
            return "date"
        distinct_ratio = non_null.nunique(dropna=True) / max(len(non_null), 1)
        if distinct_ratio <= 0.25 and non_null.nunique(dropna=True) <= max(50, math.sqrt(max(len(non_null), 1)) * 4):
            return "categorical"
    return "categorical" if sql_type_upper == "BOOLEAN" else "text"


def _classify_dataset_type(table_payloads: list[dict[str, Any]]) -> str:
    scores = defaultdict(float)
    for table in table_payloads:
        tokens = _normalized_words(table.get("name", ""), *(column.get("name", "") for column in table.get("columns", [])))
        for dataset_type, keywords in TYPE_KEYWORDS.items():
            overlap = tokens & keywords
            if overlap:
                scores[dataset_type] += float(len(overlap))
    if not scores:
        return "sales"
    return max(scores.items(), key=lambda item: item[1])[0]


def _choose_primary_keys(table_name: str, columns: list[dict[str, Any]]) -> list[str]:
    candidates: list[tuple[float, str]] = []
    for column in columns:
        name = str(column.get("name") or "")
        normalized = _normalize_token(name)
        null_ratio = float(column.get("null_ratio") or 0.0)
        unique_ratio = float(column.get("unique_ratio") or 0.0)
        semantic_type = str(column.get("semantic_type") or "")
        score = unique_ratio * 0.65 + (1.0 - min(null_ratio, 1.0)) * 0.25
        if semantic_type in {"int", "text"}:
            score += 0.05
        if normalized in {"id", f"{_normalize_token(table_name)}id"} or normalized.endswith("id"):
            score += 0.2
        if unique_ratio >= 0.98 and null_ratio <= 0.02:
            candidates.append((score, name))
    candidates.sort(reverse=True)
    if not candidates:
        return []
    best_score, best_column = candidates[0]
    return [best_column] if best_score >= 0.78 else []


def _distinct_non_null_values(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    column_name: str,
    limit: int = 5000,
) -> set[str]:
    query = (
        f"SELECT DISTINCT {quote_ident(column_name)} AS value FROM {quote_ident(table_name)} "
        f"WHERE {quote_ident(column_name)} IS NOT NULL LIMIT {int(limit)}"
    )
    frame = connection.execute(query).df()
    return {_normalize_token(str(value)) for value in frame["value"].tolist() if _normalize_token(str(value))}


def _infer_relationships(
    connection: duckdb.DuckDBPyConnection,
    tables: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    relationships: list[dict[str, Any]] = []
    primary_key_lookup = {
        table["name"]: str(table.get("primary_keys", [None])[0] or "")
        for table in tables
        if table.get("primary_keys")
    }

    for parent_table, parent_key in primary_key_lookup.items():
        if not parent_key:
            continue
        parent_values = _distinct_non_null_values(connection, parent_table, parent_key)
        if not parent_values:
            continue
        for child_table in tables:
            if child_table["name"] == parent_table:
                continue
            for column in child_table.get("columns", []):
                child_column = str(column.get("name") or "")
                if child_column == parent_key:
                    continue
                child_values = _distinct_non_null_values(connection, child_table["name"], child_column)
                if not child_values:
                    continue
                match_rate = len(child_values & parent_values) / max(len(child_values), 1)
                name_score = max(
                    _name_similarity(child_column, parent_key),
                    _name_similarity(child_column, parent_table),
                )
                confidence = match_rate * 0.75 + name_score * 0.25
                if match_rate < 0.7 or confidence < 0.72:
                    continue
                relationships.append(
                    {
                        "left_table": parent_table,
                        "left_column": parent_key,
                        "right_table": child_table["name"],
                        "right_column": child_column,
                        "relationship_type": "one_to_many",
                        "match_rate": round(match_rate, 4),
                        "confidence": round(confidence, 4),
                    }
                )
    relationships.sort(key=lambda item: (item["confidence"], item["match_rate"]), reverse=True)
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for item in relationships:
        key = (item["left_table"], item["left_column"], item["right_table"], item["right_column"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def build_schema_payload(
    connection: duckdb.DuckDBPyConnection,
    catalog: dict[str, Any],
    *,
    sample_rows: int = 5000,
) -> dict[str, Any]:
    table_payloads: list[dict[str, Any]] = []
    for table in catalog.get("tables", []):
        table_name = str(table.get("table_name") or "")
        describe = connection.execute(f"DESCRIBE {quote_ident(table_name)}").df()
        sample = connection.execute(
            f"SELECT * FROM {quote_ident(table_name)} LIMIT {int(sample_rows)}"
        ).df()
        row_count = int(connection.execute(f"SELECT COUNT(*) FROM {quote_ident(table_name)}").fetchone()[0])

        column_payloads: list[dict[str, Any]] = []
        for _, row in describe.iterrows():
            column_name = str(row["column_name"])
            column_type = str(row["column_type"])
            series = sample[column_name] if column_name in sample.columns else pd.Series(dtype="object")
            non_null_count = int(series.notna().sum()) if column_name in sample.columns else 0
            unique_count = int(series.nunique(dropna=True)) if column_name in sample.columns else 0
            sample_row_count = max(int(len(series)), 1)
            column_payloads.append(
                {
                    "name": column_name,
                    "sql_type": column_type,
                    "semantic_type": _column_semantic_type(column_type, series),
                    "nullable": bool(non_null_count < sample_row_count),
                    "non_null_count": non_null_count,
                    "missing_count": max(sample_row_count - non_null_count, 0),
                    "null_ratio": round(max(sample_row_count - non_null_count, 0) / sample_row_count, 4),
                    "unique_count": unique_count,
                    "unique_ratio": round(unique_count / sample_row_count, 4),
                    "sample_values": [
                        None if pd.isna(value) else (value.isoformat() if hasattr(value, "isoformat") else value)
                        for value in series.head(5).tolist()
                    ],
                }
            )

        primary_keys = _choose_primary_keys(table_name, column_payloads)
        table_payloads.append(
            {
                "name": table_name,
                "source_name": table.get("source_name"),
                "dataset_id": table.get("dataset_id"),
                "source_kind": table.get("source_kind"),
                "row_count": row_count,
                "column_count": len(column_payloads),
                "primary_keys": primary_keys,
                "columns": column_payloads,
                "preview_rows": _preview_rows(connection, table_name),
            }
        )

    relationships = _infer_relationships(connection, table_payloads)
    dataset_type = _classify_dataset_type(table_payloads)
    nodes = [
        {
            "id": table["name"],
            "label": table.get("source_name") or table["name"],
            "dataset_id": table.get("dataset_id"),
            "row_count": table["row_count"],
            "column_count": table["column_count"],
            "dataset_type": dataset_type,
        }
        for table in table_payloads
    ]
    edges = [
        {
            "id": f"{item['left_table']}:{item['left_column']}->{item['right_table']}:{item['right_column']}",
            "source": item["left_table"],
            "target": item["right_table"],
            "label": f"{item['left_column']} -> {item['right_column']}",
            "confidence": item["confidence"],
        }
        for item in relationships
    ]
    return {
        "dataset_type": dataset_type,
        "tables": table_payloads,
        "relationships": relationships,
        "graph": {"nodes": nodes, "edges": edges},
    }


def _preview_rows(connection: duckdb.DuckDBPyConnection, table_name: str) -> list[dict[str, Any]]:
    frame = connection.execute(f"SELECT * FROM {quote_ident(table_name)} LIMIT 8").df()
    if frame.empty:
        return []
    frame = frame.where(pd.notna(frame), None)
    return [
        {str(key): (value.isoformat() if hasattr(value, "isoformat") else value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]
