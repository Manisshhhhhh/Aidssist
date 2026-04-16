from __future__ import annotations

from numbers import Number

import pandas as pd


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, Number) and hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def serialize_result(result):
    if isinstance(result, pd.DataFrame):
        return {
            "kind": "dataframe",
            "columns": [str(column) for column in result.columns],
            "records": [_jsonable(record) for record in result.to_dict(orient="records")],
        }

    if isinstance(result, pd.Series):
        return {
            "kind": "series",
            "name": str(result.name or "value"),
            "index": [_jsonable(index) for index in result.index.tolist()],
            "values": [_jsonable(value) for value in result.tolist()],
        }

    if isinstance(result, (list, dict)):
        return {"kind": "json", "value": _jsonable(result)}

    if isinstance(result, Number) and not isinstance(result, bool):
        return {"kind": "scalar", "value": _jsonable(result)}

    return {"kind": "text", "value": str(result)}


def deserialize_result(payload):
    if payload is None:
        return None

    kind = str(payload.get("kind", "text"))
    if kind == "dataframe":
        return pd.DataFrame(payload.get("records", []), columns=payload.get("columns"))
    if kind == "series":
        return pd.Series(payload.get("values", []), index=payload.get("index", []), name=payload.get("name"))
    if kind in {"json", "scalar", "text"}:
        return payload.get("value")
    return payload


def serialize_analysis_output(output: dict) -> dict:
    serialized = _jsonable(dict(output))
    serialized["result"] = serialize_result(output.get("result"))
    return serialized


def deserialize_analysis_output(payload: dict) -> dict:
    deserialized = dict(payload)
    deserialized["result"] = deserialize_result(payload.get("result"))
    return deserialized
