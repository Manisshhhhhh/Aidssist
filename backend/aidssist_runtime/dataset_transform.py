from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any

import pandas as pd

from .ingestion import load_dataset_dataframe
from .storage import build_object_key, get_object_store
from backend import prompt_pipeline
from backend.data_sources import build_dataframe_fingerprint, build_dataset_key
from backend.workflow_store import WorkflowStore


def _match_columns(raw_text: str, columns: list[str]) -> list[str]:
    requested = [item.strip().strip("'\"") for item in raw_text.split(",") if item.strip()]
    normalized_lookup = {re.sub(r"[^a-z0-9]+", "", column.lower()): column for column in columns}
    matches: list[str] = []
    for item in requested:
        token = re.sub(r"[^a-z0-9]+", "", item.lower())
        if token in normalized_lookup:
            matches.append(normalized_lookup[token])
    return matches


def _run_deterministic_transform(df: pd.DataFrame, instruction: str) -> pd.DataFrame | None:
    normalized = str(instruction or "").strip().lower()
    if not normalized:
        return df.copy()

    if "drop duplicate" in normalized:
        return df.drop_duplicates().reset_index(drop=True)

    if "fill numeric" in normalized and "mean" in normalized:
        transformed = df.copy()
        numeric_columns = transformed.select_dtypes(include=["number"]).columns
        for column in numeric_columns:
            transformed[column] = transformed[column].fillna(transformed[column].mean())
        return transformed

    rename_match = re.search(r"rename\s+(.+?)\s+to\s+(.+)", normalized)
    if rename_match:
        source_match = _match_columns(rename_match.group(1), [str(column) for column in df.columns])
        target_name = rename_match.group(2).strip().strip("'\"")
        if source_match and target_name:
            return df.rename(columns={source_match[0]: target_name})

    keep_match = re.search(r"(?:keep|select)\s+columns?\s+(.+)", normalized)
    if keep_match:
        selected = _match_columns(keep_match.group(1), [str(column) for column in df.columns])
        if selected:
            return df[selected].copy()

    return None


def _generate_transform_code(df: pd.DataFrame, instruction: str) -> str:
    prompt = "\n".join(
        (
            "Transform the pandas dataframe named `df` according to the instruction below.",
            "Return only Python code.",
            "Use pandas and numpy only.",
            "The final transformed dataframe must be assigned to `result`.",
            "Raise ValueError if the requested transformation is unsafe or impossible.",
            "",
            f"Instruction: {instruction}",
            f"Columns: {list(df.columns)}",
            f"Dtypes: {df.dtypes.astype(str).to_dict()}",
            "Preview:",
            df.head(8).to_string(),
        )
    )
    return prompt_pipeline._generate_groq_content(  # type: ignore[attr-defined]
        model=prompt_pipeline.DEFAULT_GEMINI_MODEL,
        messages=[{"role": "user", "content": prompt}],
    ).strip()


def transform_dataset(
    *,
    store: WorkflowStore,
    dataset_id: str,
    instruction: str,
) -> dict[str, Any]:
    dataset = store.get_dataset(dataset_id)
    if dataset is None:
        raise ValueError(f"Dataset '{dataset_id}' was not found.")

    workspace = store.get_workspace_by_dataset_id(dataset_id)
    if workspace is None:
        raise ValueError("This dataset is not linked to a solver workspace yet.")

    asset = store.get_asset_by_dataset_id(dataset_id)
    df = load_dataset_dataframe(dataset)
    transformed_df = _run_deterministic_transform(df, instruction)

    transform_code = None
    if transformed_df is None:
        transform_code = _generate_transform_code(df, instruction)
        execution = prompt_pipeline._execute_with_fix_details(  # type: ignore[attr-defined]
            instruction,
            df,
            transform_code,
            max_retries=2,
            model=prompt_pipeline.DEFAULT_GEMINI_MODEL,
        )
        transformed_df = execution["result"]
        if execution["error"]:
            raise ValueError(str(execution["error"]))
        if not isinstance(transformed_df, pd.DataFrame):
            raise ValueError("The transform did not produce a dataframe result.")

    rendered_csv = transformed_df.to_csv(index=False).encode("utf-8")
    derived_name = f"{Path(dataset.dataset_name).stem}_derived.csv"
    source_fingerprint = build_dataframe_fingerprint(transformed_df)
    object_key = build_object_key("derived_datasets", workspace.workspace_id, derived_name)
    get_object_store().put_bytes(object_key, rendered_csv, content_type="text/csv")

    derived = store.create_derived_dataset(
        workspace_id=workspace.workspace_id,
        asset_id=asset.asset_id if asset else None,
        parent_dataset_id=dataset.dataset_id,
        dataset_name=derived_name,
        dataset_key=build_dataset_key(derived_name, source_fingerprint),
        source_fingerprint=source_fingerprint,
        object_key=object_key,
        content_type="text/csv",
        transform_prompt=instruction,
        row_count=int(len(transformed_df)),
        column_count=int(len(transformed_df.columns)),
        preview_columns=[str(column) for column in transformed_df.columns.tolist()],
        preview_rows=transformed_df.head(12).where(pd.notna(transformed_df.head(12)), None).to_dict(orient="records"),
    )
    return {
        "derived_dataset": derived,
        "transform_code": transform_code,
    }
