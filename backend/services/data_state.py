from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Callable

import pandas as pd


TransformationCallable = Callable[[pd.DataFrame], pd.DataFrame]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _estimate_cost_score(
    rows_before: int,
    columns_before: int,
    rows_after: int,
    columns_after: int,
    duration_ms: float,
) -> float:
    baseline_cells = max(1, rows_before * max(columns_before, 1))
    shape_delta = abs(rows_after - rows_before) + (abs(columns_after - columns_before) * max(rows_before, 1))
    normalized_shape_delta = shape_delta / baseline_cells
    runtime_factor = duration_ms / 250.0
    size_factor = baseline_cells / 100_000.0
    return round(size_factor + normalized_shape_delta * 8.0 + runtime_factor, 3)


def _cost_label(cost_score: float) -> str:
    if cost_score >= 5.0:
        return "high"
    if cost_score >= 2.0:
        return "medium"
    return "low"


@dataclass(slots=True)
class TransformationRecord:
    transformation_id: str
    dataset_id: str
    operation: str
    summary: str
    parameters: dict[str, Any] = field(default_factory=dict)
    rows_before: int = 0
    rows_after: int = 0
    columns_before: int = 0
    columns_after: int = 0
    duration_ms: float = 0.0
    cost_score: float = 0.0
    cost_label: str = "low"
    created_at: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "transformation_id": self.transformation_id,
            "dataset_id": self.dataset_id,
            "operation": self.operation,
            "summary": self.summary,
            "parameters": dict(self.parameters),
            "rows_before": self.rows_before,
            "rows_after": self.rows_after,
            "columns_before": self.columns_before,
            "columns_after": self.columns_after,
            "duration_ms": round(self.duration_ms, 2),
            "cost_score": self.cost_score,
            "cost_label": self.cost_label,
            "created_at": self.created_at,
        }


@dataclass(slots=True)
class ReactiveDatasetState:
    dataset_id: str
    dataset_name: str
    original: pd.DataFrame
    current: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)
    undo_stack: list[pd.DataFrame] = field(default_factory=list)
    redo_stack: list[pd.DataFrame] = field(default_factory=list)
    history: list[TransformationRecord] = field(default_factory=list)
    created_at: str = field(default_factory=_utc_now)

    @classmethod
    def from_dataframe(
        cls,
        dataset_id: str,
        dataset_name: str,
        dataframe: pd.DataFrame,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> "ReactiveDatasetState":
        normalized = dataframe.copy(deep=True)
        normalized.columns = [str(column) for column in normalized.columns]
        return cls(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            original=normalized.copy(deep=True),
            current=normalized.copy(deep=True),
            metadata=dict(metadata or {}),
        )

    @property
    def can_undo(self) -> bool:
        return bool(self.undo_stack)

    @property
    def can_redo(self) -> bool:
        return bool(self.redo_stack)

    def apply_transformation(
        self,
        operation: str,
        transformer: TransformationCallable,
        *,
        parameters: dict[str, Any] | None = None,
        summary: str | None = None,
    ) -> TransformationRecord:
        base_frame = self.current.copy(deep=True)
        started = perf_counter()
        transformed = transformer(base_frame.copy(deep=True))
        duration_ms = (perf_counter() - started) * 1000.0

        if not isinstance(transformed, pd.DataFrame):
            raise TypeError("Transformers must return a pandas DataFrame.")

        transformed = transformed.copy(deep=True)
        transformed.columns = [str(column) for column in transformed.columns]
        self.undo_stack.append(base_frame)
        self.current = transformed
        self.redo_stack.clear()

        cost_score = _estimate_cost_score(
            rows_before=int(len(base_frame)),
            columns_before=int(len(base_frame.columns)),
            rows_after=int(len(transformed)),
            columns_after=int(len(transformed.columns)),
            duration_ms=duration_ms,
        )
        record = TransformationRecord(
            transformation_id=uuid.uuid4().hex,
            dataset_id=self.dataset_id,
            operation=str(operation or "transform"),
            summary=str(summary or operation or "Applied transformation"),
            parameters=dict(parameters or {}),
            rows_before=int(len(base_frame)),
            rows_after=int(len(transformed)),
            columns_before=int(len(base_frame.columns)),
            columns_after=int(len(transformed.columns)),
            duration_ms=duration_ms,
            cost_score=cost_score,
            cost_label=_cost_label(cost_score),
        )
        self.history.append(record)
        return record

    def undo(self) -> bool:
        if not self.undo_stack:
            return False
        self.redo_stack.append(self.current.copy(deep=True))
        self.current = self.undo_stack.pop()
        return True

    def redo(self) -> bool:
        if not self.redo_stack:
            return False
        self.undo_stack.append(self.current.copy(deep=True))
        self.current = self.redo_stack.pop()
        return True

    def reset(self) -> None:
        if not self.current.equals(self.original):
            self.undo_stack.append(self.current.copy(deep=True))
        self.current = self.original.copy(deep=True)
        self.redo_stack.clear()

    def history_table(self) -> list[dict[str, Any]]:
        return [record.to_dict() for record in self.history]

    def snapshot(self) -> dict[str, Any]:
        row_delta = int(len(self.current) - len(self.original))
        column_delta = int(len(self.current.columns) - len(self.original.columns))
        total_cost = sum(record.cost_score for record in self.history)
        return {
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "row_count_original": int(len(self.original)),
            "row_count_current": int(len(self.current)),
            "column_count_original": int(len(self.original.columns)),
            "column_count_current": int(len(self.current.columns)),
            "row_delta": row_delta,
            "column_delta": column_delta,
            "transformation_count": len(self.history),
            "undo_depth": len(self.undo_stack),
            "redo_depth": len(self.redo_stack),
            "total_cost_score": round(total_cost, 3),
            "total_cost_label": _cost_label(total_cost),
            "metadata": dict(self.metadata),
        }


class MultiDatasetStateEngine:
    """Maintains reactive state per dataset and supports dataset switching."""

    def __init__(self) -> None:
        self._states: dict[str, ReactiveDatasetState] = {}
        self.active_dataset_id: str | None = None

    def ensure_state(
        self,
        dataset_id: str,
        dataset_name: str,
        dataframe: pd.DataFrame,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ReactiveDatasetState:
        existing = self._states.get(dataset_id)
        signature = str((metadata or {}).get("dataset_key") or (metadata or {}).get("signature") or "")
        existing_signature = str((existing.metadata or {}).get("dataset_key") or (existing.metadata or {}).get("signature") or "")
        if existing is not None and (not signature or signature == existing_signature):
            return existing

        state = ReactiveDatasetState.from_dataframe(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            dataframe=dataframe,
            metadata=metadata,
        )
        self._states[dataset_id] = state
        if self.active_dataset_id is None:
            self.active_dataset_id = dataset_id
        return state

    def register_state(self, state: ReactiveDatasetState) -> ReactiveDatasetState:
        self._states[state.dataset_id] = state
        if self.active_dataset_id is None:
            self.active_dataset_id = state.dataset_id
        return state

    def set_active_dataset(self, dataset_id: str) -> None:
        if dataset_id not in self._states:
            raise ValueError(f"Dataset '{dataset_id}' does not have reactive state yet.")
        self.active_dataset_id = dataset_id

    def get_state(self, dataset_id: str | None = None) -> ReactiveDatasetState | None:
        resolved_id = dataset_id or self.active_dataset_id
        if resolved_id is None:
            return None
        return self._states.get(resolved_id)

    def list_states(self) -> list[ReactiveDatasetState]:
        return sorted(self._states.values(), key=lambda item: item.dataset_name.lower())

    def remove_state(self, dataset_id: str) -> None:
        self._states.pop(dataset_id, None)
        if self.active_dataset_id == dataset_id:
            remaining_ids = sorted(self._states)
            self.active_dataset_id = remaining_ids[0] if remaining_ids else None


__all__ = [
    "MultiDatasetStateEngine",
    "ReactiveDatasetState",
    "TransformationCallable",
    "TransformationRecord",
]
