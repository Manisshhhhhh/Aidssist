from __future__ import annotations

import hashlib
import io
import re
from dataclasses import asdict
from pathlib import Path
from typing import Literal, Union

import pandas as pd

from backend.compat import dataclass

try:
    from sqlalchemy import create_engine
    from sqlalchemy.engine import URL
except ImportError:
    create_engine = None
    URL = None


SQL_ENGINES = ("postgres", "mysql")
SELECT_QUERY_PATTERN = re.compile(r"^\s*select\b", re.IGNORECASE)
VALID_TABLE_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_\.]*$")


class DataSourceError(RuntimeError):
    """Raised when a configured data source cannot be loaded safely."""


@dataclass(slots=True)
class CSVSourceConfig:
    kind: Literal["csv"] = "csv"
    file_name: str = "dataset.csv"
    file_bytes: bytes | None = None
    snapshot_path: str | None = None
    delimiter: str = ","
    encoding: str = "utf-8"


@dataclass(slots=True)
class ExcelSourceConfig:
    kind: Literal["excel"] = "excel"
    file_name: str = "dataset.xlsx"
    file_bytes: bytes | None = None
    snapshot_path: str | None = None
    sheet_name: str | int | None = 0


@dataclass(slots=True)
class SQLSourceConfig:
    kind: Literal["postgres", "mysql"]
    host: str
    port: int
    database: str
    username: str
    password: str
    table_name: str | None = None
    query: str | None = None
    limit: int = 1000


DataSourceConfig = Union[CSVSourceConfig, ExcelSourceConfig, SQLSourceConfig]


@dataclass(slots=True)
class LoadedDataFrame:
    dataframe: pd.DataFrame
    dataset_name: str
    dataset_key: str
    source_fingerprint: str
    source_label: str


def _safe_string(value: object) -> str:
    return str(value or "").strip()


def build_dataframe_fingerprint(df: pd.DataFrame) -> str:
    if df.empty:
        payload = f"empty:{list(df.columns)}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]

    normalized = df.copy()
    normalized.columns = [str(column) for column in normalized.columns]
    digest = pd.util.hash_pandas_object(normalized, index=True).values.tobytes()
    return hashlib.sha256(digest).hexdigest()[:16]


def build_dataset_key(dataset_name: str, source_fingerprint: str) -> str:
    return f"{dataset_name}:{source_fingerprint}"


def _read_snapshot_bytes(snapshot_path: str | None) -> bytes:
    if not snapshot_path:
        raise DataSourceError("No source snapshot is available for this workflow.")

    snapshot_file = Path(snapshot_path)
    if not snapshot_file.exists():
        raise DataSourceError(f"Saved source snapshot was not found: {snapshot_file}")

    return snapshot_file.read_bytes()


def _require_file_bytes(config: CSVSourceConfig | ExcelSourceConfig) -> bytes:
    if config.file_bytes is not None:
        return config.file_bytes
    return _read_snapshot_bytes(config.snapshot_path)


def persist_file_source_snapshot(
    config: CSVSourceConfig | ExcelSourceConfig,
    snapshot_dir: str | Path,
) -> CSVSourceConfig | ExcelSourceConfig:
    file_bytes = _require_file_bytes(config)
    snapshot_root = Path(snapshot_dir)
    snapshot_root.mkdir(parents=True, exist_ok=True)

    source_digest = hashlib.sha256(file_bytes).hexdigest()[:16]
    file_suffix = Path(config.file_name).suffix or (".csv" if config.kind == "csv" else ".xlsx")
    snapshot_path = snapshot_root / f"{source_digest}{file_suffix}"
    if not snapshot_path.exists():
        snapshot_path.write_bytes(file_bytes)

    snapshot_kwargs = asdict(config)
    snapshot_kwargs["file_bytes"] = None
    snapshot_kwargs["snapshot_path"] = str(snapshot_path)

    if config.kind == "csv":
        return CSVSourceConfig(**snapshot_kwargs)
    return ExcelSourceConfig(**snapshot_kwargs)


def serialize_source_config(config: DataSourceConfig) -> dict:
    payload = asdict(config)
    payload.pop("file_bytes", None)
    return payload


def deserialize_source_config(payload: dict) -> DataSourceConfig:
    kind = _safe_string(payload.get("kind")).lower()

    if kind == "csv":
        return CSVSourceConfig(**payload)
    if kind == "excel":
        return ExcelSourceConfig(**payload)
    if kind in SQL_ENGINES:
        return SQLSourceConfig(**payload)

    raise DataSourceError(f"Unsupported source kind: {kind or 'unknown'}")


def validate_sql_source_config(config: SQLSourceConfig) -> list[str]:
    issues: list[str] = []

    if config.kind not in SQL_ENGINES:
        issues.append("SQL engine must be either postgres or mysql.")
    if not _safe_string(config.host):
        issues.append("SQL host is required.")
    if not _safe_string(config.database):
        issues.append("SQL database is required.")
    if not _safe_string(config.username):
        issues.append("SQL username is required.")
    if config.port <= 0:
        issues.append("SQL port must be a positive integer.")
    if config.limit <= 0:
        issues.append("SQL preview limit must be a positive integer.")

    has_query = bool(_safe_string(config.query))
    has_table = bool(_safe_string(config.table_name))

    if has_query and has_table:
        issues.append("Choose either a table name or a custom SELECT query, not both.")
    if not has_query and not has_table:
        issues.append("Provide either a table name or a custom SELECT query.")

    if has_query:
        normalized_query = _safe_string(config.query)
        if not SELECT_QUERY_PATTERN.match(normalized_query):
            issues.append("Only read-only SELECT queries are allowed in phase 1.")
        if ";" in normalized_query.rstrip(";"):
            issues.append("Only a single SELECT statement is allowed.")

    if has_table and not VALID_TABLE_NAME_PATTERN.match(_safe_string(config.table_name)):
        issues.append("Table name may contain only letters, numbers, underscores, and dots.")

    return issues


def build_sqlalchemy_url(config: SQLSourceConfig) -> object:
    if create_engine is None or URL is None:
        raise DataSourceError(
            "SQL support requires `sqlalchemy`, `psycopg`, and `pymysql`. Install dependencies from requirements.txt."
        )

    drivername = "postgresql+psycopg" if config.kind == "postgres" else "mysql+pymysql"
    return URL.create(
        drivername=drivername,
        username=config.username,
        password=config.password,
        host=config.host,
        port=int(config.port),
        database=config.database,
    )


def _load_csv(config: CSVSourceConfig) -> pd.DataFrame:
    file_bytes = _require_file_bytes(config)
    try:
        return pd.read_csv(
            io.BytesIO(file_bytes),
            sep=config.delimiter or ",",
            encoding=config.encoding or "utf-8",
            low_memory=False,
        )
    except Exception as error:
        raise DataSourceError(f"Could not read CSV source: {error}") from error


def _normalize_excel_sheet_name(sheet_name: str | int | None) -> str | int | None:
    if sheet_name is None:
        return 0
    if isinstance(sheet_name, int):
        return sheet_name
    normalized = _safe_string(sheet_name)
    if not normalized:
        return 0
    if normalized.isdigit():
        return int(normalized)
    return normalized


def _load_excel(config: ExcelSourceConfig) -> pd.DataFrame:
    file_bytes = _require_file_bytes(config)
    try:
        return pd.read_excel(
            io.BytesIO(file_bytes),
            sheet_name=_normalize_excel_sheet_name(config.sheet_name),
        )
    except Exception as error:
        raise DataSourceError(f"Could not read Excel source: {error}") from error


def _build_sql_query(config: SQLSourceConfig) -> str:
    validation_issues = validate_sql_source_config(config)
    if validation_issues:
        raise DataSourceError(" ".join(validation_issues))

    if _safe_string(config.query):
        return _safe_string(config.query)

    table_name = _safe_string(config.table_name)
    return f"SELECT * FROM {table_name} LIMIT {int(config.limit)}"


def _load_sql(config: SQLSourceConfig) -> pd.DataFrame:
    sql_query = _build_sql_query(config)
    try:
        engine = create_engine(build_sqlalchemy_url(config))
        with engine.connect() as connection:
            return pd.read_sql(sql_query, connection)
    except Exception as error:
        raise DataSourceError(f"Could not read SQL source: {error}") from error


def describe_source(config: DataSourceConfig) -> str:
    if isinstance(config, CSVSourceConfig):
        return f"CSV file: {config.file_name}"
    if isinstance(config, ExcelSourceConfig):
        sheet_name = _normalize_excel_sheet_name(config.sheet_name)
        return f"Excel file: {config.file_name} (sheet: {sheet_name})"

    target = _safe_string(config.table_name) or "custom query"
    return f"{config.kind.title()} source: {config.database} / {target}"


def load_dataframe_from_source(config: DataSourceConfig) -> LoadedDataFrame:
    if isinstance(config, CSVSourceConfig):
        dataframe = _load_csv(config)
        dataset_name = config.file_name
    elif isinstance(config, ExcelSourceConfig):
        dataframe = _load_excel(config)
        dataset_name = config.file_name
    else:
        dataframe = _load_sql(config)
        source_target = _safe_string(config.table_name) or "query_result"
        dataset_name = f"{config.database}_{source_target}.csv"

    dataframe.columns = [str(column) for column in dataframe.columns]
    source_fingerprint = build_dataframe_fingerprint(dataframe)

    return LoadedDataFrame(
        dataframe=dataframe,
        dataset_name=dataset_name,
        dataset_key=build_dataset_key(dataset_name, source_fingerprint),
        source_fingerprint=source_fingerprint,
        source_label=describe_source(config),
    )
