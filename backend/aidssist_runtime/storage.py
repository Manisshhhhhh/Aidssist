from __future__ import annotations

import mimetypes
from functools import lru_cache
from pathlib import Path

try:
    import boto3
except ImportError:
    boto3 = None

from .config import get_settings


class ObjectStore:
    def __init__(self):
        settings = get_settings()
        self._settings = settings
        self._local_root = settings.object_store_dir
        self._local_root.mkdir(parents=True, exist_ok=True)
        self._client = None

        wants_s3 = settings.object_store_backend in {"s3", "minio"}
        can_use_s3 = bool(
            boto3
            and settings.s3_endpoint_url
            and settings.s3_access_key_id
            and settings.s3_secret_access_key
        )
        if wants_s3 or (settings.object_store_backend == "auto" and can_use_s3):
            if can_use_s3:
                self._client = boto3.client(
                    "s3",
                    endpoint_url=settings.s3_endpoint_url,
                    aws_access_key_id=settings.s3_access_key_id,
                    aws_secret_access_key=settings.s3_secret_access_key,
                    region_name=settings.s3_region_name,
                )
                self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        if self._client is None:
            return
        bucket_name = self._settings.s3_bucket_name
        existing_buckets = {bucket["Name"] for bucket in self._client.list_buckets().get("Buckets", [])}
        if bucket_name not in existing_buckets:
            self._client.create_bucket(Bucket=bucket_name)

    def put_bytes(self, object_key: str, data: bytes, *, content_type: str | None = None) -> str:
        normalized_key = object_key.lstrip("/")
        if self._client is not None:
            self._client.put_object(
                Bucket=self._settings.s3_bucket_name,
                Key=normalized_key,
                Body=data,
                ContentType=content_type or "application/octet-stream",
            )
            return normalized_key

        target_path = self._local_root / normalized_key
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(data)
        return normalized_key

    def get_bytes(self, object_key: str) -> bytes:
        normalized_key = object_key.lstrip("/")
        if self._client is not None:
            response = self._client.get_object(Bucket=self._settings.s3_bucket_name, Key=normalized_key)
            return response["Body"].read()

        return (self._local_root / normalized_key).read_bytes()


def build_object_key(prefix: str, identifier: str, file_name: str) -> str:
    raw_name = str(file_name or "artifact.bin").replace("\\", "/")
    parts = [part for part in raw_name.split("/") if part and part not in {".", ".."}]
    if not parts:
        parts = ["artifact.bin"]
    safe_relative_path = Path(*parts)
    final_name = safe_relative_path.name
    extension = Path(final_name).suffix or mimetypes.guess_extension(mimetypes.guess_type(final_name)[0] or "") or ""
    if not Path(final_name).suffix:
        safe_relative_path = safe_relative_path.with_name(f"{final_name}{extension}")
    return f"{prefix.strip('/')}/{identifier}/{safe_relative_path.as_posix()}"


@lru_cache(maxsize=1)
def get_object_store() -> ObjectStore:
    return ObjectStore()
