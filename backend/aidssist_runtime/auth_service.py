from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone

from backend.workflow_store import WorkflowStore

from .config import get_settings
from .state_store import UserRecord


def normalize_email(email: str) -> str:
    return str(email or "").strip().lower()


def hash_password(password: str, salt_hex: str | None = None) -> tuple[str, str]:
    if len(str(password or "")) < 8:
        raise ValueError("Password must be at least 8 characters long.")

    salt = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        str(password).encode("utf-8"),
        salt,
        120_000,
    )
    return salt.hex(), digest.hex()


def verify_password(password: str, salt_hex: str, expected_hash: str) -> bool:
    _, actual_hash = hash_password(password, salt_hex=salt_hex)
    return hmac.compare_digest(actual_hash, expected_hash)


def issue_session_token() -> str:
    return secrets.token_urlsafe(32)


def hash_session_token(token: str) -> str:
    return hashlib.sha256(str(token or "").encode("utf-8")).hexdigest()


def _build_session_expiry() -> str:
    return (
        datetime.now(timezone.utc).replace(microsecond=0)
        + timedelta(hours=max(int(get_settings().session_ttl_hours), 1))
    ).isoformat()


def register_user(email: str, password: str, display_name: str | None = None) -> tuple[UserRecord, str]:
    normalized_email = normalize_email(email)
    if not normalized_email:
        raise ValueError("Email is required.")

    salt_hex, password_hash = hash_password(password)
    with WorkflowStore() as store:
        user = store.create_user(
            email=normalized_email,
            display_name=str(display_name or "").strip() or normalized_email.split("@")[0],
            password_hash=password_hash,
            password_salt=salt_hex,
        )
        token = issue_session_token()
        store.create_session(
            user_id=user.user_id,
            token_hash=hash_session_token(token),
            expires_at=_build_session_expiry(),
        )
    return user, token


def authenticate_user(email: str, password: str) -> tuple[UserRecord, str]:
    normalized_email = normalize_email(email)
    with WorkflowStore() as store:
        user_secret = store.get_user_with_secret(normalized_email)
        if user_secret is None:
            raise ValueError("Invalid email or password.")

        user, password_hash, password_salt = user_secret
        if not verify_password(password, password_salt, password_hash):
            raise ValueError("Invalid email or password.")

        token = issue_session_token()
        store.create_session(
            user_id=user.user_id,
            token_hash=hash_session_token(token),
            expires_at=_build_session_expiry(),
        )
    return user, token


def get_user_from_token(token: str) -> UserRecord | None:
    if not token:
        return None
    with WorkflowStore() as store:
        return store.get_user_by_token_hash(hash_session_token(token))


def revoke_token(token: str) -> bool:
    if not token:
        return False
    with WorkflowStore() as store:
        return store.revoke_session(hash_session_token(token))
