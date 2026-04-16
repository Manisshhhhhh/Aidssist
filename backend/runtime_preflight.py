from __future__ import annotations

import importlib.util
import sys


MINIMUM_PYTHON = (3, 11)

STREAMLIT_REQUIRED_MODULES = {
    "altair": "altair",
    "PIL": "pillow",
    "pandas": "pandas",
    "requests": "requests",
    "sklearn": "scikit-learn",
    "sqlalchemy": "sqlalchemy",
    "streamlit": "streamlit",
}

API_REQUIRED_MODULES = {
    "boto3": "boto3",
    "fastapi": "fastapi",
    "pandas": "pandas",
    "psutil": "psutil",
    "redis": "redis",
    "rq": "rq",
    "sklearn": "scikit-learn",
    "sqlalchemy": "sqlalchemy",
    "uvicorn": "uvicorn",
}


def _format_version(version_info: tuple[int, ...]) -> str:
    return ".".join(str(part) for part in version_info)


def get_runtime_preflight_issues(target: str) -> list[str]:
    issues: list[str] = []
    minimum_version_label = _format_version(MINIMUM_PYTHON)
    current_version = _format_version(sys.version_info[:3])
    if sys.version_info < MINIMUM_PYTHON:
        issues.append(
            f"Python {current_version} is running, but Aidssist now requires Python {minimum_version_label}+ "
            "because the pinned dependency set in requirements.txt no longer supports Python 3.9."
        )

    required_modules = STREAMLIT_REQUIRED_MODULES if target == "streamlit" else API_REQUIRED_MODULES
    missing_packages = [
        package_name
        for module_name, package_name in required_modules.items()
        if importlib.util.find_spec(module_name) is None
    ]
    if missing_packages:
        issues.append("Missing required packages: " + ", ".join(sorted(missing_packages)) + ".")

    return issues


def build_setup_commands(target: str) -> str:
    commands = [
        "python3.11 -m venv .venv",
        "source .venv/bin/activate",
        "pip install --upgrade pip",
        "pip install -r requirements.txt",
    ]
    if target == "streamlit":
        commands.append("streamlit run app.py")
    else:
        commands.append("uvicorn backend.aidssist_runtime.api:app --reload")
    return "\n".join(commands)


def build_runtime_error_message(target: str) -> str:
    issues = get_runtime_preflight_issues(target)
    if not issues:
        return ""
    issue_list = "\n".join(f"- {issue}" for issue in issues)
    return (
        "Aidssist environment setup is incomplete.\n"
        f"{issue_list}\n\n"
        "Recreate the virtual environment with Python 3.11 and reinstall the project requirements:\n"
        f"{build_setup_commands(target)}"
    )


def ensure_runtime_or_raise(target: str) -> None:
    message = build_runtime_error_message(target)
    if message:
        raise RuntimeError(message)
