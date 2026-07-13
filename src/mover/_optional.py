"""Helpers for clear optional-dependency errors."""

from __future__ import annotations

import importlib
import importlib.util
from types import ModuleType
from typing import Iterable


def extra_install_command(extra: str) -> str:
    return f'pip install "mover[{extra}]"'


def missing_extra_error(
    *,
    extra: str,
    feature: str,
    distributions: Iterable[str],
) -> ModuleNotFoundError:
    missing = ", ".join(sorted(set(distributions)))
    return ModuleNotFoundError(
        f"{feature} requires optional dependencies ({missing}). "
        f"Install them with: {extra_install_command(extra)}"
    )


def is_module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def require_modules(
    *,
    extra: str,
    feature: str,
    modules: dict[str, str],
) -> None:
    missing = [
        distribution
        for module_name, distribution in modules.items()
        if not is_module_available(module_name)
    ]
    if missing:
        raise missing_extra_error(
            extra=extra,
            feature=feature,
            distributions=missing,
        )


def import_optional_module(
    module_name: str,
    *,
    distribution: str,
    extra: str,
    feature: str,
) -> ModuleType:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        root_module = module_name.split(".", 1)[0]
        if error.name not in {module_name, root_module}:
            raise
        raise missing_extra_error(
            extra=extra,
            feature=feature,
            distributions=[distribution],
        ) from error
