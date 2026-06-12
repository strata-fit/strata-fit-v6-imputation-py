from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "central",
    "run_local_imputation",
    "partial_compute",
    "get_local_sums",
]


if TYPE_CHECKING:
    from .central import central, run_local_imputation
    from .partial import get_local_sums, partial_compute


def __getattr__(name: str) -> Any:
    if name in {"central", "run_local_imputation"}:
        central_module = import_module(".central", __name__)
        return getattr(central_module, name)
    if name in {"partial_compute", "get_local_sums"}:
        partial_module = import_module(".partial", __name__)
        return getattr(partial_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
