from __future__ import annotations

import sys
from dataclasses import dataclass as _dataclass
from typing import Any, TypeVar


_T = TypeVar("_T")


if sys.version_info >= (3, 10):
    dataclass = _dataclass
else:
    def dataclass(_cls: type[_T] | None = None, /, *args: Any, **kwargs: Any):
        kwargs.pop("slots", None)
        if _cls is None:
            def wrap(cls: type[_T]) -> type[_T]:
                return _dataclass(cls, *args, **kwargs)

            return wrap
        return _dataclass(_cls, *args, **kwargs)


__all__ = ["dataclass"]
