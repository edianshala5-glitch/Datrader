"""Memory-bounded LRU cache for feature arrays, and memory-aware worker counts.

The swarm explores thousands of indicator parameter combinations. Caching every computed
array forever made each worker's memory grow every generation until the machine ran out
(10 years of 15m bars: about 2 MB per array, several arrays per indicator). This cache
keeps the most recently used arrays within a byte budget and recomputes the rest on demand.
The results are identical either way; only speed depends on the budget.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from collections.abc import Callable, Hashable
from typing import Any

import numpy as np
import pandas as pd

MB = 1024 * 1024
# Per process. Override with SWARM_FEATURE_CACHE_MB (e.g. 1024 on a machine with lots of RAM).
DEFAULT_CACHE_MB = 384
# Memory a swarm worker needs besides its cache: interpreter, numpy/numba, bars, features.
WORKER_BASE_MB = 450


def cache_budget_bytes() -> int:
    raw = os.environ.get("SWARM_FEATURE_CACHE_MB", "")
    try:
        mb = int(raw) if raw else DEFAULT_CACHE_MB
    except ValueError:
        mb = DEFAULT_CACHE_MB
    return max(16, mb) * MB


def nbytes(obj: Any, _depth: int = 0) -> int:
    """Approximate memory held by a cached value (arrays dominate)."""
    if isinstance(obj, np.ndarray):
        return int(obj.nbytes)
    if isinstance(obj, (pd.Series, pd.DataFrame, pd.Index)):
        usage = obj.memory_usage(deep=False)
        return int(usage.sum() if hasattr(usage, "sum") else usage)
    if _depth < 4:
        if isinstance(obj, dict):
            return 64 + sum(nbytes(v, _depth + 1) for v in obj.values())
        if isinstance(obj, (tuple, list)):
            return 64 + sum(nbytes(v, _depth + 1) for v in obj)
    return 64


class BoundedCache:
    """``dict``-like LRU cache limited by the total size of its values."""

    def __init__(self, max_bytes: int | None = None) -> None:
        self.max_bytes = cache_budget_bytes() if max_bytes is None else max_bytes
        self._data: OrderedDict[Hashable, tuple[Any, int]] = OrderedDict()
        self.bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: Hashable) -> bool:
        return key in self._data

    def __getitem__(self, key: Hashable) -> Any:
        value, _ = self._data[key]
        self._data.move_to_end(key)
        return value

    def __setitem__(self, key: Hashable, value: Any) -> None:
        size = nbytes(value)
        if key in self._data:
            self.bytes -= self._data.pop(key)[1]
        self._data[key] = (value, size)
        self.bytes += size
        # Evict least recently used entries, but always keep the one just stored.
        while self.bytes > self.max_bytes and len(self._data) > 1:
            _, (_, old) = self._data.popitem(last=False)
            self.bytes -= old
            self.evictions += 1

    def get_or_compute(self, key: Hashable, fn: Callable[[], Any]) -> Any:
        if key in self._data:
            self.hits += 1
            return self[key]
        self.misses += 1
        value = fn()
        self[key] = value
        return value

    def clear(self) -> None:
        self._data.clear()
        self.bytes = 0


def available_memory_bytes() -> int | None:
    """Physical memory currently available, or None if it can't be determined."""
    if os.name == "nt":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
            return int(status.ullAvailPhys)
        return None
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):
        return None


def default_workers(cpu_count: int | None = None, available: int | None = None) -> int:
    """Parallel swarm workers that fit both the CPU and the free RAM.

    One core stays free for the main process; each worker is budgeted its baseline plus the
    feature cache, using at most 70% of the RAM that is free right now.
    """
    cpus = cpu_count if cpu_count is not None else (os.cpu_count() or 2)
    by_cpu = max(1, cpus - 1)
    free = available if available is not None else available_memory_bytes()
    if free is None:
        return min(by_cpu, 4)
    per_worker = WORKER_BASE_MB * MB + cache_budget_bytes()
    by_ram = int(free * 0.7 // per_worker)
    return max(1, min(by_cpu, by_ram))
