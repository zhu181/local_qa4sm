"""Time series cache for avoiding redundant file reads when processing grid points."""

import threading
from collections import OrderedDict

import pandas as pd

_ts_cache: "TimeSeriesCache | None" = None
_ts_cache_lock = threading.Lock()


class TimeSeriesCache:
    """LRU cache for time series DataFrames with byte-size eviction.

    Stores DataFrames keyed by a tuple identity. When total cached size exceeds
    ``max_size_bytes``, the least recently used entries are evicted until the
    cache fits within the budget.
    """

    def __init__(self, max_size_mb: int = 512):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self._store: OrderedDict[tuple, pd.DataFrame] = OrderedDict()
        self._size_bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._lock = threading.Lock()

    def get(self, key: tuple) -> pd.DataFrame | None:
        """Return cached DataFrame (or None) and mark as most recently used."""
        with self._lock:
            if key not in self._store:
                self._misses += 1
                return None
            self._store.move_to_end(key)
            self._hits += 1
            return self._store[key]

    def put(self, key: tuple, value: pd.DataFrame) -> None:
        """Store value, evicting LRU entries if total size exceeds the limit."""
        with self._lock:
            if key in self._store:
                old = self._store.pop(key)
                self._size_bytes -= self._estimate_size(old)
            self._store[key] = value
            self._size_bytes += self._estimate_size(value)
            self._evict()

    def _evict(self) -> None:
        """Evict least recently used entries while size exceeds the limit."""
        while self._size_bytes > self.max_size_bytes and self._store:
            _, evicted = self._store.popitem(last=False)
            self._size_bytes -= self._estimate_size(evicted)
            self._evictions += 1

    @staticmethod
    def _estimate_size(df: pd.DataFrame) -> int:
        """Estimate DataFrame memory usage in bytes."""
        try:
            return int(df.memory_usage(deep=True).sum())
        except Exception:
            return 0

    def stats(self) -> dict:
        """Return a snapshot of cache statistics."""
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "size_bytes": self._size_bytes,
                "entries": len(self._store),
            }

    def reset_stats(self) -> None:
        """Reset hit/miss/eviction counters without clearing entries."""
        with self._lock:
            self._hits = 0
            self._misses = 0
            self._evictions = 0

    def clear(self) -> None:
        """Remove all cached entries."""
        with self._lock:
            self._store.clear()
            self._size_bytes = 0


def get_ts_cache() -> TimeSeriesCache:
    """Return the process-wide singleton TimeSeriesCache, creating it on first use."""
    global _ts_cache
    if _ts_cache is None:
        with _ts_cache_lock:
            if _ts_cache is None:
                from validator import settings

                _ts_cache = TimeSeriesCache(max_size_mb=settings.TS_CACHE_SIZE_MB)
    return _ts_cache


def cached_read(reader, *args, **kwargs) -> pd.DataFrame:
    """Read a time series from ``reader``, caching the result for reuse.

    The cache key combines the reader identity with the call arguments. On a
    cache hit the cached DataFrame is returned as a copy to avoid mutation by
    callers. On a miss the underlying ``reader.read`` is invoked and its
    result cached (also returning a copy to the caller).
    """
    cache = get_ts_cache()
    if cache.max_size_bytes <= 0:
        return reader.read(*args, **kwargs)

    key = (id(reader), args, tuple(sorted(kwargs.items())))
    cached = cache.get(key)
    if cached is not None:
        return cached.copy()

    result = reader.read(*args, **kwargs)
    cache.put(key, result)
    return result.copy()


class CachedReader:
    """Wraps a reader to cache its .read() results."""

    def __init__(self, reader):
        self._reader = reader

    @property
    def grid(self):
        return self._reader.grid

    def read(self, *args, **kwargs):
        return cached_read(self._reader, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._reader, name)
