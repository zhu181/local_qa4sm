import numpy as np
import pandas as pd
import pytest

from validator.ts_cache import CachedReader, TimeSeriesCache, cached_read


@pytest.fixture
def small_cache():
    return TimeSeriesCache(max_size_mb=1)  # 1MB


def test_cache_put_get(small_cache):
    """Test basic put/get."""
    df = pd.DataFrame({"val": [1, 2, 3]})
    small_cache.put(("reader1", 1, "sm"), df)

    result = small_cache.get(("reader1", 1, "sm"))
    assert result is not None
    assert len(result) == 3


def test_cache_miss(small_cache):
    """Test cache miss returns None."""
    result = small_cache.get(("nonexistent", 0, "x"))
    assert result is None


def test_cache_stats(small_cache):
    """Test that stats track hits and misses."""
    df = pd.DataFrame({"val": [1, 2, 3]})
    small_cache.put(("key", 0, "sm"), df)

    small_cache.get(("key", 0, "sm"))  # hit
    small_cache.get(("missing", 0, "sm"))  # miss

    stats = small_cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1


def test_cache_lru_eviction():
    """Test LRU eviction when cache is full."""
    cache = TimeSeriesCache(max_size_mb=0.02)  # ~20KB

    # Add multiple DataFrames to exceed capacity
    for i in range(10):
        df = pd.DataFrame({"val": np.random.randn(1000)})  # ~8KB each
        cache.put(("reader", i, "sm"), df)

    stats = cache.stats()
    # Some entries should have been evicted
    assert stats["evictions"] > 0


def test_cached_read_returns_copy():
    """Test that cached_read returns a copy, protecting the cached original."""
    call_count = 0

    class MockReader:
        def read(self, gpi):
            nonlocal call_count
            call_count += 1
            return pd.DataFrame({"val": [1, 2, 3]})

    reader = MockReader()
    result1 = cached_read(reader, 1)
    result1.loc[0, "val"] = 999  # mutate the returned copy
    # The cached original should be unchanged
    result2 = cached_read(reader, 1)
    assert result2["val"][0] == 1
    # Second read should be a cache hit
    assert call_count == 1


def test_cached_reader_wrapper():
    """Test that CachedReader delegates to underlying reader."""
    call_count = 0

    class MockReader:
        def read(self, gpi):
            nonlocal call_count
            call_count += 1
            return pd.DataFrame({"val": [gpi]})

        @property
        def grid(self):
            return "mock_grid"

    mock = MockReader()
    cached = CachedReader(mock)

    # First read - cache miss
    result1 = cached.read(42)
    assert call_count == 1
    assert result1["val"][0] == 42

    # Second read of same gpi - should be cache hit
    result2 = cached.read(42)
    assert call_count == 1  # No additional read
    assert result2["val"][0] == 42

    # Grid should be delegated
    assert cached.grid == "mock_grid"
