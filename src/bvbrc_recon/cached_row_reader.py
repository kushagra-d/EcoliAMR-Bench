"""Memory-budgeted batch cache wrapping `SqliteRowStore`.

This is the second iteration of this module. The first (prior task)
cached batches up to a single `max_cache_bytes` ceiling with no
eviction - once the budget was reached, NO further batches were ever
cached again for the rest of that reader's life, even if earlier
batches were no longer needed. Measured consequence (on the real
100-genome pilot, k=15/n_buckets=262144, fully dense rows): a 16-row
batch costs 50,331,716 bytes (~50.3 MB; `data` int64 + `indices` int32 +
`indptr`, measured exactly - not estimated), so the old 64 MB default
fit only ONE of the random split's 5 training batches, no matter how
many times the training loop re-read the other 4.

TWO DISTINCT BUDGETS - DO NOT CONFLATE (this is explicit per this task's
requirement):

- `max_cache_bytes`: a ceiling on how much data may be RETAINED in the
  cache once a batch has been read and handed back to the caller. This
  is the "how much do we remember between calls" budget.
- `max_reader_memory_bytes`: a ceiling on `cache_bytes + the batch
  currently being read and returned`, checked and enforced (via
  eviction) BEFORE that batch is handed back - regardless of whether it
  ends up cached. This is the "how much memory does this reader cause to
  exist at any single instant" budget, and it is always `>=
  max_cache_bytes` in any sensible configuration (it defaults to being
  EQUAL to `max_cache_bytes` when not given separately, which reproduces
  simple single-budget behavior; setting it higher gives the cache more
  room to retain large batches that would otherwise be evicted purely to
  make transient room for an even larger incoming one).

Concretely, for each requested batch:
  1. Evict least-recently-used cached entries until
     `cache_bytes + this_batch_bytes <= max_reader_memory_bytes`
     (always enforced - this is the hard, instant-in-time ceiling).
  2. If, after that, `cache_bytes + this_batch_bytes <= max_cache_bytes`
     too, retain (cache) this batch. Otherwise it is still returned
     correctly (budget enforcement NEVER drops rows or features - see
     Step 4 verification in `reports/memory_budgeted_reader_assessment.md`)
     but is not cached - counted as a `n_fallbacks` event, distinct from
     an ordinary cache miss that WAS cached.

Eviction order is deterministic: strict LRU (least-recently-used first),
implemented with `collections.OrderedDict` - a cache hit moves that
entry to the most-recently-used end; eviction always pops from the
least-recently-used end. Given the same sequence of reads, eviction
order is reproducible run to run (tested).

`max_batch_size` is a separate, independent safety ceiling: the
EFFECTIVE batch size used for any call is `min(caller's batch_size,
max_batch_size)` - protects against a caller accidentally requesting an
unbounded batch that would balloon memory regardless of cache
bookkeeping.

THE SQLITE STORE REMAINS THE DURABLE SOURCE OF TRUTH and row
reconstruction stays fully vectorized (delegated to
`SqliteRowStore._rows_to_csr`, unchanged by this module). This cache is
a pure performance layer: `max_cache_bytes=0` (or
`max_reader_memory_bytes=0`) disables caching entirely and still
produces correct results, just without the speedup.

CACHE INVALIDATION: unchanged from the prior version - rows are
append-only and immutable once written (enforced by the schema), so a
cached batch for a fixed `row_index` tuple cannot go stale. `clear_cache()`
resets all state for explicit invalidation (tests, or a future caller
that could mutate rows).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from scipy import sparse

from .sqlite_row_store import SqliteRowStore

DEFAULT_MAX_CACHE_BYTES = 64_000_000
DEFAULT_MAX_BATCH_SIZE = 256


@dataclass
class CacheStats:
    n_hits: int = 0
    n_misses: int = 0
    n_evictions: int = 0  # individual cached ENTRIES evicted, cumulative
    n_fallbacks: int = 0  # misses correctly served via SQLite but NOT cached (budget-constrained)
    n_bytes_cached: int = 0  # CURRENT cache occupancy (not cumulative)
    n_batches_cached: int = 0  # CURRENT number of cached entries
    peak_bytes_cached: int = 0  # high-water mark of n_bytes_cached, for reporting


class CachedRowReader:
    """Read-only, memory-budgeted wrapper around `SqliteRowStore`. Same
    `read_rows_for_indices` contract as `SqliteRowStore`, so
    `evaluate_streaming_hashed_matrix` works with it unmodified.
    `iter_batches` is an uncached passthrough (a single full pass never
    repeats, so caching it has no benefit)."""

    def __init__(
        self,
        row_store: SqliteRowStore,
        max_cache_bytes: int = DEFAULT_MAX_CACHE_BYTES,
        max_reader_memory_bytes: int | None = None,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
    ):
        self._store = row_store
        self.max_cache_bytes = max_cache_bytes
        self.max_reader_memory_bytes = (
            max_reader_memory_bytes if max_reader_memory_bytes is not None else max_cache_bytes
        )
        self.max_batch_size = max_batch_size
        self._cache: OrderedDict[tuple[int, ...], tuple[list[str], list[int], sparse.csr_matrix]] = OrderedDict()
        self.stats = CacheStats()

    @property
    def n_buckets(self) -> int:
        return self._store.n_buckets

    @property
    def path(self):
        return self._store.path

    def clear_cache(self) -> None:
        self._cache = OrderedDict()
        self.stats = CacheStats()

    @staticmethod
    def _csr_bytes(matrix: sparse.csr_matrix) -> int:
        return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)

    def _evict_lru_until(self, incoming_bytes: int, ceiling: int) -> None:
        while self._cache and (self.stats.n_bytes_cached + incoming_bytes > ceiling):
            _, evicted = self._cache.popitem(last=False)  # LRU: pop the OLDEST entry
            self.stats.n_bytes_cached -= self._csr_bytes(evicted[2])
            self.stats.n_batches_cached -= 1
            self.stats.n_evictions += 1

    def iter_batches(self, batch_size: int):
        """Uncached passthrough - see module docstring for why."""
        yield from self._store.iter_batches(min(batch_size, self.max_batch_size))

    def read_rows_for_indices(self, wanted_row_indices: set[int], batch_size: int = 64):
        effective_batch_size = max(1, min(batch_size, self.max_batch_size))
        wanted_sorted = tuple(sorted(int(i) for i in wanted_row_indices))
        for i in range(0, len(wanted_sorted), effective_batch_size):
            chunk_key = wanted_sorted[i : i + effective_batch_size]

            if chunk_key in self._cache:
                self.stats.n_hits += 1
                self._cache.move_to_end(chunk_key)  # mark most-recently-used
                yield self._cache[chunk_key]
                continue

            self.stats.n_misses += 1
            batch = None
            for genome_ids, row_indices, matrix in self._store.read_rows_for_indices(
                set(chunk_key), batch_size=len(chunk_key)
            ):
                batch = (genome_ids, row_indices, matrix)
            if batch is None:
                continue  # none of this chunk's rows exist in the store

            size = self._csr_bytes(batch[2])

            # Ceiling 1 (always enforced): cache + THIS batch must fit the
            # total reader-memory budget, regardless of caching decision.
            self._evict_lru_until(size, self.max_reader_memory_bytes)
            fits_reader_budget = self.stats.n_bytes_cached + size <= self.max_reader_memory_bytes

            # Ceiling 2: only if the above holds, decide whether to RETAIN
            # (cache) this batch, under the (<=) narrower cache-only budget.
            if fits_reader_budget:
                self._evict_lru_until(size, self.max_cache_bytes)
                fits_cache_budget = self.stats.n_bytes_cached + size <= self.max_cache_bytes
            else:
                fits_cache_budget = False

            if fits_reader_budget and fits_cache_budget:
                self._cache[chunk_key] = batch
                self.stats.n_bytes_cached += size
                self.stats.n_batches_cached += 1
                self.stats.peak_bytes_cached = max(self.stats.peak_bytes_cached, self.stats.n_bytes_cached)
            else:
                self.stats.n_fallbacks += 1
            yield batch


class NonEvictingCachedRowReader:
    """Strategy #3 (this task): retains batches until `max_cache_bytes`
    is reached, then PERMANENTLY falls back to SQLite for anything that
    doesn't already fit - never evicts anything once cached.

    Reinstated deliberately after measurement (not assumed): for a
    workload that repeats the SAME fixed key set every pass (training
    epochs re-scanning the same batches in the same order), whatever
    fits on the very first pass stays cached forever, giving a stable
    partial hit rate proportional to the fraction of the working set
    that fits. `CachedRowReader` (LRU, above) can thrash to a ZERO hit
    rate on exactly this access pattern when the budget doesn't cover
    the full working set, because it evicts the batch that's about to
    be needed again to make room for the one in front of it - this
    class cannot do that, by construction, since it never evicts.

    Same two-budget semantics as `CachedRowReader` (`max_cache_bytes` /
    `max_reader_memory_bytes` - see that class's docstring), same
    `max_batch_size` safety ceiling, same read-only relationship to the
    SQLite store as the durable source of truth."""

    def __init__(
        self,
        row_store: SqliteRowStore,
        max_cache_bytes: int = DEFAULT_MAX_CACHE_BYTES,
        max_reader_memory_bytes: int | None = None,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
    ):
        self._store = row_store
        self.max_cache_bytes = max_cache_bytes
        self.max_reader_memory_bytes = (
            max_reader_memory_bytes if max_reader_memory_bytes is not None else max_cache_bytes
        )
        self.max_batch_size = max_batch_size
        self._cache: dict[tuple[int, ...], tuple[list[str], list[int], sparse.csr_matrix]] = {}
        self.stats = CacheStats()

    @property
    def n_buckets(self) -> int:
        return self._store.n_buckets

    @property
    def path(self):
        return self._store.path

    def clear_cache(self) -> None:
        self._cache = {}
        self.stats = CacheStats()

    def iter_batches(self, batch_size: int):
        yield from self._store.iter_batches(min(batch_size, self.max_batch_size))

    def read_rows_for_indices(self, wanted_row_indices: set[int], batch_size: int = 64):
        effective_batch_size = max(1, min(batch_size, self.max_batch_size))
        wanted_sorted = tuple(sorted(int(i) for i in wanted_row_indices))
        for i in range(0, len(wanted_sorted), effective_batch_size):
            chunk_key = wanted_sorted[i : i + effective_batch_size]

            cached = self._cache.get(chunk_key)
            if cached is not None:
                self.stats.n_hits += 1
                yield cached
                continue

            self.stats.n_misses += 1
            batch = None
            for genome_ids, row_indices, matrix in self._store.read_rows_for_indices(
                set(chunk_key), batch_size=len(chunk_key)
            ):
                batch = (genome_ids, row_indices, matrix)
            if batch is None:
                continue

            size = CachedRowReader._csr_bytes(batch[2])
            fits_reader_budget = self.stats.n_bytes_cached + size <= self.max_reader_memory_bytes
            fits_cache_budget = self.stats.n_bytes_cached + size <= self.max_cache_bytes

            if fits_reader_budget and fits_cache_budget:
                self._cache[chunk_key] = batch
                self.stats.n_bytes_cached += size
                self.stats.n_batches_cached += 1
                self.stats.peak_bytes_cached = max(self.stats.peak_bytes_cached, self.stats.n_bytes_cached)
            else:
                self.stats.n_fallbacks += 1  # permanent for this key - never retried
            yield batch
