"""Deterministic minimizer-based k-mer selection, as a separate, reduced
feature representation from the full canonical-k-mer count already used
elsewhere in this project.

MINIMIZER CONVENTION (documented precisely, since this is a genuine
reproducibility-affecting design choice, not a universal standard):

  - k-mers are CANONICALIZED FIRST (via `kmer_features.canonical_kmer`,
    the same function used everywhere else in this project), then the
    minimizer of a window of `w` consecutive canonical k-mers is simply
    the lexicographically smallest of those `w` strings. Canonicalizing
    before selecting means strand does not affect which k-mers are
    chosen as minimizers - consistent with this project's "a k-mer and
    its reverse complement are the same feature" rule everywhere else.
  - A window requires `w` CONSECUTIVE, ALL-VALID k-mer start positions
    (none containing an ambiguous base). If any of the `w` positions is
    invalid, NO minimizer is emitted for that window at all - this is a
    strict, simple, documented choice; it is NOT the only defensible
    convention (some tools would still emit a minimizer from the
    remaining valid sub-window), but it is unambiguous and easy to test.
  - Contigs are processed independently; a window never spans two
    contigs (same rule as everywhere else in this project).
  - A contig producing fewer than `w` valid consecutive k-mer positions
    (too short, or k-mers invalidated by ambiguous bases) contributes NO
    minimizer at all - handled explicitly, not an error.
  - Tie-break: when two k-mers within a window are equal, the algorithm
    below (a monotonic-deque sliding-window minimum) keeps the MOST
    RECENT (rightmost) occurrence as "the" minimum - an arbitrary but
    deterministic and documented choice. It only matters for which
    POSITION is credited; the minimizer VALUE counted is identical
    either way since the tied k-mers are, by definition, equal strings.
  - The feature matrix's value for a given (genome, minimizer k-mer) is
    the COUNT of valid window positions whose minimizer equals that
    k-mer - directly analogous to how raw k-mer counts are built
    elsewhere, just counting "which k-mer won its window" instead of
    "every k-mer".

Performance note: minimizer selection is implemented with a monotonic
deque (classic O(n) amortized sliding-window-minimum algorithm), NOT a
brute-force O(n*w) rescan of each window - verified by a dedicated timing
assumption check is unnecessary here since correctness, not asymptotic
complexity, is what tests cover; the complexity choice is what makes
running this on 100 full genomes computationally feasible at all.
"""

from __future__ import annotations

from collections import Counter, deque

from .kmer_features import canonical_kmer
from .streaming_kmer_build import StreamingBuildResult, peak_memory_mb

VALID_BASES = set("ACGT")


def canonical_kmers_for_contig(contig: str, k: int) -> list[str | None]:
    """One entry per valid k-mer start position in this contig; `None`
    where that window contains an ambiguous base. Bounded by contig
    length (not genome length), so safe to materialize per contig."""
    seq = contig.upper()
    n = len(seq)
    result: list[str | None] = []
    for i in range(n - k + 1):
        window = seq[i : i + k]
        result.append(canonical_kmer(window) if VALID_BASES.issuperset(window) else None)
    return result


def minimizer_counts_for_contig(contig: str, k: int, w: int) -> Counter:
    """Counter of {minimizer_kmer: count} for ONE contig. See module
    docstring for the exact convention (canonicalize-then-minimize,
    strict all-valid window, no cross-gap windows, rightmost tie-break).
    """
    canonical = canonical_kmers_for_contig(contig, k)
    counts: Counter = Counter()
    dq: deque[tuple[int, str]] = deque()  # (position, kmer), strictly increasing kmer order
    valid_run_start = 0

    for i, kmer in enumerate(canonical):
        if kmer is None:
            dq.clear()
            valid_run_start = i + 1
            continue

        while dq and dq[-1][1] >= kmer:
            dq.pop()
        dq.append((i, kmer))

        window_start = i - w + 1
        while dq and dq[0][0] < window_start:
            dq.popleft()

        if i - valid_run_start + 1 >= w:
            counts[dq[0][1]] += 1

    return counts


def minimizer_counts_for_assembly(contigs: list[str], k: int, w: int) -> Counter:
    """Sums per-contig minimizer counts - contigs are ALWAYS processed
    independently (never concatenated), so a minimizer window never
    spans a contig boundary."""
    total: Counter = Counter()
    for contig in contigs:
        total.update(minimizer_counts_for_contig(contig, k, w))
    return total


def build_minimizer_matrix_streaming(
    contigs_by_genome: dict[str, list[str]],
    genome_order: list[str],
    k: int,
    w: int,
    memory_limit_mb: float = 11_000.0,
    memory_check_fn=peak_memory_mb,
    progress_callback=None,
) -> StreamingBuildResult:
    """Same memory-bounded, one-genome-at-a-time pattern as
    `streaming_kmer_build.build_matrix_streaming` (reused, not
    reimplemented, for the result type and memory-check contract) - here
    counting minimizers per genome instead of every raw k-mer.
    """
    import numpy as np
    from scipy import sparse

    vocabulary: dict[str, int] = {}
    rows: list[int] = []
    cols: list[int] = []
    data: list[int] = []
    processed_genome_ids: list[str] = []
    aborted_for_memory = False

    for row_idx, genome_id in enumerate(genome_order):
        genome_counts = minimizer_counts_for_assembly(contigs_by_genome[genome_id], k, w)

        for kmer, count in genome_counts.items():
            col_idx = vocabulary.setdefault(kmer, len(vocabulary))
            rows.append(row_idx)
            cols.append(col_idx)
            data.append(count)
        processed_genome_ids.append(genome_id)
        del genome_counts

        mem_mb = memory_check_fn()
        if progress_callback is not None:
            progress_callback(row_idx, genome_id, len(vocabulary), len(data), mem_mb)
        if mem_mb > memory_limit_mb:
            aborted_for_memory = True
            break

    n_processed = len(processed_genome_ids)
    matrix = sparse.coo_matrix(
        (data, (rows, cols)), shape=(n_processed, len(vocabulary)), dtype=np.int64
    ).tocsr()

    return StreamingBuildResult(
        matrix=matrix,
        vocabulary=vocabulary,
        processed_genome_ids=processed_genome_ids,
        n_requested_genomes=len(genome_order),
        complete=(not aborted_for_memory) and n_processed == len(genome_order),
        aborted_for_memory=aborted_for_memory,
    )
