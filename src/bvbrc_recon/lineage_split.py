"""MLST-group-aware train/test split, for comparison against the pilot's
existing random-split baseline.

Grouping rule, stated rather than assumed: a genome's group key is its
`genome_mlst` value when present. A genome with a MISSING MLST value is
given its OWN singleton group (its genome_id) - missing-MLST genomes are
never merged into one shared "unknown" lineage, since that would be an
invented assumption that they share a lineage, which is not supported by
the data.

MLST grouping is a practical proxy for relatedness, not a guarantee of
complete genomic independence between groups: genomes with different MLST
sequence types can still share recent horizontal transfers, plasmids, or
other close relationships that MLST (a 7-gene scheme) does not capture.
Nothing in this module claims otherwise.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from sklearn.model_selection import GroupShuffleSplit


@dataclass
class LineageGroupAssignment:
    group_by_genome: dict[str, str] = field(default_factory=dict)
    missing_mlst_genome_ids: list[str] = field(default_factory=list)
    scheme_by_genome: dict[str, str | None] = field(default_factory=dict)
    majority_scheme: str | None = None
    anomalous_scheme_genome_ids: list[str] = field(default_factory=list)
    group_sizes: Counter = field(default_factory=Counter)

    @property
    def n_groups(self) -> int:
        return len(self.group_sizes)


def _extract_scheme(mlst_value: str | None) -> str | None:
    """`"MLST.ecoli_achtman_4.73"` -> `"ecoli_achtman_4"`. Returns None for a
    missing value or a value that doesn't match the expected `MLST.<scheme>.<st>`
    shape (reported as-is, not guessed at)."""
    if mlst_value is None:
        return None
    parts = str(mlst_value).split(".")
    return parts[1] if len(parts) >= 2 else None


def assign_lineage_groups(
    genome_ids: list[str], mlst_by_genome: dict[str, str | None]
) -> LineageGroupAssignment:
    """Builds the group-key map and reports MLST coverage/anomalies for
    exactly the given genome_ids (callers pass the pilot's own 100
    genome_ids - this never looks beyond them)."""
    assignment = LineageGroupAssignment()

    for genome_id in genome_ids:
        mlst_value = mlst_by_genome.get(genome_id)
        # Defends against a raw pandas NaN float sneaking in (NaN is not
        # `None` and `str(nan)` is the truthy, non-empty string "nan") -
        # this exact pattern has bitten this project before; callers should
        # use csv_loader.load_antibiotic_csv, but this is checked here too.
        is_nan_float = isinstance(mlst_value, float) and mlst_value != mlst_value
        has_mlst = (
            mlst_value is not None
            and not is_nan_float
            and str(mlst_value).strip() != ""
        )
        if has_mlst:
            assignment.group_by_genome[genome_id] = mlst_value
        else:
            # own singleton group - never shared with another missing-MLST genome
            assignment.group_by_genome[genome_id] = f"__missing_mlst__{genome_id}"
            assignment.missing_mlst_genome_ids.append(genome_id)
        assignment.scheme_by_genome[genome_id] = _extract_scheme(mlst_value) if has_mlst else None

    assignment.group_sizes = Counter(assignment.group_by_genome.values())

    observed_schemes = Counter(s for s in assignment.scheme_by_genome.values() if s is not None)
    if observed_schemes:
        assignment.majority_scheme = observed_schemes.most_common(1)[0][0]
        assignment.anomalous_scheme_genome_ids = [
            gid
            for gid, scheme in assignment.scheme_by_genome.items()
            if scheme is not None and scheme != assignment.majority_scheme
        ]

    return assignment


@dataclass
class LineageSplitResult:
    seed: int
    requested_test_size: float
    feasible: bool
    blocker: str | None = None
    train_genome_ids: list[str] = field(default_factory=list)
    test_genome_ids: list[str] = field(default_factory=list)
    train_group_count: int = 0
    test_group_count: int = 0
    train_class_counts: dict[str, int] = field(default_factory=dict)
    test_class_counts: dict[str, int] = field(default_factory=dict)
    actual_test_fraction: float | None = None
    warnings: list[str] = field(default_factory=list)


def split_by_lineage_group(
    genome_ids: list[str],
    labels: list[str],
    group_by_genome: dict[str, str],
    seed: int,
    test_size: float = 0.3,
) -> LineageSplitResult:
    """One GroupShuffleSplit (train_test_split's group-aware counterpart,
    already part of this project's scikit-learn dependency) over the given
    genome_ids, with group integrity verified explicitly afterward rather
    than only trusted from the library's documented behavior."""
    result = LineageSplitResult(seed=seed, requested_test_size=test_size, feasible=False)

    groups = np.array([group_by_genome[gid] for gid in genome_ids])
    n_groups = len(set(groups))

    if n_groups < 2:
        result.blocker = (
            f"Only {n_groups} distinct MLST group(s) among the selected genomes - a train/test "
            f"split that keeps every group on one side is not possible."
        )
        return result

    y = np.array(labels)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    try:
        train_idx, test_idx = next(gss.split(np.zeros(len(genome_ids)), y, groups))
    except ValueError as exc:
        result.blocker = f"GroupShuffleSplit could not produce a split: {exc}"
        return result

    if len(train_idx) == 0 or len(test_idx) == 0:
        result.blocker = (
            f"The resulting split left one partition empty (train={len(train_idx)}, "
            f"test={len(test_idx)}) given this group structure and test_size={test_size} - "
            f"not a usable split."
        )
        return result

    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])
    overlap = train_groups & test_groups
    if overlap:
        # Should never happen given GroupShuffleSplit's contract - verified
        # explicitly rather than assumed, per this task's own instructions.
        result.blocker = f"INTERNAL INVARIANT VIOLATED: {len(overlap)} group(s) appear in both partitions: {overlap}"
        return result

    result.feasible = True
    result.train_genome_ids = [genome_ids[i] for i in train_idx]
    result.test_genome_ids = [genome_ids[i] for i in test_idx]
    result.train_group_count = len(train_groups)
    result.test_group_count = len(test_groups)
    result.actual_test_fraction = len(test_idx) / len(genome_ids)

    # str(...) avoids numpy.str_ keys leaking into reports/tests - plain Python str.
    result.train_class_counts = dict(Counter(str(label) for label in y[train_idx]))
    result.test_class_counts = dict(Counter(str(label) for label in y[test_idx]))

    if abs(result.actual_test_fraction - test_size) > 0.1:
        result.warnings.append(
            f"Actual test fraction ({result.actual_test_fraction:.2%}) deviates from the "
            f"requested {test_size:.0%} by more than 10 points - a direct consequence of keeping "
            f"MLST groups intact with this group-size distribution, not a bug."
        )
    for class_name in set(labels):
        if result.train_class_counts.get(class_name, 0) == 0:
            result.warnings.append(f"Training partition has ZERO genomes of class {class_name!r}.")
        if result.test_class_counts.get(class_name, 0) == 0:
            result.warnings.append(f"Test partition has ZERO genomes of class {class_name!r}.")

    return result
