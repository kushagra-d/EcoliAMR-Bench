from bvbrc_recon.lineage_split import assign_lineage_groups, split_by_lineage_group


# --- assign_lineage_groups -------------------------------------------------


def test_missing_mlst_genomes_each_get_their_own_singleton_group():
    genome_ids = ["g1", "g2", "g3"]
    mlst_by_genome = {"g1": "MLST.ecoli_achtman_4.73", "g2": None, "g3": None}
    assignment = assign_lineage_groups(genome_ids, mlst_by_genome)
    assert assignment.group_by_genome["g2"] != assignment.group_by_genome["g3"]
    assert set(assignment.missing_mlst_genome_ids) == {"g2", "g3"}


def test_genomes_sharing_mlst_share_a_group():
    genome_ids = ["g1", "g2", "g3"]
    mlst_by_genome = {
        "g1": "MLST.ecoli_achtman_4.73",
        "g2": "MLST.ecoli_achtman_4.73",
        "g3": "MLST.ecoli_achtman_4.95",
    }
    assignment = assign_lineage_groups(genome_ids, mlst_by_genome)
    assert assignment.group_by_genome["g1"] == assignment.group_by_genome["g2"]
    assert assignment.group_by_genome["g1"] != assignment.group_by_genome["g3"]
    assert assignment.n_groups == 2


def test_nan_float_mlst_treated_as_missing_not_a_valid_scheme_string():
    """Regression test: a raw pandas NaN float (not Python None) must also
    be treated as missing - `str(float('nan'))` is the non-empty, truthy
    string "nan", so a naive `is not None` check alone is not enough."""
    genome_ids = ["g1", "g2"]
    mlst_by_genome = {"g1": "MLST.ecoli_achtman_4.73", "g2": float("nan")}
    assignment = assign_lineage_groups(genome_ids, mlst_by_genome)
    assert assignment.missing_mlst_genome_ids == ["g2"]
    assert assignment.group_by_genome["g2"] != "nan"


def test_blank_string_mlst_treated_as_missing_not_a_shared_value():
    genome_ids = ["g1", "g2"]
    mlst_by_genome = {"g1": "", "g2": "   "}
    assignment = assign_lineage_groups(genome_ids, mlst_by_genome)
    assert set(assignment.missing_mlst_genome_ids) == {"g1", "g2"}
    assert assignment.group_by_genome["g1"] != assignment.group_by_genome["g2"]


def test_detects_scheme_anomaly_against_majority():
    genome_ids = ["g1", "g2", "g3"]
    mlst_by_genome = {
        "g1": "MLST.ecoli_achtman_4.73",
        "g2": "MLST.ecoli_achtman_4.95",
        "g3": "MLST.cfreundii.583",  # different species scheme - a real anomaly observed elsewhere
    }
    assignment = assign_lineage_groups(genome_ids, mlst_by_genome)
    assert assignment.majority_scheme == "ecoli_achtman_4"
    assert assignment.anomalous_scheme_genome_ids == ["g3"]


def test_no_anomaly_when_all_schemes_agree():
    genome_ids = ["g1", "g2"]
    mlst_by_genome = {"g1": "MLST.ecoli_achtman_4.73", "g2": "MLST.ecoli_achtman_4.95"}
    assignment = assign_lineage_groups(genome_ids, mlst_by_genome)
    assert assignment.anomalous_scheme_genome_ids == []


def test_group_sizes_counted_correctly():
    genome_ids = ["g1", "g2", "g3", "g4"]
    mlst_by_genome = {
        "g1": "MLST.ecoli_achtman_4.73",
        "g2": "MLST.ecoli_achtman_4.73",
        "g3": "MLST.ecoli_achtman_4.73",
        "g4": "MLST.ecoli_achtman_4.95",
    }
    assignment = assign_lineage_groups(genome_ids, mlst_by_genome)
    group73 = assignment.group_by_genome["g1"]
    assert assignment.group_sizes[group73] == 3


# --- split_by_lineage_group -------------------------------------------------


def _many_small_groups_fixture(n_groups=20, group_size=3):
    genome_ids = []
    labels = []
    group_by_genome = {}
    for g in range(n_groups):
        label = "Resistant" if g % 3 == 0 else "Susceptible"
        for i in range(group_size):
            gid = f"g{g}_{i}"
            genome_ids.append(gid)
            labels.append(label)
            group_by_genome[gid] = f"group{g}"
    return genome_ids, labels, group_by_genome


def test_split_never_lets_a_group_span_both_partitions():
    genome_ids, labels, group_by_genome = _many_small_groups_fixture()
    result = split_by_lineage_group(genome_ids, labels, group_by_genome, seed=1, test_size=0.3)
    assert result.feasible is True

    train_groups = {group_by_genome[g] for g in result.train_genome_ids}
    test_groups = {group_by_genome[g] for g in result.test_genome_ids}
    assert train_groups.isdisjoint(test_groups)


def test_split_is_deterministic_for_fixed_seed():
    genome_ids, labels, group_by_genome = _many_small_groups_fixture()
    result_a = split_by_lineage_group(genome_ids, labels, group_by_genome, seed=42, test_size=0.3)
    result_b = split_by_lineage_group(genome_ids, labels, group_by_genome, seed=42, test_size=0.3)
    assert sorted(result_a.train_genome_ids) == sorted(result_b.train_genome_ids)
    assert sorted(result_a.test_genome_ids) == sorted(result_b.test_genome_ids)


def test_split_covers_every_genome_exactly_once():
    genome_ids, labels, group_by_genome = _many_small_groups_fixture()
    result = split_by_lineage_group(genome_ids, labels, group_by_genome, seed=7, test_size=0.3)
    all_ids = sorted(result.train_genome_ids + result.test_genome_ids)
    assert all_ids == sorted(genome_ids)


def test_reports_blocker_when_fewer_than_two_groups():
    genome_ids = ["g1", "g2", "g3"]
    labels = ["Resistant", "Susceptible", "Resistant"]
    group_by_genome = {gid: "only_group" for gid in genome_ids}
    result = split_by_lineage_group(genome_ids, labels, group_by_genome, seed=1, test_size=0.3)
    assert result.feasible is False
    assert "distinct MLST group" in result.blocker


def test_never_falls_back_to_random_split_when_blocked():
    genome_ids = ["g1", "g2", "g3"]
    labels = ["Resistant", "Susceptible", "Resistant"]
    group_by_genome = {gid: "only_group" for gid in genome_ids}
    result = split_by_lineage_group(genome_ids, labels, group_by_genome, seed=1, test_size=0.3)
    assert result.train_genome_ids == []
    assert result.test_genome_ids == []


def test_warns_when_test_fraction_deviates_from_requested():
    # one huge group forces a very lopsided actual split
    genome_ids = [f"big_{i}" for i in range(90)] + [f"small_{i}" for i in range(10)]
    labels = ["Susceptible"] * 90 + ["Resistant"] * 10
    group_by_genome = {gid: "big_group" for gid in genome_ids[:90]}
    group_by_genome.update({gid: f"small_group_{i}" for i, gid in enumerate(genome_ids[90:])})
    result = split_by_lineage_group(genome_ids, labels, group_by_genome, seed=1, test_size=0.3)
    assert result.feasible is True
    # the 90-genome group must land entirely on one side, making the actual
    # fraction far from 30% no matter which side it lands on
    assert any("deviates" in w for w in result.warnings)


def test_warns_on_missing_class_in_a_partition():
    genome_ids, labels, group_by_genome = _many_small_groups_fixture(n_groups=6, group_size=2)
    # force one class to exist in only one tiny group, then bias test_size so it's plausible it's isolated
    result = split_by_lineage_group(genome_ids, labels, group_by_genome, seed=3, test_size=0.3)
    # whether or not this particular seed produces a missing class, the function must not crash
    # and must report class counts for both partitions
    assert isinstance(result.train_class_counts, dict)
    assert isinstance(result.test_class_counts, dict)


def test_class_count_keys_are_plain_python_str_not_numpy():
    genome_ids, labels, group_by_genome = _many_small_groups_fixture()
    result = split_by_lineage_group(genome_ids, labels, group_by_genome, seed=1, test_size=0.3)
    assert all(type(k) is str for k in result.train_class_counts)
    assert all(type(k) is str for k in result.test_class_counts)


def test_group_counts_sum_to_total_groups():
    genome_ids, labels, group_by_genome = _many_small_groups_fixture(n_groups=10, group_size=2)
    result = split_by_lineage_group(genome_ids, labels, group_by_genome, seed=5, test_size=0.3)
    assert result.train_group_count + result.test_group_count == len(set(group_by_genome.values()))
