import pytest

from bvbrc_recon.pilot_cohort import select_pilot_cohort


def rec(genome_id, phenotype, antibiotic="ciprofloxacin"):
    return {"genome_id": genome_id, "resistant_phenotype": phenotype, "antibiotic": antibiotic}


def _synthetic_population():
    records = []
    # 6 pure-Resistant genomes
    for i in range(6):
        records.append(rec(f"r{i}", "Resistant"))
    # 12 pure-Susceptible genomes
    for i in range(12):
        records.append(rec(f"s{i}", "Susceptible"))
    # 2 Intermediate-only genomes (excluded)
    for i in range(2):
        records.append(rec(f"i{i}", "Intermediate"))
    # 2 conflicting genomes (excluded)
    for i in range(2):
        records.append(rec(f"c{i}", "Resistant"))
        records.append(rec(f"c{i}", "Susceptible"))
    # 2 missing-only genomes (excluded)
    for i in range(2):
        records.append(rec(f"m{i}", ""))
    return records


def test_select_pilot_cohort_excludes_intermediate_conflicting_and_missing():
    cohort = select_pilot_cohort(_synthetic_population(), "ciprofloxacin", target_size=100, seed=1)
    selected = set(cohort.selected_genome_ids)
    assert not selected & {"i0", "i1"}
    assert not selected & {"c0", "c1"}
    assert not selected & {"m0", "m1"}


def test_select_pilot_cohort_uses_entire_pool_when_smaller_than_target():
    cohort = select_pilot_cohort(_synthetic_population(), "ciprofloxacin", target_size=100, seed=1)
    # eligible pool is 6 Resistant + 12 Susceptible = 18, far below target 100
    assert len(cohort.selected_genome_ids) == 18
    assert cohort.n_selected_resistant == 6
    assert cohort.n_selected_susceptible == 12
    assert any("Only 18 genomes" in note for note in cohort.limitations)


def test_select_pilot_cohort_includes_both_classes():
    cohort = select_pilot_cohort(_synthetic_population(), "ciprofloxacin", target_size=10, seed=1)
    labels = set(cohort.selected_labels.values())
    assert "Resistant" in labels
    assert "Susceptible" in labels


def test_select_pilot_cohort_is_reproducible_with_fixed_seed():
    cohort_a = select_pilot_cohort(_synthetic_population(), "ciprofloxacin", target_size=10, seed=42)
    cohort_b = select_pilot_cohort(_synthetic_population(), "ciprofloxacin", target_size=10, seed=42)
    assert sorted(cohort_a.selected_genome_ids) == sorted(cohort_b.selected_genome_ids)


def test_select_pilot_cohort_different_seeds_can_differ():
    pop = []
    for i in range(50):
        pop.append(rec(f"r{i}", "Resistant"))
    for i in range(50):
        pop.append(rec(f"s{i}", "Susceptible"))
    cohort_a = select_pilot_cohort(pop, "ciprofloxacin", target_size=20, seed=1)
    cohort_b = select_pilot_cohort(pop, "ciprofloxacin", target_size=20, seed=2)
    assert sorted(cohort_a.selected_genome_ids) != sorted(cohort_b.selected_genome_ids)


def test_select_pilot_cohort_decision_table_covers_every_genome():
    records = _synthetic_population()
    cohort = select_pilot_cohort(records, "ciprofloxacin", target_size=100, seed=1)
    n_unique_genomes = len({r["genome_id"] for r in records})
    assert len(cohort.genome_decision_table) == n_unique_genomes


def test_select_pilot_cohort_rejects_mixed_antibiotics():
    records = [rec("g1", "Resistant", antibiotic="ciprofloxacin"), rec("g2", "Resistant", antibiotic="ampicillin")]
    with pytest.raises(ValueError):
        select_pilot_cohort(records, "ciprofloxacin", target_size=10, seed=1)


def test_select_pilot_cohort_reports_blocker_when_pool_is_empty():
    records = [rec("i0", "Intermediate"), rec("m0", "")]
    cohort = select_pilot_cohort(records, "ciprofloxacin", target_size=10, seed=1)
    assert cohort.selected_genome_ids == []
    assert any("BLOCKER" in note for note in cohort.limitations)


def test_select_pilot_cohort_never_includes_conflicting_genome_even_if_space_allows():
    pop = [rec("r0", "Resistant"), rec("c0", "Resistant"), rec("c0", "Susceptible")]
    cohort = select_pilot_cohort(pop, "ciprofloxacin", target_size=100, seed=1)
    assert "c0" not in cohort.selected_genome_ids
