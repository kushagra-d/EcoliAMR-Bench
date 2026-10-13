import pandas as pd
import pytest

from bvbrc_recon.conflict_pipeline import analyze_antibiotic, main


def _amr_row(genome_id, phenotype, **kwargs):
    row = {
        "id": f"{genome_id}-{phenotype}-{kwargs.get('testing_standard', 'EUCAST')}",
        "genome_id": genome_id,
        "genome_name": f"Escherichia coli {genome_id}",
        "antibiotic": "ciprofloxacin",
        "resistant_phenotype": phenotype,
        "measurement": "",
        "measurement_sign": "",
        "measurement_value": "",
        "measurement_unit": "",
        "laboratory_typing_method": "Disk diffusion",
        "laboratory_typing_platform": "",
        "testing_standard": "EUCAST",
        "testing_standard_year": 2021,
        "evidence": "Laboratory Method",
        "source": "",
        "pmid": "[]",
    }
    row.update(kwargs)
    return row


@pytest.fixture
def synthetic_csv_dir(tmp_path):
    rows = [
        _amr_row("g1", "Resistant"),
        _amr_row("g2", "Susceptible"),
        _amr_row("g2", ""),  # Resistant+blank regression case: NOT a real conflict
        _amr_row("g3", "Resistant"),
        _amr_row("g3", "Susceptible"),  # true categorical conflict
    ]
    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(data_dir / "ciprofloxacin_amr_metadata.csv", index=False)
    return data_dir


def test_analyze_antibiotic_reconciles_old_and_new_conflict_counts(synthetic_csv_dir):
    from bvbrc_recon.csv_loader import load_antibiotic_csv

    records = load_antibiotic_csv(synthetic_csv_dir / "ciprofloxacin_amr_metadata.csv")
    report = analyze_antibiotic("ciprofloxacin", records)

    # old broad definition counts BOTH g2 (Resistant+blank... wait Susceptible+blank) and g3
    assert report.old_broad_conflict_count == 2
    # the stricter, audited definition counts only the genuine disagreement (g3)
    assert report.audit.conflicting_genome_ids == ["g3"]


def test_main_writes_report_and_summary_csv_without_touching_other_paths(tmp_path, synthetic_csv_dir):
    report_path = tmp_path / "out_reports" / "phenotype_conflict_analysis.md"
    summary_csv_path = tmp_path / "out_reports" / "policy_summary.csv"

    reports = main(
        data_dir=synthetic_csv_dir,
        report_path=report_path,
        summary_csv_path=summary_csv_path,
        antibiotics=["ciprofloxacin"],
    )

    assert report_path.exists()
    assert summary_csv_path.exists()
    assert "ciprofloxacin" in reports

    summary_df = pd.read_csv(summary_csv_path)
    assert set(summary_df["genome_id"]) == {"g1", "g2", "g3"}
    assert "policy1_outcome" in summary_df.columns
    assert "policy2_outcome" in summary_df.columns

    report_text = report_path.read_text(encoding="utf-8")
    assert "Ciprofloxacin" in report_text
    assert "Task C" in report_text

    # the original synthetic input CSV must not have been modified
    original = pd.read_csv(synthetic_csv_dir / "ciprofloxacin_amr_metadata.csv")
    assert len(original) == 5


def test_main_does_not_require_summary_csv(tmp_path, synthetic_csv_dir):
    report_path = tmp_path / "out_reports" / "report.md"
    main(
        data_dir=synthetic_csv_dir,
        report_path=report_path,
        summary_csv_path=None,
        antibiotics=["ciprofloxacin"],
    )
    assert report_path.exists()
