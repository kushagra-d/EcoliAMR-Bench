import pandas as pd

from bvbrc_recon.pipeline import main, run_antibiotic
from fakes import FakeClient


def _amr_record(record_id, genome_id, phenotype):
    return {
        "id": record_id,
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
        "pmid": [],
    }


def test_run_antibiotic_merges_genome_metadata_and_tracks_missing():
    client = FakeClient(
        {
            "genome_amr": [
                (
                    [
                        _amr_record("r1", "562.1", "Resistant"),
                        _amr_record("r2", "562.2", "Susceptible"),
                    ],
                    2,
                )
            ],
            "genome": [
                (
                    [{"genome_id": "562.1", "mlst": "ST131", "contigs": 80}],
                    1,
                )  # 562.2 deliberately missing from genome collection
            ],
        }
    )
    result = run_antibiotic(client, "ciprofloxacin", page_size=10, genome_batch_size=10)

    assert len(result.merged_rows) == 2
    row_1 = next(r for r in result.merged_rows if r["genome_id"] == "562.1")
    assert row_1["genome_mlst"] == "ST131"
    assert row_1["genome_contigs"] == 80

    row_2 = next(r for r in result.merged_rows if r["genome_id"] == "562.2")
    assert "genome_mlst" not in row_2  # no genome record was found for 562.2

    assert result.genome_join.missing_genome_ids == ["562.2"]
    assert result.phenotype_counts["Resistant"] == 1
    assert result.phenotype_counts["Susceptible"] == 1


def test_main_writes_one_csv_per_antibiotic_and_a_report(tmp_path):
    antibiotics = ["ciprofloxacin", "ampicillin"]
    client = FakeClient(
        {
            "genome_amr": [
                ([_amr_record("r1", "562.1", "Resistant")], 1),
                ([_amr_record("r2", "562.2", "Susceptible")], 1),
            ],
            "genome": [
                ([{"genome_id": "562.1", "mlst": "ST131"}], 1),
                ([{"genome_id": "562.2", "mlst": "ST73"}], 1),
            ],
        }
    )
    out_data = tmp_path / "data" / "processed"
    out_reports = tmp_path / "reports"

    results = main(
        out_data_dir=out_data, out_report_dir=out_reports, client=client, antibiotics=antibiotics
    )

    assert set(results) == set(antibiotics)
    for antibiotic in antibiotics:
        csv_path = out_data / f"{antibiotic}_amr_metadata.csv"
        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        assert "resistant_phenotype" in df.columns
        assert len(df) == 1

    report_path = out_reports / "bvbrc_feasibility_report.md"
    assert report_path.exists()
    report_text = report_path.read_text(encoding="utf-8")
    assert "Ciprofloxacin" in report_text
    assert "Ampicillin" in report_text
    assert "no genome sequence data was downloaded" in report_text.lower()


def test_main_does_not_write_any_fasta_or_sequence_files(tmp_path):
    client = FakeClient(
        {
            "genome_amr": [([_amr_record("r1", "562.1", "Resistant")], 1)],
            "genome": [([{"genome_id": "562.1", "mlst": "ST131"}], 1)],
        }
    )
    out_data = tmp_path / "data" / "processed"
    main(out_data_dir=out_data, out_report_dir=tmp_path / "reports", client=client, antibiotics=["ciprofloxacin"])

    written_files = list(out_data.rglob("*"))
    assert all(f.suffix == ".csv" for f in written_files if f.is_file())
