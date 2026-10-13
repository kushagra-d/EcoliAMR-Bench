from bvbrc_recon.client import BVBRCQueryError
from bvbrc_recon.genome_join import fetch_genome_metadata
from fakes import FakeClient


def test_batches_requests_instead_of_one_per_genome():
    genome_ids = ["562.1", "562.2", "562.3", "562.4", "562.5"]
    client = FakeClient(
        {
            "genome": [
                ([{"genome_id": "562.1", "mlst": "ST1"}, {"genome_id": "562.2", "mlst": "ST2"}], 2),
                ([{"genome_id": "562.3", "mlst": "ST3"}, {"genome_id": "562.4", "mlst": "ST4"}], 2),
                ([{"genome_id": "562.5", "mlst": "ST5"}], 1),
            ]
        }
    )
    result = fetch_genome_metadata(client, genome_ids, batch_size=2)
    assert len(client.calls) == 3  # 3 batches, not 5 individual requests
    assert set(result.metadata_by_genome_id) == set(genome_ids)
    assert result.missing_genome_ids == []


def test_detects_genome_ids_missing_from_genome_collection():
    genome_ids = ["562.1", "562.2"]
    client = FakeClient(
        {"genome": [([{"genome_id": "562.1", "mlst": "ST1"}], 1)]}  # 562.2 not returned
    )
    result = fetch_genome_metadata(client, genome_ids, batch_size=5)
    assert result.missing_genome_ids == ["562.2"]
    assert "562.1" in result.metadata_by_genome_id


def test_batch_error_is_recorded_without_aborting_other_batches():
    genome_ids = ["562.1", "562.2", "562.3", "562.4"]
    client = FakeClient(
        {
            "genome": [
                BVBRCQueryError("simulated failure"),
                ([{"genome_id": "562.3"}, {"genome_id": "562.4"}], 2),
            ]
        }
    )
    result = fetch_genome_metadata(client, genome_ids, batch_size=2)
    assert len(result.errors) == 1
    assert "562.3" in result.metadata_by_genome_id
    assert "562.4" in result.metadata_by_genome_id


def test_deduplicates_input_genome_ids_before_batching():
    client = FakeClient({"genome": [([{"genome_id": "562.1"}], 1)]})
    result = fetch_genome_metadata(client, ["562.1", "562.1", "562.1"], batch_size=5)
    assert len(client.calls) == 1
    assert set(result.metadata_by_genome_id) == {"562.1"}
