from bvbrc_recon.client import BVBRCQueryError
from bvbrc_recon.pagination import paginate
from fakes import FakeClient


SELECT = ["id", "genome_id", "resistant_phenotype"]


def test_single_short_page_is_complete():
    client = FakeClient({"genome_amr": [([{"id": "a"}], 1)]})
    result = paginate(client, "genome_amr", "eq(x,y)", SELECT, page_size=2)
    assert [r["id"] for r in result.records] == ["a"]
    assert result.reported_total_first == 1
    assert result.reported_total_last == 1
    assert result.complete is True
    assert result.duplicate_record_ids == []


def test_multi_page_exact_boundary_stops_via_total():
    client = FakeClient(
        {
            "genome_amr": [
                ([{"id": "a"}, {"id": "b"}], 4),
                ([{"id": "c"}, {"id": "d"}], 4),
            ]
        }
    )
    result = paginate(client, "genome_amr", "eq(x,y)", SELECT, page_size=2)
    assert [r["id"] for r in result.records] == ["a", "b", "c", "d"]
    assert result.complete is True
    assert len(client.calls) == 2


def test_duplicate_id_across_pages_is_detected_and_not_double_counted():
    client = FakeClient(
        {
            "genome_amr": [
                ([{"id": "a"}, {"id": "b"}], 3),
                ([{"id": "b"}, {"id": "c"}], 3),  # "b" repeated: pagination overlap
            ]
        }
    )
    result = paginate(client, "genome_amr", "eq(x,y)", SELECT, page_size=2)
    assert [r["id"] for r in result.records] == ["a", "b", "c"]
    assert result.duplicate_record_ids == ["b"]
    assert result.complete is True  # unique count (3) matches reported total (3)


def test_total_drift_is_flagged_but_does_not_crash():
    client = FakeClient(
        {
            "genome_amr": [
                ([{"id": "a"}, {"id": "b"}], 5),
                ([{"id": "c"}, {"id": "d"}], 4),  # total shrank mid-retrieval
            ]
        }
    )
    result = paginate(client, "genome_amr", "eq(x,y)", SELECT, page_size=2)
    assert result.reported_total_first == 5
    assert result.reported_total_last == 4
    assert result.total_drifted is True


def test_page_failure_stops_pagination_and_marks_incomplete():
    client = FakeClient(
        {
            "genome_amr": [
                ([{"id": "a"}, {"id": "b"}], 5),
                BVBRCQueryError("simulated 503 after retries"),
            ]
        }
    )
    result = paginate(client, "genome_amr", "eq(x,y)", SELECT, page_size=2)
    assert [r["id"] for r in result.records] == ["a", "b"]
    assert len(result.errors) == 1
    assert result.complete is False


def test_no_content_range_header_means_completeness_cannot_be_claimed():
    client = FakeClient({"genome_amr": [([{"id": "a"}], None)]})
    result = paginate(client, "genome_amr", "eq(x,y)", SELECT, page_size=2)
    assert result.reported_total_last is None
    assert result.complete is False
