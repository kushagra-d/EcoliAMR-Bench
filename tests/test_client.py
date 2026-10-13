import json
import urllib.error

import pytest

from bvbrc_recon.client import (
    BVBRCClient,
    BVBRCQueryError,
    ContentRange,
    rql_and,
    rql_eq,
    rql_in,
    rql_limit,
    rql_select,
)


# --- RQL builders -----------------------------------------------------


def test_rql_eq_quotes_value_with_space():
    assert rql_eq("evidence", "Laboratory Method") == "eq(evidence,Laboratory%20Method)"


def test_rql_eq_accepts_int():
    assert rql_eq("taxon_id", 562) == "eq(taxon_id,562)"


def test_rql_and_joins_clauses():
    assert rql_and("a", "b", "c") == "and(a,b,c)"


def test_rql_in_quotes_each_value():
    assert rql_in("genome_id", ["562.1", "562.2"]) == "in(genome_id,(562.1,562.2))"


def test_rql_select_and_limit():
    assert rql_select(["a", "b"]) == "select(a,b)"
    assert rql_limit(100, 50) == "limit(100,50)"


# --- ContentRange parsing ----------------------------------------------


def test_content_range_parses_valid_header():
    cr = ContentRange.parse("items 10-15/13317")
    assert cr == ContentRange(start=10, end=15, total=13317)


@pytest.mark.parametrize("value", [None, "", "garbage", "items 10-15"])
def test_content_range_returns_none_for_invalid_header(value):
    assert ContentRange.parse(value) is None


# --- BVBRCClient.query: fakes for urllib -------------------------------


class FakeResponse:
    def __init__(self, body: bytes, headers: dict):
        self._body = body
        self._headers = headers

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body

    @property
    def headers(self):
        return self

    def get(self, name, default=None):
        return self._headers.get(name, default)


class ScriptedOpener:
    """Pops one scripted result per call; a result is a FakeResponse or an exception."""

    def __init__(self, results):
        self.results = list(results)
        self.calls = 0

    def __call__(self, request, timeout=None):
        self.calls += 1
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def make_json_response(records, total=None):
    headers = {}
    if total is not None:
        headers["Content-Range"] = f"items 0-{max(len(records) - 1, 0)}/{total}"
    return FakeResponse(json.dumps(records).encode("utf-8"), headers)


def test_query_succeeds_on_first_try():
    opener = ScriptedOpener([make_json_response([{"id": "1"}], total=1)])
    client = BVBRCClient(opener=opener, sleep_fn=lambda s: None)
    records, content_range = client.query("genome_amr", "eq(a,b)")
    assert records == [{"id": "1"}]
    assert content_range == ContentRange(start=0, end=0, total=1)
    assert opener.calls == 1


def test_query_retries_transient_url_error_then_succeeds():
    opener = ScriptedOpener(
        [
            urllib.error.URLError("connection reset"),
            make_json_response([{"id": "1"}], total=1),
        ]
    )
    sleeps = []
    client = BVBRCClient(opener=opener, sleep_fn=sleeps.append, max_retries=3)
    records, _ = client.query("genome_amr", "eq(a,b)")
    assert records == [{"id": "1"}]
    assert opener.calls == 2
    assert sleeps  # backoff was invoked


def test_query_retries_retryable_http_status_then_raises_after_exhausting():
    error = urllib.error.HTTPError("url", 503, "Service Unavailable", {}, None)
    opener = ScriptedOpener([error, error, error])
    client = BVBRCClient(opener=opener, sleep_fn=lambda s: None, max_retries=3)
    with pytest.raises(BVBRCQueryError):
        client.query("genome_amr", "eq(a,b)")
    assert opener.calls == 3


def test_query_raises_immediately_on_non_retryable_http_status():
    error = urllib.error.HTTPError("url", 400, "Bad Request", {}, None)
    opener = ScriptedOpener([error])
    client = BVBRCClient(opener=opener, sleep_fn=lambda s: None, max_retries=5)
    with pytest.raises(BVBRCQueryError):
        client.query("genome_amr", "eq(bad_field,x)")
    assert opener.calls == 1  # no retry for a malformed-query error


def test_query_retries_malformed_json_then_succeeds():
    bad = FakeResponse(b"not json", {})
    opener = ScriptedOpener([bad, make_json_response([], total=0)])
    client = BVBRCClient(opener=opener, sleep_fn=lambda s: None, max_retries=3)
    records, content_range = client.query("genome_amr", "eq(a,b)")
    assert records == []
    assert content_range.total == 0


def test_query_raises_on_unexpected_response_shape():
    not_a_list = FakeResponse(json.dumps({"error": "nope"}).encode("utf-8"), {})
    opener = ScriptedOpener([not_a_list])
    client = BVBRCClient(opener=opener, sleep_fn=lambda s: None, max_retries=3)
    with pytest.raises(BVBRCQueryError):
        client.query("genome_amr", "eq(a,b)")
    assert opener.calls == 1  # not retried - this is not a transient failure
