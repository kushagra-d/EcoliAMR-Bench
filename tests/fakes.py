"""Test doubles shared across the bvbrc_recon test suite.

FakeClient duck-types BVBRCClient.query(collection, rql) -> (records, content_range)
so pagination/genome_join/pipeline tests never touch the network.
"""

from __future__ import annotations

from bvbrc_recon.client import BVBRCQueryError, ContentRange


class FakeClient:
    """Replays a pre-scripted sequence of responses per collection.

    ``script[collection]`` is a list consumed in order, one entry per
    call to that collection. Each entry is either:
      - (records: list[dict], total: int | None) -> succeeds
      - an Exception instance -> raised as-is (wrapped, if not already
        a BVBRCQueryError, so callers see the same error type the real
        client raises for non-retryable failures)
    """

    def __init__(self, script: dict[str, list]):
        self.script = {k: list(v) for k, v in script.items()}
        self.calls: list[tuple[str, str]] = []

    def query(self, collection: str, rql: str):
        self.calls.append((collection, rql))
        queue = self.script.get(collection, [])
        if not queue:
            raise AssertionError(f"FakeClient: no more scripted responses for {collection!r}")
        entry = queue.pop(0)
        if isinstance(entry, Exception):
            if isinstance(entry, BVBRCQueryError):
                raise entry
            raise BVBRCQueryError(str(entry)) from entry
        records, total = entry
        content_range = None
        if total is not None:
            content_range = ContentRange(start=0, end=max(len(records) - 1, 0), total=total)
        return records, content_range
