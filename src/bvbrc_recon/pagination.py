"""Safe pagination over a BV-BRC collection, with completeness bookkeeping.

BV-BRC enforces a hard per-request cap (25,000 records) and is a live,
continuously-updated database, so a multi-page pull can run into: the
reported total changing mid-retrieval, a page failing outright, or the
same record id appearing on two pages (pagination overlap). This module
surfaces all three rather than silently producing a possibly-incomplete
result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .client import BVBRCClient, BVBRCQueryError, rql_limit


@dataclass
class PaginationResult:
    records: list[dict] = field(default_factory=list)
    reported_total_first: int | None = None
    reported_total_last: int | None = None
    duplicate_record_ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def total_drifted(self) -> bool:
        return (
            self.reported_total_first is not None
            and self.reported_total_last is not None
            and self.reported_total_first != self.reported_total_last
        )

    @property
    def complete(self) -> bool:
        """True only if every page succeeded and the unique record count
        matches the most recently reported total. Does not assume
        completeness in the absence of this evidence."""
        if self.errors:
            return False
        if self.reported_total_last is None:
            return False
        return len(self.records) == self.reported_total_last


def paginate(
    client: BVBRCClient,
    collection: str,
    rql_filter: str,
    select_fields: list[str],
    id_field: str = "id",
    page_size: int = 1000,
) -> PaginationResult:
    """Fetch every record matching ``rql_filter`` from ``collection``.

    ``rql_filter`` must NOT include select()/limit() - those are added here.
    Stops when a page returns fewer than ``page_size`` records, the
    reported total has been reached, or a page fails after retries.
    """
    result = PaginationResult()
    seen_ids: set[str] = set()
    select_clause = f"select({','.join(select_fields)})"
    start = 0

    while True:
        rql = f"{rql_filter}&{select_clause}&{rql_limit(page_size, start)}"
        try:
            page_records, content_range = client.query(collection, rql)
        except BVBRCQueryError as exc:
            result.errors.append(f"page at start={start} failed: {exc}")
            break

        if content_range is not None:
            if result.reported_total_first is None:
                result.reported_total_first = content_range.total
            result.reported_total_last = content_range.total

        for record in page_records:
            record_id = record.get(id_field)
            if record_id is not None and record_id in seen_ids:
                result.duplicate_record_ids.append(record_id)
                continue
            if record_id is not None:
                seen_ids.add(record_id)
            result.records.append(record)

        if len(page_records) < page_size:
            break
        start += page_size
        if result.reported_total_last is not None and start >= result.reported_total_last:
            break

    return result
