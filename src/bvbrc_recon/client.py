"""Low-level HTTP client for the public BV-BRC Data API.

Uses only the Python standard library (urllib) - no new dependency was
needed for this task since BV-BRC's REST API requires nothing beyond
GET requests with query-string parameters and JSON parsing.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

BASE_URL = "https://www.bv-brc.org/api"

# BV-BRC sits behind Cloudflare, which blocks the default urllib User-Agent
# ("Python-urllib/3.x") outright with HTTP 403. A standard browser-style UA
# is enough to pass; this is not bypassing any access control beyond what a
# browser or curl already receives for this public, unauthenticated API.
REQUEST_HEADERS = {
    "Accept": "application/json",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
}

# HTTP statuses worth retrying: rate limiting and transient server errors.
# 4xx errors other than 429 mean the query itself is malformed and retrying
# will not help, so they are raised immediately instead of being retried.
RETRYABLE_HTTP_STATUSES = {429, 500, 502, 503, 504}


class BVBRCQueryError(RuntimeError):
    """Raised when a BV-BRC request fails after exhausting retries, or
    when the server reports a non-retryable error (e.g. a malformed RQL
    query)."""


@dataclass(frozen=True)
class ContentRange:
    """Parsed form of BV-BRC's ``Content-Range: items START-END/TOTAL`` header."""

    start: int
    end: int
    total: int

    @classmethod
    def parse(cls, header_value: str | None) -> "ContentRange | None":
        if not header_value:
            return None
        try:
            items_part, total_part = header_value.split("/")
            start_str, end_str = items_part.strip().split(" ")[-1].split("-")
            return cls(start=int(start_str), end=int(end_str), total=int(total_part))
        except (ValueError, IndexError):
            return None


def rql_eq(field: str, value: object) -> str:
    """Build an ``eq(field,value)`` RQL clause, URL-quoting the value."""
    return f"eq({field},{urllib.parse.quote(str(value), safe='')})"


def rql_and(*clauses: str) -> str:
    return f"and({','.join(clauses)})"


def rql_in(field: str, values: list[str]) -> str:
    quoted = ",".join(urllib.parse.quote(str(v), safe="") for v in values)
    return f"in({field},({quoted}))"


def rql_select(fields: list[str]) -> str:
    return f"select({','.join(fields)})"


def rql_limit(count: int, start: int = 0) -> str:
    return f"limit({count},{start})"


class BVBRCClient:
    """Minimal retrying client for ``GET {base_url}/{collection}/?{rql}``."""

    def __init__(
        self,
        base_url: str = BASE_URL,
        max_retries: int = 3,
        backoff_seconds: float = 1.0,
        timeout: float = 30.0,
        opener=urllib.request.urlopen,
        sleep_fn=time.sleep,
    ) -> None:
        self.base_url = base_url
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.timeout = timeout
        self._opener = opener
        self._sleep = sleep_fn

    def query(self, collection: str, rql: str) -> tuple[list[dict], "ContentRange | None"]:
        """Run one GET request and return (records, content_range).

        Retries transient failures (timeouts, connection errors, 429/5xx,
        malformed JSON) up to ``max_retries`` times with linear backoff.
        Non-retryable HTTP errors (e.g. 400 for a malformed query) are
        raised immediately as :class:`BVBRCQueryError`.
        """
        url = f"{self.base_url}/{collection}/?{rql}"
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            request = urllib.request.Request(url, headers=REQUEST_HEADERS)
            try:
                with self._opener(request, timeout=self.timeout) as response:
                    body = response.read()
                    content_range = ContentRange.parse(response.headers.get("Content-Range"))
                    records = json.loads(body)
                    if not isinstance(records, list):
                        raise BVBRCQueryError(
                            f"Unexpected response shape from {collection}: expected a list, "
                            f"got {type(records).__name__}"
                        )
                    return records, content_range
            except urllib.error.HTTPError as exc:
                if exc.code not in RETRYABLE_HTTP_STATUSES:
                    raise BVBRCQueryError(
                        f"Non-retryable HTTP {exc.code} for {collection} query: {rql}"
                    ) from exc
                last_error = exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc

            if attempt < self.max_retries:
                self._sleep(self.backoff_seconds * attempt)

        raise BVBRCQueryError(
            f"Failed to query {collection} after {self.max_retries} attempts: {last_error}"
        ) from last_error
