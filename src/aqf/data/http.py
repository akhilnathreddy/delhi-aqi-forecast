"""Small HTTP helpers: retrying GETs, byte-range reads and anonymous S3 listing.

Everything in this project reads from public, unauthenticated endpoints
(AWS Open Data buckets). Each worker thread keeps its own keep-alive
``requests.Session``: re-using TLS connections is ~5x faster than opening
one per range request when fetching tens of thousands of small objects.
"""

from __future__ import annotations

import re
import threading
import time
import urllib.parse
from collections.abc import Iterator

import requests

USER_AGENT = "aqf/0.1 (+https://github.com/akhilnathreddy/delhi-aqi-forecast)"


_local = threading.local()


def _session() -> requests.Session:
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
        _local.s.headers["User-Agent"] = USER_AGENT
    return _local.s


def get(url: str, byte_range: tuple[int, int] | None = None, retries: int = 4, timeout: float = 60) -> bytes:
    """GET ``url`` (optionally an inclusive byte range) with exponential backoff.

    A 403/404 is returned as ``b""`` rather than raised: missing days and
    granules are normal in these archives and callers treat them as gaps.
    """
    headers = {"Range": f"bytes={byte_range[0]}-{byte_range[1]}"} if byte_range else {}
    for attempt in range(retries + 1):
        try:
            r = _session().get(url, headers=headers, timeout=timeout)
            if r.status_code in (403, 404):
                return b""
            r.raise_for_status()
            return r.content
        except requests.RequestException:
            if attempt == retries:
                raise
            _local.__dict__.pop("s", None)  # drop a possibly broken connection pool
        time.sleep(2**attempt)
    return b""  # unreachable


def s3_list(bucket: str, prefix: str, delimiter: str | None = None) -> Iterator[tuple[str, int]]:
    """Yield ``(key_or_prefix, size)`` for an anonymous S3 ListObjectsV2 call.

    With a ``delimiter`` the common prefixes are yielded with ``size == -1``.
    """
    token = None
    while True:
        q = {"list-type": "2", "prefix": prefix}
        if delimiter:
            q["delimiter"] = delimiter
        if token:
            q["continuation-token"] = token
        body = get(f"https://{bucket}.s3.amazonaws.com/?{urllib.parse.urlencode(q)}").decode()
        for key, size in re.findall(r"<Key>([^<]*)</Key>.*?<Size>(\d+)</Size>", body):
            yield key, int(size)
        for p in re.findall(r"<CommonPrefixes><Prefix>([^<]*)</Prefix>", body):
            yield p, -1
        m = re.search(r"<NextContinuationToken>([^<]*)</NextContinuationToken>", body)
        if not m:
            return
        token = m.group(1)


def s3_url(bucket: str, key: str) -> str:
    return f"https://{bucket}.s3.amazonaws.com/{urllib.parse.quote(key)}"
