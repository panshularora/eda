"""A deliberately polite HTTP layer.

Every endpoint in this project is someone else's public service, and none of
them owe us anything. This module is the single place where that obligation is
honoured, so no collector can accidentally hammer a host:

* one identifying User-Agent with a contact address on every request;
* a per-host minimum interval, enforced globally across all collectors;
* exponential backoff that *respects* ``Retry-After`` on 429 and 503;
* an on-disk cache, so re-running the pipeline during development replays from
  disk instead of re-requesting;
* a request ledger, so the report can state exactly how many calls were made
  to whom, and how many failed.

Nothing here is specific to the topic - it is infrastructure.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from config import BACKOFF_BASE, MAX_RETRIES, RATE_LIMITS, RAW, USER_AGENT

CACHE = RAW / "_httpcache"
CACHE.mkdir(parents=True, exist_ok=True)


@dataclass
class Ledger:
    """Counts of what we asked of whom - quoted verbatim in the report."""

    requests: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    failures: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    cached: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    bytes_: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def summary(self) -> dict:
        hosts = sorted(set(self.requests) | set(self.cached))
        return {
            "hosts": {
                h: {
                    "live_requests": self.requests.get(h, 0),
                    "cache_hits": self.cached.get(h, 0),
                    "failures": self.failures.get(h, 0),
                    "bytes": self.bytes_.get(h, 0),
                }
                for h in hosts
            },
            "total_live_requests": sum(self.requests.values()),
            "total_cache_hits": sum(self.cached.values()),
            "total_failures": sum(self.failures.values()),
            "total_bytes": sum(self.bytes_.values()),
        }


LEDGER = Ledger()
_last_hit: dict[str, float] = {}


def _host(url: str) -> str:
    return urllib.parse.urlparse(url).netloc.lower()


def _throttle(host: str) -> None:
    """Block until this host's minimum interval has elapsed."""
    wait = RATE_LIMITS.get(host, RATE_LIMITS["default"])
    last = _last_hit.get(host)
    if last is not None:
        sleep_for = wait - (time.monotonic() - last)
        if sleep_for > 0:
            time.sleep(sleep_for)
    _last_hit[host] = time.monotonic()


def _cache_path(url: str) -> Path:
    return CACHE / (hashlib.sha256(url.encode()).hexdigest()[:32] + ".gz")


def get(url: str, *, timeout: int = 30, use_cache: bool = True,
        max_age: float | None = None,
        max_attempts: int | None = None) -> bytes | None:
    """Fetch a URL politely. Returns ``None`` if it could not be retrieved.

    Returning None rather than raising is deliberate: one dead feed among forty
    should degrade the dataset, not abort the collection. Every failure is
    counted in the ledger and surfaced in the source audit.

    ``max_attempts`` exists because politeness and progress can conflict.
    Reddit rate-limits unpredictably - the same query returns 429 twice and 200
    a few seconds later - and with four retries, a nine-second host interval
    and a backoff capped at sixty seconds, a single URL that was never going to
    answer can consume four and a half minutes. Forty-three of those is a
    pipeline that never finishes. Callers that know a host behaves this way
    lower the ceiling and take the smaller dataset, which the audit then
    reports honestly rather than hiding behind a long wait.
    """
    host = _host(url)
    path = _cache_path(url)
    if use_cache and path.exists():
        fresh = max_age is None or (time.time() - path.stat().st_mtime) < max_age
        if fresh:
            LEDGER.cached[host] += 1
            return gzip.decompress(path.read_bytes())

    attempts = max_attempts or MAX_RETRIES
    for attempt in range(attempts):
        _throttle(host)
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": USER_AGENT,
                "Accept": "*/*",
                "Accept-Language": "en",
            })
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read()
            LEDGER.requests[host] += 1
            LEDGER.bytes_[host] += len(body)
            if use_cache:
                path.write_bytes(gzip.compress(body))
            return body
        except urllib.error.HTTPError as e:
            # 4xx that is not rate limiting will not fix itself; stop early.
            if e.code in (429, 503):
                retry_after = e.headers.get("Retry-After") if e.headers else None
                delay = float(retry_after) if (retry_after or "").isdigit() else \
                    BACKOFF_BASE ** (attempt + 1)
                time.sleep(min(delay, 60))
                continue
            if 400 <= e.code < 500:
                LEDGER.failures[host] += 1
                return None
            time.sleep(BACKOFF_BASE ** attempt)
        except Exception:
            time.sleep(BACKOFF_BASE ** attempt)
    LEDGER.failures[host] += 1
    return None


def get_json(url: str, **kw):
    raw = get(url, **kw)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def write_jsonl(records: list[dict], path: Path) -> Path:
    """Append-safe newline-delimited JSON - the raw landing format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    return path


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def anonymise(handle: str | None) -> str:
    """Stable pseudonym for an author.

    We study *what was said and when*, never *who said it*. Raw handles are
    replaced at the point of collection with a salted hash, so no identifier
    ever reaches the published dataset, while repeat posters remain countable.
    """
    if not handle:
        return ""
    return "u_" + hashlib.sha256(f"datavortex-se7en::{handle}".encode()).hexdigest()[:16]
