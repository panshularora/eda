"""Append-only multi-wave collection: what is actually new since last time.

Why this exists
---------------
The rulebook asks for *real-time* data and the detection of *evolving*
conversations. A single ten-minute pull, however deep, is a retrospective
snapshot: it reconstructs the past 45 days from whatever the platforms still
hold today. That is a legitimate way to build a baseline and an illegitimate
way to claim live monitoring, and the distinction is worth being explicit
about rather than blurring.

So collection runs in **waves**. Each wave is a full run, appended to a
per-source archive keyed on `record_id`. What the wave *adds* - the records no
previous wave had seen - is the genuinely live increment, and it is the only
thing this module lets the report call live.

What a wave measures that a snapshot cannot
-------------------------------------------
* **arrival latency** - the gap between a reaction being written and our seeing
  it. For Play reviews this is near zero for the newest rows and days for the
  older ones; for Reddit it is minutes. Quantifying it is the difference
  between "we monitor" and "we could monitor".
* **backfill** - records dated *before* the previous wave that only appeared
  now. Play moderates and releases reviews with a lag, so yesterday's numbers
  keep changing after yesterday. A monitoring system that reports a daily
  figure without knowing its backfill rate is reporting a figure that will move
  underneath it, and the size of that effect is measured here rather than
  assumed away.
* **revision** - whether a metric computed on wave N survives on wave N+1.

Usage
-----
    python round3/src/waves.py            # run a wave now
    python round3/src/waves.py --report   # summarise the waves on disk

Each wave writes `data/waves/wave_<n>_<timestamp>.json` with its own counts and
appends new rows to `data/waves/archive_<source>.jsonl`.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "sources"))

from config import DATA, RAW                                  # noqa: E402
from fetch import read_jsonl, write_jsonl                     # noqa: E402

WAVES = DATA / "waves"
WAVES.mkdir(parents=True, exist_ok=True)

SOURCES = ("play_reviews", "reddit", "lemmy", "mastodon", "news",
           "news_brand", "hackernews", "incidents")


def _key(row: dict, source: str) -> str:
    native = str(row.get("record_native_id") or row.get("url") or "")
    text = str(row.get("text") or row.get("title") or "")[:160]
    return hashlib.sha256(f"{source}|{native}|{text}".encode()).hexdigest()[:20]


def _seen(source: str) -> set[str]:
    path = WAVES / f"archive_{source}.jsonl"
    if not path.exists():
        return set()
    return {json.loads(line).get("_wave_key", "")
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _parse(ts: str):
    if not ts:
        return None
    try:
        import pandas as pd
        v = pd.to_datetime(ts, utc=True, errors="coerce")
        return None if pd.isna(v) else v
    except Exception:
        return None


def record_wave(label: str | None = None) -> dict:
    """Diff the current `data/raw` against everything previous waves archived."""
    now = datetime.now(timezone.utc)
    existing = sorted(WAVES.glob("wave_*.json"))
    wave_n = len(existing) + 1
    summary: dict = {
        "wave": wave_n,
        "label": label or f"wave {wave_n}",
        "collected_utc": now.isoformat(timespec="seconds"),
        "by_source": {},
    }

    for source in SOURCES:
        path = RAW / f"{source}.jsonl"
        if not path.exists():
            continue
        rows = read_jsonl(path)
        seen = _seen(source)
        fresh = []
        for r in rows:
            k = _key(r, source)
            if k in seen:
                continue
            r = dict(r)
            r["_wave_key"] = k
            r["_wave"] = wave_n
            r["_first_seen_utc"] = now.isoformat(timespec="seconds")
            fresh.append(r)

        stat = {"in_raw": len(rows), "new_this_wave": len(fresh),
                "already_seen": len(rows) - len(fresh)}

        if fresh and wave_n > 1:
            lats, backfill = [], 0
            prev_stamp = None
            if existing:
                try:
                    prev = json.loads(existing[-1].read_text(encoding="utf-8"))
                    prev_stamp = _parse(prev.get("collected_utc", ""))
                except Exception:
                    prev_stamp = None
            for r in fresh:
                created = _parse(str(r.get("created_utc", "")))
                if created is None:
                    continue
                lats.append((now - created).total_seconds() / 3600.0)
                if prev_stamp is not None and created < prev_stamp:
                    backfill += 1
            if lats:
                lats.sort()
                stat["arrival_latency_hours"] = {
                    "median": round(lats[len(lats) // 2], 2),
                    "p10": round(lats[int(len(lats) * 0.10)], 2),
                    "p90": round(lats[int(len(lats) * 0.90)], 2),
                }
                stat["backfilled_before_previous_wave"] = backfill
                stat["backfill_share"] = round(backfill / len(lats), 3)

        summary["by_source"][source] = stat

        if fresh:
            arch = WAVES / f"archive_{source}.jsonl"
            with arch.open("a", encoding="utf-8") as fh:
                for r in fresh:
                    fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    summary["total_new_this_wave"] = sum(
        v["new_this_wave"] for v in summary["by_source"].values())
    out = WAVES / f"wave_{wave_n}_{now.strftime('%Y%m%dT%H%M%SZ')}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def report() -> dict:
    """Summarise every wave on disk - what the submission may call 'live'."""
    waves = []
    for p in sorted(WAVES.glob("wave_*.json")):
        try:
            waves.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    out = {
        "waves_run": len(waves),
        "waves": waves,
        "interpretation": (
            "Wave 1 is the retrospective baseline: everything the platforms "
            "still held when we first asked. Only waves 2 and later contain a "
            "genuinely live increment, and only that increment is described as "
            "live anywhere in this submission. `backfill_share` is the share "
            "of a wave's new records that are dated before the previous wave - "
            "records that existed but had not surfaced yet. It is the reason a "
            "daily figure computed today is not the daily figure you will get "
            "for the same day tomorrow, and it is measured rather than "
            "assumed away."),
    }
    (WAVES / "waves_summary.json").write_text(json.dumps(out, indent=2),
                                              encoding="utf-8")
    return out


if __name__ == "__main__":
    if "--report" in sys.argv:
        r = report()
        print(json.dumps(r, indent=2)[:4000])
    else:
        s = record_wave()
        print(f"wave {s['wave']}: {s['total_new_this_wave']:,} new records")
        for k, v in s["by_source"].items():
            extra = ""
            if "arrival_latency_hours" in v:
                extra = (f"  median latency {v['arrival_latency_hours']['median']}h"
                         f"  backfill {v.get('backfill_share', 0):.1%}")
            print(f"  {k:14s} raw={v['in_raw']:>7,}  new={v['new_this_wave']:>7,}{extra}")
