"""Run every collector and write a source audit.

Order matters only in that the slow, high-yield source (Play reviews) runs
first, so a failure later still leaves the backbone of the dataset on disk.
Each collector is independent and failure-tolerant: a dead feed costs us that
feed, not the run.

The audit it writes is not decoration. A dataset assembled from nine public
endpoints is only trustworthy if the reader can see which ones answered, which
ones refused, how many requests we made and what we did about the refusals -
so that is recorded per source and reproduced in the report.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "sources"))

from config import PROCESSED, RAW, REPORTS, REVIEWS_PER_APP, TOPIC, WINDOW_DAYS
from fetch import LEDGER, read_jsonl


def _stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main(skip_play: bool = False) -> dict:
    import attention
    import incidents
    import play_census
    import social_news

    started = _stamp()
    t0 = time.time()
    results: dict[str, int] = {}

    print(f"topic: {TOPIC}")
    print(f"window: last {WINDOW_DAYS} days  |  started {started}\n")

    if skip_play and (RAW / "play_reviews.jsonl").exists():
        results["google_play"] = len(read_jsonl(RAW / "play_reviews.jsonl"))
        print(f"[1/4] Google Play reviews: reusing {results['google_play']:,} on disk\n")
    else:
        print(f"[1/4] Google Play reviews (budget {REVIEWS_PER_APP:,}/app)")
        results["google_play"] = len(play_reviews.collect())
        print()

    print("[2/5] Social and news")
    social = social_news.collect()
    results.update({k: len(v) for k, v in social.items()})
    print()

    print("[3/5] Ground-truth incidents")
    results["incidents"] = len(incidents.collect())
    print()

    print("[4/5] Attention (Wikipedia pageviews)")
    results["attention_rows"] = len(attention.collect())
    print()

    # [5/5] Record what this run added that no previous run had seen. A single
    # pull is a retrospective snapshot; only the increment between waves is
    # genuinely live, and only the increment is described as live in the report.
    print("[5/5] Wave bookkeeping")
    import waves
    wave = waves.record_wave()
    results["new_records_this_wave"] = wave["total_new_this_wave"]
    print(f"      wave {wave['wave']}: {wave['total_new_this_wave']:,} records "
          f"not seen in any previous wave")
    print()

    audit = {
        "topic": TOPIC,
        "started_utc": started,
        "finished_utc": _stamp(),
        "elapsed_seconds": round(time.time() - t0, 1),
        "window_days": WINDOW_DAYS,
        "records_by_source": results,
        "total_reaction_records": sum(
            v for k, v in results.items()
            if k not in ("incidents", "attention_rows", "new_records_this_wave",
                         "google_play_census_days")),
        "wave": wave,
        "http": LEDGER.summary(),
    }
    (PROCESSED / "collection_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8")

    print("=" * 66)
    for k, v in results.items():
        print(f"  {k:22s} {v:>8,}")
    print("=" * 66)
    print(f"  total reaction records {audit['total_reaction_records']:>8,}")
    print(f"  live HTTP requests     {audit['http']['total_live_requests']:>8,}")
    print(f"  failures               {audit['http']['total_failures']:>8,}")
    print(f"  elapsed                {audit['elapsed_seconds']:>8.0f}s")
    return audit


if __name__ == "__main__":
    main(skip_play="--skip-play" in sys.argv)
