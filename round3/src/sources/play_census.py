"""L1 primary source, rebuilt as a *survey* rather than a convenience sample.

Why this file replaces the first Play collector
-----------------------------------------------
The first version paginated newest-first with a flat 5,000-review budget per
app. On a low-volume app that reaches the window edge; on a high-volume app it
does not. The consequence was measured after the fact and it was severe:

    Flipkart      5 days of history      Uber Eats    45 days
    Blinkit       5 days                 DoorDash     45 days
    Temu          9 days                 Amazon       45 days

Eighteen of forty-four brands therefore *entered the corpus mid-window*. Daily
volume rose from 59 rows on 6 Aug to 822 on 18 Sep, and that rise correlated
r = 0.89 with the day index - it was the scraper, not the public. Every
"engagement spike" in the ride-hailing series was Rapido appearing on 5 Sep and
Uber on 8 Sep. Every "distinctive term" at the headline change point was the
name of a brand that had just entered the sample.

The fix is not a bigger budget. A bigger budget still gives a corpus whose
composition is a function of each brand's posting rate. The fix is to separate
the two things the first collector conflated:

**Census.** Paginate every app all the way back to the window edge and *count*
every review seen, per brand-day, along with its rating and its thumbs-up.
Counting is cheap - it does not need the text kept. This is a complete
enumeration of the sampling frame, so daily review volume becomes a real
measurement of activity instead of a measurement of how far we paged.

**Quota sample.** Keep at most ``PER_BRAND_DAY`` reviews for each brand-day,
drawn by *reservoir sampling* so the kept rows are a uniform random sample of
that brand-day rather than its most recent hour. Text volume is then a design
constant, and any movement in sampled sentiment cannot be an artefact of
sampling depth.

The two combine into a standard ratio estimator: the true number of
delay-related reactions for a brand-day is estimated as

    census_n x (delay-related share observed in that brand-day's quota sample)

with a binomial standard error that the analysis carries through. That is the
difference between "our scraper found more complaints" and "more people
complained".

Known bias, stated rather than discovered later
-----------------------------------------------
``thumbsUpCount`` is cumulative to the moment of collection, so a review from
6 August has had six more weeks to gather endorsements than one from 18
September. Engagement per day is therefore biased *upward for older days*,
which works against finding a recent spike, not for it. Any engagement spike
surviving this bias is a conservative finding; the analysis states the
direction of the bias next to the result.
"""
from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from google_play_scraper import Sort, reviews

from config import PLAY_APPS, RAW, SEED, window_start
from fetch import anonymise, write_jsonl

SOURCE = "google_play"

PER_BRAND_DAY = 80      # quota: kept rows per brand-day
PAGE = 200              # Play's maximum page size
MAX_PAGES = 900         # hard stop, ~180k reviews, so one runaway app cannot
                        # consume the whole run
SLEEP = 0.12            # be a good guest


def _as_utc(dt) -> str:
    if dt is None:
        return ""
    if isinstance(dt, str):
        return dt
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _row(r: dict, app_id: str, country: str, brand: str, domain: str,
         collected_at: str) -> dict:
    created = r.get("at")
    if created is not None and getattr(created, "tzinfo", None) is None:
        created = created.replace(tzinfo=timezone.utc)
    return {
        "source": SOURCE,
        "source_id": app_id,
        "store_country": country,
        "brand": brand,
        "delay_domain": domain,
        "record_native_id": r.get("reviewId"),
        "created_utc": _as_utc(created),
        "collected_utc": collected_at,
        "title": "",
        "text": (r.get("content") or "").strip(),
        "author_pseudonym": anonymise(r.get("userName")),
        "rating": r.get("score"),
        "thumbs_up": r.get("thumbsUpCount") or 0,
        "app_version": r.get("reviewCreatedVersion") or r.get("appVersion") or "",
        "company_replied": bool(r.get("replyContent")),
        "company_reply_utc": _as_utc(r.get("repliedAt")),
        "company_reply_text": (r.get("replyContent") or "").strip(),
        "url": f"https://play.google.com/store/apps/details?id={app_id}",
    }


def collect_app(app_id: str, country: str, brand: str, domain: str,
                stop_before: datetime, per_day: int = PER_BRAND_DAY,
                rng: random.Random | None = None) -> tuple[list[dict], list[dict], dict]:
    """Enumerate one app back to ``stop_before``; return (sample, census, audit).

    Reservoir sampling keeps the quota uniform over the day. The naive
    alternative - keep the first ``per_day`` seen - keeps the *latest* reviews
    of each day, because pagination is newest-first, and would put a within-day
    recency bias into every daily mean.
    """
    rng = rng or random.Random(SEED)
    collected_at = datetime.now(timezone.utc).isoformat()

    reservoir: dict[str, list[dict]] = defaultdict(list)
    seen_per_day: dict[str, int] = defaultdict(int)
    census: dict[str, dict] = {}

    token = None
    pages = 0
    total_seen = 0
    oldest = None
    stop_reason = "token_exhausted"

    while pages < MAX_PAGES:
        try:
            batch, token = reviews(app_id, lang="en", country=country,
                                   sort=Sort.NEWEST, count=PAGE,
                                   continuation_token=token)
        except Exception as exc:
            stop_reason = f"error:{type(exc).__name__}"
            break
        pages += 1
        if not batch:
            stop_reason = "empty_page"
            break

        for r in batch:
            created = r.get("at")
            if created is None:
                continue
            if getattr(created, "tzinfo", None) is None:
                created = created.replace(tzinfo=timezone.utc)
            oldest = created
            if created < stop_before:
                continue                       # counted only inside the window
            day = created.date().isoformat()
            total_seen += 1

            c = census.setdefault(day, {"n": 0, "rating_sum": 0.0, "rating_n": 0,
                                        "thumbs_sum": 0, "replied": 0})
            c["n"] += 1
            if r.get("score") is not None:
                c["rating_sum"] += float(r["score"])
                c["rating_n"] += 1
            c["thumbs_sum"] += int(r.get("thumbsUpCount") or 0)
            c["replied"] += 1 if r.get("replyContent") else 0

            # --- reservoir sampling, per brand-day --------------------------
            seen_per_day[day] += 1
            k = seen_per_day[day]
            row = None
            if len(reservoir[day]) < per_day:
                row = _row(r, app_id, country, brand, domain, collected_at)
                reservoir[day].append(row)
            else:
                j = rng.randrange(k)
                if j < per_day:
                    row = _row(r, app_id, country, brand, domain, collected_at)
                    reservoir[day][j] = row

        if token is None:
            stop_reason = "token_exhausted"
            break
        if oldest is not None and oldest < stop_before:
            stop_reason = "reached_window_edge"
            break
        if pages >= MAX_PAGES:
            stop_reason = "page_cap"
        time.sleep(SLEEP)

    sample = [r for rows in reservoir.values() for r in rows]
    census_rows = [{
        "source": "google_play_census",
        "brand": brand,
        "delay_domain": domain,
        "source_id": app_id,
        "store_country": country,
        "date": day,
        "reviews_total": v["n"],
        "rating_mean": round(v["rating_sum"] / v["rating_n"], 4) if v["rating_n"] else None,
        "thumbs_total": v["thumbs_sum"],
        "company_replied_total": v["replied"],
        "sampled": len(reservoir[day]),
        "collected_utc": collected_at,
    } for day, v in sorted(census.items())]

    audit = {
        "brand": brand, "app_id": app_id, "country": country, "domain": domain,
        "pages": pages, "reviews_enumerated": total_seen,
        "reviews_kept": len(sample),
        "days_covered": len(census),
        "oldest_seen_utc": _as_utc(oldest),
        "stop_reason": stop_reason,
        "complete_window": stop_reason == "reached_window_edge",
    }
    return sample, census_rows, audit


def collect(apps=None, per_day: int = PER_BRAND_DAY) -> tuple[list[dict], list[dict]]:
    apps = apps or PLAY_APPS
    stop_before = window_start().replace(hour=0, minute=0, second=0, microsecond=0)
    rng = random.Random(SEED)

    sample: list[dict] = []
    census: list[dict] = []
    audits: list[dict] = []

    for i, (app_id, cc, brand, domain) in enumerate(apps, 1):
        t0 = time.time()
        s, c, a = collect_app(app_id, cc, brand, domain, stop_before, per_day, rng)
        sample.extend(s)
        census.extend(c)
        audits.append(a)
        print(f"   [{i:2d}/{len(apps)}] {brand:12s} {domain:15s} "
              f"enum={a['reviews_enumerated']:7,d}  kept={a['reviews_kept']:5,d}  "
              f"days={a['days_covered']:3d}  {a['stop_reason']:20s} "
              f"({time.time() - t0:5.1f}s)", flush=True)

    write_jsonl(sample, RAW / "play_reviews.jsonl")
    write_jsonl(census, RAW / "play_census.jsonl")
    complete = sum(1 for a in audits if a["complete_window"])
    (RAW / "play_census_audit.json").write_text(json.dumps({
        "per_brand_day_quota": per_day,
        "window_start_utc": stop_before.isoformat(),
        "apps": len(apps),
        "apps_with_complete_window": complete,
        "reviews_enumerated": sum(a["reviews_enumerated"] for a in audits),
        "reviews_kept": len(sample),
        "by_app": audits,
    }, indent=2), encoding="utf-8")
    print(f"\n   enumerated {sum(a['reviews_enumerated'] for a in audits):,} reviews, "
          f"kept {len(sample):,}; {complete}/{len(apps)} apps reached the window edge")
    return sample, census


if __name__ == "__main__":
    collect()
