"""L1 primary source: Google Play reviews for 44 delivery and service apps.

This is the backbone of the dataset, for four reasons no other source offers
at once:

* **A rating.** Every review carries 1-5 stars written by the same person who
  wrote the text. That is an *independent* sentiment label, which lets us
  validate the Round 2 model against something we did not produce - the single
  most useful cross-check available anywhere in this round.
* **An engagement count.** ``thumbsUpCount`` is other users endorsing the
  complaint, which is what an "engagement spike" should actually be made of.
* **A precise timestamp**, so reactions can be binned hourly.
* **An app version and a company reply**, which give two candidate trigger
  mechanisms: a bad release, and how fast the operator answered.

Reviews are pulled newest-first and paginated backwards until the per-app
budget or the window edge is reached, whichever comes first.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

from google_play_scraper import Sort, reviews

from config import (PLAY_APPS, RAW, REVIEW_PAGE, REVIEWS_PER_APP, window_start)
from fetch import anonymise, write_jsonl

SOURCE = "google_play"


def _as_utc(dt) -> str:
    """Play returns naive local-ish datetimes; treat as UTC and say so."""
    if dt is None:
        return ""
    if isinstance(dt, str):
        return dt
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def collect_app(app_id: str, country: str, brand: str, domain: str,
                budget: int = REVIEWS_PER_APP, stop_before=None) -> list[dict]:
    """Page backwards through one app's reviews until budget or window edge."""
    stop_before = stop_before or window_start()
    out: list[dict] = []
    token = None
    collected_at = datetime.now(timezone.utc).isoformat()

    while len(out) < budget:
        try:
            batch, token = reviews(
                app_id, lang="en", country=country, sort=Sort.NEWEST,
                count=min(REVIEW_PAGE, budget - len(out)),
                continuation_token=token,
            )
        except Exception as exc:                       # one dead app must not
            print(f"      ! {app_id}: {type(exc).__name__}", flush=True)
            break
        if not batch:
            break

        oldest_in_batch = None
        for r in batch:
            created = r.get("at")
            if created is not None and created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            oldest_in_batch = created
            out.append({
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
            })

        if token is None:
            break
        if oldest_in_batch is not None and oldest_in_batch < stop_before:
            break                                      # walked past the window
        time.sleep(0.15)                               # be a good guest
    return out


def collect(budget: int = REVIEWS_PER_APP, apps=None) -> list[dict]:
    apps = apps or PLAY_APPS
    everything: list[dict] = []
    for i, (app_id, cc, brand, domain) in enumerate(apps, 1):
        t0 = time.time()
        rows = collect_app(app_id, cc, brand, domain, budget=budget)
        everything.extend(rows)
        span = ""
        if rows:
            span = f"{rows[-1]['created_utc'][:10]} .. {rows[0]['created_utc'][:10]}"
        print(f"   [{i:2d}/{len(apps)}] {brand:12s} {domain:15s} "
              f"{len(rows):5d} reviews  {span}  ({time.time()-t0:4.1f}s)", flush=True)
    write_jsonl(everything, RAW / "play_reviews.jsonl")
    return everything


if __name__ == "__main__":
    rows = collect()
    print(f"\ntotal: {len(rows):,} reviews -> data/raw/play_reviews.jsonl")
