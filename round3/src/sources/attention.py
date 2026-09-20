"""L3: Wikipedia pageviews - an attention signal we did not generate.

Why this layer exists
---------------------
If an "engagement spike" is measured only in the reviews we scraped, and the
sentiment shift is measured in the same reviews, then the spike and the shift
share every source of error: the same scraper, the same window, the same
sampling. A coincidence between them proves very little.

Wikipedia pageviews are produced by a different population, through a different
mechanism, and recorded by a third party. When a brand's review volume spikes
*and* its encyclopaedia article is suddenly being read more on the same day,
the spike is about the world rather than about our collector. That is the
cheapest available defence against fooling ourselves, so it is worth one API.

Daily granularity, 45-day window, one series per brand.
"""
from __future__ import annotations

import urllib.parse
from datetime import datetime, timedelta, timezone

from config import RAW, WIKI_ARTICLES, WINDOW_DAYS
from fetch import get_json, write_jsonl

API = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
       "en.wikipedia/all-access/all-agents/{article}/daily/{start}/{end}")


def collect(articles=None, days: int = WINDOW_DAYS) -> list[dict]:
    articles = articles or WIKI_ARTICLES
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    s, e = start.strftime("%Y%m%d"), end.strftime("%Y%m%d")
    collected = datetime.now(timezone.utc).isoformat()

    out: list[dict] = []
    for art in articles:
        data = get_json(API.format(article=urllib.parse.quote(art, safe=""),
                                   start=s, end=e), max_age=6 * 3600)
        if not data or "items" not in data:
            print(f"      - {art}: unavailable", flush=True)
            continue
        items = data["items"]
        for it in items:
            ts = it["timestamp"]
            out.append({
                "source": "wikipedia_pageviews",
                "article": art,
                "brand_hint": art.replace("_", " ").split(" (")[0],
                "date": f"{ts[0:4]}-{ts[4:6]}-{ts[6:8]}",
                "views": it["views"],
                "collected_utc": collected,
            })
        vals = [i["views"] for i in items]
        peak = max(vals) if vals else 0
        mean = sum(vals) / len(vals) if vals else 0
        ratio = peak / mean if mean else 0
        print(f"      {art:28s} {len(items):3d}d  mean={mean:8,.0f} "
              f"peak={peak:9,d}  peak/mean={ratio:4.1f}x", flush=True)

    write_jsonl(out, RAW / "attention.jsonl")
    return out


if __name__ == "__main__":
    rows = collect()
    print(f"\n{len(rows)} daily pageview rows -> data/raw/attention.jsonl")
