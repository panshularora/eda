"""L1 secondary sources: Reddit, Mastodon, Google News, Hacker News.

Play reviews are rich but they are one genre - a complaint written to a
company, inside its own app store. Relying on them alone would let a genre
artefact masquerade as a finding, exactly the trap Round 2 caught with
``topic_category``.

So the dataset triangulates across four more genres:

``reddit``     peer-to-peer venting and advice, where people talk to each other
               rather than to the company;
``mastodon``   open-web microblogging, the closest available stand-in for the
               Twitter/X firehose that is no longer free to read;
``news``       journalism, which is mostly *about* the trigger rather than a
               reaction to it - which is precisely what makes it useful for
               separating cause from response;
``hackernews`` technically literate discussion of service outages, with a real
               engagement metric (points).

Each returns the same flat record shape as ``play_reviews`` so they can be
concatenated without special-casing downstream.
"""
from __future__ import annotations

import html
import re
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from config import (HN_QUERIES, MASTODON_INSTANCES, MASTODON_TAGS, NEWS_QUERIES,
                    RAW, REDDIT_SUBS, WINDOW_DAYS, window_start)
from fetch import anonymise, get, get_json, write_jsonl

ATOM = {"a": "http://www.w3.org/2005/Atom"}
_TAGS = re.compile(r"<[^>]+>")


def _strip_html(s: str) -> str:
    return html.unescape(_TAGS.sub(" ", s or "")).replace("\xa0", " ").strip()


def _iso(dt: datetime | None) -> str:
    if dt is None:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _blank(**kw) -> dict:
    """The shared record shape; sources fill what they have and leave the rest."""
    base = dict(
        source="", source_id="", store_country="", brand="", delay_domain="",
        record_native_id="", created_utc="", collected_utc=_iso(datetime.now(timezone.utc)),
        title="", text="", author_pseudonym="", rating=None, thumbs_up=0,
        publisher="",
        app_version="", company_replied=False, company_reply_utc="",
        company_reply_text="", url="",
    )
    base.update(kw)
    return base


# ---------------------------------------------------------------------------
def collect_reddit(subs=None) -> list[dict]:
    """Subreddit Atom feeds. Reddit's JSON API rejects unauthenticated reads
    (HTTP 403); the RSS surface is still open, so that is what we use."""
    subs = subs or REDDIT_SUBS
    out: list[dict] = []
    for sub in subs:
        raw = get(f"https://www.reddit.com/r/{sub}/new/.rss", max_age=900)
        if not raw:
            print(f"      - r/{sub}: unavailable", flush=True)
            continue
        try:
            root = ET.fromstring(raw)
        except ET.ParseError:
            continue
        n = 0
        for e in root.findall("a:entry", ATOM):
            def txt(tag):
                el = e.find(f"a:{tag}", ATOM)
                return el.text if el is not None and el.text else ""
            link = e.find("a:link", ATOM)
            author = e.find("a:author/a:name", ATOM)
            out.append(_blank(
                source="reddit", source_id=f"r/{sub}",
                record_native_id=txt("id"),
                created_utc=txt("published") or txt("updated"),
                title=_strip_html(txt("title")),
                text=_strip_html(txt("content")),
                author_pseudonym=anonymise(author.text if author is not None else ""),
                url=link.get("href") if link is not None else "",
            ))
            n += 1
        print(f"      r/{sub:22s} {n:3d} posts", flush=True)
    return out


# ---------------------------------------------------------------------------
def collect_mastodon(tags=None, instances=None, per_tag: int = 40) -> list[dict]:
    """Public hashtag timelines. Favourites and boosts are real engagement."""
    tags = tags or MASTODON_TAGS
    instances = instances or MASTODON_INSTANCES
    out: list[dict] = []
    for inst in instances:
        for tag in tags:
            data = get_json(
                f"https://{inst}/api/v1/timelines/tag/{urllib.parse.quote(tag)}"
                f"?limit={per_tag}", max_age=900)
            if not data:
                continue
            for p in data:
                acct = (p.get("account") or {}).get("acct", "")
                out.append(_blank(
                    source="mastodon", source_id=f"{inst}#{tag}",
                    record_native_id=str(p.get("id", "")),
                    created_utc=p.get("created_at", ""),
                    text=_strip_html(p.get("content", "")),
                    author_pseudonym=anonymise(acct),
                    thumbs_up=(p.get("favourites_count") or 0)
                              + (p.get("reblogs_count") or 0)
                              + (p.get("replies_count") or 0),
                    url=p.get("url") or "",
                ))
            print(f"      {inst}#{tag:18s} {len(data):3d} posts", flush=True)
    return out


# ---------------------------------------------------------------------------
def collect_news(queries=None) -> list[dict]:
    """Google News RSS. One row per article; the body is the outlet's summary."""
    queries = queries or NEWS_QUERIES
    out: list[dict] = []
    for q in queries:
        url = ("https://news.google.com/rss/search?q="
               f"{urllib.parse.quote(q)}+when:{WINDOW_DAYS}d&hl=en-US&gl=US&ceid=US:en")
        raw = get(url, max_age=1800)
        if not raw:
            continue
        try:
            root = ET.fromstring(raw)
        except ET.ParseError:
            continue
        items = root.findall(".//item")
        for it in items:
            def txt(tag):
                el = it.find(tag)
                return el.text or "" if el is not None else ""
            try:
                pub = _iso(parsedate_to_datetime(txt("pubDate")))
            except Exception:
                pub = ""
            src_el = it.find("source")
            # The publisher is NOT the brand. "The Times of India" is who wrote
            # about the delay, not who caused it; putting it in `brand` produced
            # 417 spurious brands. It goes in `publisher`, and `brand` is left
            # empty so normalise.py attributes it from the text like any other
            # source that does not carry it.
            out.append(_blank(
                source="news", source_id=q.strip('"'),
                publisher=(src_el.text if src_el is not None else "") or "",
                record_native_id=txt("guid"),
                created_utc=pub,
                title=_strip_html(txt("title")),
                text=_strip_html(txt("description")),
                url=txt("link"),
            ))
        print(f"      news {q:26s} {len(items):3d} articles", flush=True)
    return out


# ---------------------------------------------------------------------------
def collect_hackernews(queries=None, per_query: int = 200) -> list[dict]:
    """Algolia's HN index: full text, precise timestamps, points and comments."""
    queries = queries or HN_QUERIES
    start = int(window_start().timestamp())
    out: list[dict] = []
    for q in queries:
        data = get_json(
            "https://hn.algolia.com/api/v1/search_by_date?"
            f"query={urllib.parse.quote(q)}&tags=(story,comment)"
            f"&numericFilters=created_at_i>{start}&hitsPerPage={per_query}",
            max_age=1800)
        if not data:
            continue
        hits = data.get("hits", [])
        for h in hits:
            body = h.get("story_text") or h.get("comment_text") or ""
            out.append(_blank(
                source="hackernews", source_id=q,
                record_native_id=str(h.get("objectID", "")),
                created_utc=h.get("created_at", ""),
                title=_strip_html(h.get("title") or ""),
                text=_strip_html(body),
                author_pseudonym=anonymise(h.get("author")),
                thumbs_up=(h.get("points") or 0) + (h.get("num_comments") or 0),
                url=h.get("url") or f"https://news.ycombinator.com/item?id={h.get('objectID')}",
            ))
        print(f"      hn   {q:26s} {len(hits):3d} hits", flush=True)
    return out


# ---------------------------------------------------------------------------
def collect() -> dict[str, list[dict]]:
    print("    reddit:", flush=True)
    reddit = collect_reddit()
    print("    mastodon:", flush=True)
    masto = collect_mastodon()
    print("    news:", flush=True)
    news = collect_news()
    print("    hacker news:", flush=True)
    hn = collect_hackernews()
    for name, rows in (("reddit", reddit), ("mastodon", masto),
                       ("news", news), ("hackernews", hn)):
        write_jsonl(rows, RAW / f"{name}.jsonl")
    return {"reddit": reddit, "mastodon": masto, "news": news, "hackernews": hn}


if __name__ == "__main__":
    res = collect()
    for k, v in res.items():
        print(f"{k:12s} {len(v):6,d}")
