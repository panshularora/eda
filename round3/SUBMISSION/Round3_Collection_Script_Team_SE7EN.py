"""
=============================================================================
 DATA VORTEX A'26 - ROUND 3 - DATA COLLECTION / SCRAPING SCRIPT
 Team SE7EN: Tanmay Singh, Panshul Arora
 Topic: Reaction to a Major Delivery or Service Delay
=============================================================================

 This is the complete collection layer, concatenated from the seven modules it
 is split across in the repository. The module boundaries are preserved as
 banners below so it reads as the package it is.

 Run the real thing from the repository with:

     pip install -r round3/requirements.txt
     python round3/run_round3.py

 Sources, by layer:

   L1 REACTION   Google Play reviews (44 apps, 7 delay domains, 7 countries)
                 Reddit (subreddit Atom feeds)
                 Mastodon (public hashtag timelines)
                 Google News RSS  ~  Hacker News (Algolia)
   L2 TRIGGER    FAA national airspace delay register
                 10 public status pages (Atlassian Statuspage RSS)
   L3 ATTENTION  Wikipedia pageviews API

 Conduct: one identifying User-Agent with a contact address on every request,
 a per-host minimum interval, exponential backoff that respects Retry-After,
 an on-disk cache so re-runs do not re-request, and author handles replaced
 with salted hashes at the point of collection. Only public endpoints are
 read; nothing is authenticated, and no personal identifier is stored.
=============================================================================
"""

from __future__ import annotations


##############################################################################
# src/config.py
# configuration: topic, source roster, taxonomies
##############################################################################

"""Round 3 configuration: the assigned topic, the source roster, and the taxonomies.

Assigned topic
--------------
**"Reaction to a Major Delivery or Service Delay."**

Everything in this file exists to make that sentence measurable. A "reaction"
needs text, a timestamp and an engagement signal; a "delay" needs a type and a
responsible brand; and "major" needs a trigger you can point at. The collector
is therefore built in three layers, and each source below is tagged with the
layer it serves:

  L1 REACTION   what people said, with engagement and a timestamp
  L2 TRIGGER    documented incidents with start times - the ground truth that
                turns "sentiment moved" into "sentiment moved *because*"
  L3 ATTENTION  an independent volume signal (Wikipedia pageviews) that can
                corroborate an engagement spike without using our own text

Pinned identifiers
------------------
App IDs are pinned rather than searched. ``google_play_scraper.search`` returns
different results run to run and currently raises on apps with a null field, so
a pinned, verified list is both reproducible and honest about what was sampled.
Every ID in ``PLAY_APPS`` was verified to resolve before being added.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
ROUND3 = Path(__file__).resolve().parents[1]
REPO = ROUND3.parent
DATA = ROUND3 / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
REPORTS = ROUND3 / "reports"
FIGURES = REPORTS / "figures"
SUBMISSION = ROUND3 / "SUBMISSION"
NOTEBOOKS = ROUND3 / "notebooks"

# the Round 2 deliverable we are required to re-apply here
ROUND2_SRC = REPO / "round2" / "src"
ROUND2_MODEL = REPO / "round2" / "models" / "social_engine_nlp_team_se7en.pkl"

for _d in (RAW, PROCESSED, REPORTS, FIGURES, SUBMISSION, NOTEBOOKS):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------
TOPIC = "Reaction to a Major Delivery or Service Delay"
TEAM = "Team SE7EN"
MEMBERS = "Tanmay Singh · Panshul Arora"
EVENT = "Data Vortex A'26 · Round 3 · Rebuilding the Social Engine"
CONTACT = "tanmaysingh07082005@gmail.com"
USER_AGENT = (
    f"DataVortexResearch/1.0 (academic competition project; Team SE7EN; "
    f"contact {CONTACT}) python-urllib"
)
SEED = 42

# ---------------------------------------------------------------------------
# collection window
# ---------------------------------------------------------------------------
# Reviews are pulled newest-first and paginated backwards until either the
# budget or WINDOW_DAYS is hit, so the window is a *request*, not a guarantee -
# a low-volume app simply will not have that much history. The realised window
# per source is measured and reported rather than assumed.
WINDOW_DAYS = 45
ANALYSIS_TZ = "UTC"


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def window_start() -> datetime:
    return now_utc() - timedelta(days=WINDOW_DAYS)


# ---------------------------------------------------------------------------
# L1 - Google Play review roster (the primary reaction source)
# ---------------------------------------------------------------------------
# Why Play and not the App Store: Apple's customer-review RSS endpoint still
# returns HTTP 200 but an empty entry list for every app, country and sort order
# we tried (documented in reports/source_audit.md). Play additionally gives a
# 1-5 star rating, a thumbs-up count, the app version and any company reply -
# four fields the analysis leans on that Apple never exposed.
#
# (app_id, store_country, brand, delay_domain)
PLAY_APPS: list[tuple[str, str, str, str]] = [
    # --- food delivery -----------------------------------------------------
    ("com.dd.doordash",                 "us", "DoorDash",     "food_delivery"),
    ("com.ubercab.eats",                "us", "Uber Eats",    "food_delivery"),
    ("com.grubhub.android",             "us", "Grubhub",      "food_delivery"),
    ("com.deliveroo.orderapp",          "gb", "Deliveroo",    "food_delivery"),
    ("com.justeat.app.uk",              "gb", "Just Eat",     "food_delivery"),
    ("com.application.zomato",          "in", "Zomato",       "food_delivery"),
    ("in.swiggy.android",               "in", "Swiggy",       "food_delivery"),
    ("com.Dominos",                     "in", "Domino's",     "food_delivery"),
    ("com.global.foodpanda.android",    "sg", "foodpanda",    "food_delivery"),
    ("com.glovo",                       "es", "Glovo",        "food_delivery"),
    ("com.talabat",                     "ae", "talabat",      "food_delivery"),
    # --- quick commerce ----------------------------------------------------
    ("com.grofers.customerapp",         "in", "Blinkit",      "quick_commerce"),
    ("com.zeptoconsumerapp",            "in", "Zepto",        "quick_commerce"),
    ("in.swiggy.android.instamart",     "in", "Instamart",    "quick_commerce"),
    ("com.instacart.client",            "us", "Instacart",    "quick_commerce"),
    ("com.bigbasket.mobileapp",         "in", "bigbasket",    "quick_commerce"),
    # --- parcel / courier --------------------------------------------------
    ("com.fedex.ida.android",           "us", "FedEx",        "parcel_courier"),
    ("com.ups.mobile.android",          "us", "UPS",          "parcel_courier"),
    ("de.dhl.paket",                    "de", "DHL",          "parcel_courier"),
    ("com.aftership.AfterShip",         "us", "AfterShip",    "parcel_courier"),
    ("com.utilex.bluedart",             "in", "Bluedart",     "parcel_courier"),
    # --- e-commerce fulfilment ---------------------------------------------
    ("com.amazon.mShop.android.shopping", "us", "Amazon",     "ecommerce"),
    ("in.amazon.mShop.android.shopping",  "in", "Amazon IN",  "ecommerce"),
    ("com.flipkart.android",            "in", "Flipkart",     "ecommerce"),
    ("com.myntra.android",              "in", "Myntra",       "ecommerce"),
    ("com.meesho.supply",               "in", "Meesho",       "ecommerce"),
    ("com.einnovation.temu",            "us", "Temu",         "ecommerce"),
    ("com.contextlogic.wish",           "us", "Wish",         "ecommerce"),
    ("com.zzkko",                       "us", "SHEIN",        "ecommerce"),
    ("com.alibaba.aliexpresshd",        "us", "AliExpress",   "ecommerce"),
    ("com.ebay.mobile",                 "us", "eBay",         "ecommerce"),
    # --- ride hailing ------------------------------------------------------
    ("com.ubercab",                     "us", "Uber",         "ride_hailing"),
    ("com.olacabs.customer",            "in", "Ola",          "ride_hailing"),
    ("me.lyft.android",                 "us", "Lyft",         "ride_hailing"),
    ("com.rapido.passenger",            "in", "Rapido",       "ride_hailing"),
    # --- airline -----------------------------------------------------------
    ("in.goindigo.android",             "in", "IndiGo",       "airline"),
    ("com.united.mobile.android",       "us", "United",       "airline"),
    ("com.delta.mobile.android",        "us", "Delta",        "airline"),
    ("com.aa.android",                  "us", "American",     "airline"),
    ("com.southwestairlines.mobile",    "us", "Southwest",    "airline"),
    ("com.ryanair.cheapflights",        "gb", "Ryanair",      "airline"),
    # --- telecom / ISP -----------------------------------------------------
    ("com.jio.myjio",                   "in", "Jio",          "telecom_isp"),
    ("com.myairtelapp",                 "in", "Airtel",       "telecom_isp"),
    ("com.xfinity.digitalhome",         "us", "Xfinity",      "telecom_isp"),
]

DELAY_DOMAINS = sorted({d for *_, d in PLAY_APPS})
BRANDS = sorted({b for _, _, b, _ in PLAY_APPS})

# ---------------------------------------------------------------------------
# L1 - social and news sources
# ---------------------------------------------------------------------------
REDDIT_SUBS = [
    "doordash", "doordash_drivers", "UberEATS", "grubhub", "deliveroo",
    "FedEx", "UPS", "usps", "amazonprime", "AmazonFlexDrivers",
    "shipping", "InstacartShoppers", "Zomato", "india", "delhi",
    "flightattendants", "delta", "unitedairlines", "Flights",
]

MASTODON_INSTANCES = ["mastodon.social", "mstdn.social"]
MASTODON_TAGS = [
    "delivery", "delays", "delayed", "outage", "doordash", "ubereats",
    "amazon", "fedex", "ups", "shipping", "logistics", "flightdelay",
    "customerservice", "downtime",
]

NEWS_QUERIES = [
    '"delivery delay"', '"delivery delays"', '"shipping delay"',
    '"shipping delays"', '"parcel delays"', '"late delivery"',
    '"flight delays"', '"service outage"', '"order delayed"',
    '"supply chain delay"', 'courier delay', 'delivery disruption',
    'quick commerce delay', 'food delivery late',
]

HN_QUERIES = ["outage", "delivery delay", "shipping delay", "downtime",
              "service disruption", "logistics"]

# ---------------------------------------------------------------------------
# L2 - trigger sources (documented incidents with timestamps)
# ---------------------------------------------------------------------------
FAA_STATUS_URL = "https://nasstatus.faa.gov/api/airport-status-information"

STATUSPAGE_FEEDS = [
    ("Discord",    "https://discordstatus.com/history.rss"),
    ("Cloudflare", "https://www.cloudflarestatus.com/history.rss"),
    ("AWS",        "https://status.aws.amazon.com/rss/all.rss"),
    ("Dropbox",    "https://status.dropbox.com/history.rss"),
    ("Twilio",     "https://status.twilio.com/history.rss"),
    ("Shopify",    "https://www.shopifystatus.com/history.rss"),
    ("Squarespace","https://status.squarespace.com/history.rss"),
    ("Datadog",    "https://status.datadoghq.com/history.rss"),
    ("Zoom",       "https://status.zoom.us/history.rss"),
    ("GitHub",     "https://www.githubstatus.com/history.rss"),
]

# ---------------------------------------------------------------------------
# L3 - attention (independent of anything we scrape)
# ---------------------------------------------------------------------------
WIKI_ARTICLES = [
    "DoorDash", "Uber_Eats", "Grubhub", "Deliveroo", "Just_Eat",
    "Zomato", "Swiggy", "Blinkit", "Zepto", "Instacart",
    "FedEx", "United_Parcel_Service", "DHL", "Amazon_(company)",
    "Flipkart", "Myntra", "Meesho", "Temu_(marketplace)", "Shein",
    "AliExpress", "EBay", "Uber", "Ola_Cabs", "Lyft", "IndiGo",
    "United_Airlines", "Delta_Air_Lines", "American_Airlines",
    "Ryanair", "Reliance_Jio", "Bharti_Airtel", "Comcast",
]

# ---------------------------------------------------------------------------
# taxonomies - "different types of delays and their different types of reaction"
# ---------------------------------------------------------------------------
# Ordered: the first pattern that fires wins, so the more specific failure modes
# are listed before the generic ones. Patterns are regex, matched case-insensitively
# against the normalised text.
DELAY_TYPES: list[tuple[str, str]] = [
    ("never_arrived",     r"never (?:arriv|came|deliver|show|got|receiv)|not deliver|no[t]? receiv|missing (?:order|parcel|package)|marked (?:as )?deliver(?:ed)? but|no (?:food|order|item|parcel|package|delivery)\b.{0,20}(?:deliver|arriv|came)|did ?n[o']?t (?:arrive|come|deliver|show)"),
    ("missing_items",     r"missing item|item[s]? missing|incomplete order|half (?:the )?order|wrong item|items? (?:were|was) not"),
    ("stuck_in_transit",  r"stuck in transit|no (?:tracking )?update|not mov(?:ed|ing)|in transit for|tracking (?:has ?n[o']t|not) updat|same status"),
    ("cancelled",         r"\bcancel(?:led|ed|lation|s|ling)?\b|auto[- ]cancel"),
    ("refund_delay",      r"refund (?:not|still|hasn'?t|delay|pending)|no refund|waiting for (?:my )?refund|money not (?:refund|credit|return)"),
    ("outage",            r"\b(?:outage|server (?:down|error)|app (?:is )?down|site (?:is )?down|not working|can'?t (?:log ?in|open|access)|crash)"),
    ("support_delay",     r"(?:no|zero|poor) (?:response|reply|support)|support (?:never|not|does ?n[o']t) (?:respond|reply|help)|no one (?:responds|replies|helps)|chatbot"),
    ("long_wait",         r"\b(?:\d+\s*(?:hour|hr|min|minute|day)s?)\b.*(?:wait|late|delay)|wait(?:ing|ed)? (?:for )?(?:\d+|ages|hours|forever)|took (?:\d+|forever|ages)"),
    ("late_delivery",     r"\b(?:late|delay(?:ed|s|ing)?|slow|took (?:too )?long|behind schedule|past (?:the )?eta|overdue)\b"),
]

REACTION_TYPES: list[tuple[str, str]] = [
    ("churn_threat",      r"(?:un)?install(?:ing|ed)?|delet(?:e|ing) (?:the )?app|never (?:order|use|buy)(?:ing)? again|switch(?:ing)? to|last time|done with (?:this|you)|cancel(?:ling)? (?:my )?(?:subscription|prime|plus|membership)"),
    ("refund_demand",     r"(?:want|need|give|demand|asking for) (?:a |my )?refund|refund me|compensat|money back|reimburse"),
    ("escalation",        r"complain(?:t|ed|ing)?|consumer (?:court|forum)|legal|lawyer|sue|ombudsman|report(?:ing|ed) (?:this|them)|bbc|bbb|twitter\b.*\bescalat"),
    ("anger",             r"worst|terrible|awful|horrible|pathetic|disgust|furious|angry|rubbish|garbage|scam|fraud|cheat|useless|hate\b|ridiculous|unacceptable"),
    ("sarcasm_humour",    r"lol|lmao|haha|\bironic|congrat(?:s|ulations)\b.*\b(?:fail|late)|great job\b|well done\b.*\b(?:late|fail)|¡?thanks for nothing"),
    ("resigned",          r"(?:as )?usual|every time|always (?:late|happens)|not surprised|expected|typical|again and again|used to it"),
    ("praise_recovery",   r"(?:but|however|although).{0,40}(?:resolved|refunded|sorted|fixed|apolog)|customer (?:care|service) (?:was|were) (?:good|great|helpful|quick)|quick(?:ly)? resolv|good recovery"),
    ("informational",     r"^(?:fyi|note|update|psa)\b|for (?:your )?(?:info|reference)"),
]

# A reaction only counts as on-topic if it mentions a delay or service failure.
#
# Every alternative below is word-bounded, and that is not stylistic. The first
# version of this filter was written without boundaries and reproduced, in our
# own code, precisely the defect we caught in the Round 2 labels: bare `late`
# matched "chocolate" and "translate", bare `down` matched "download", and bare
# `eta` matched "retail", "beta" and "meta". A hand audit of the residual bucket
# found those false positives sitting in the dataset, which is the entire reason
# to audit a rule rather than trust it. Boundaries fixed it; the audit is
# reproduced in reports/relevance_audit.md.
DELAY_RELEVANCE = (
    r"\blate\b|\bdelay|\bslow\b|\bwait(?:ing|ed|s)?\b|\bstuck\b|\bpending\b|"
    r"never (?:arriv|came|deliver|show|got|receiv)|\bnot deliver|\bno[t]? receiv|"
    r"\bmissing\b|\bcancel|\brefund|\boutage\b|\bdown\b|\bdowntime\b|"
    r"\bnot working\b|\bno update\b|\beta\b|\btook (?:too )?long\b|"
    r"\b\d+\s*(?:hour|hr|min|minute|day|week)s?\b|\boverdue\b|\bon time\b|"
    r"\bbehind schedule\b|\bheld up\b|\bpostpon|\bno show\b|"
    r"\bno (?:food|order|item|parcel|package)\b"
)

# ---------------------------------------------------------------------------
# politeness - we are a guest on every one of these endpoints
# ---------------------------------------------------------------------------
RATE_LIMITS = {            # minimum seconds between requests, per host
    "default":              1.0,
    "www.reddit.com":       6.0,
    "mastodon.social":      1.5,
    "mstdn.social":         1.5,
    "news.google.com":      2.0,
    "api.gdeltproject.org": 6.0,
    "wikimedia.org":        0.5,
    "hn.algolia.com":       1.0,
    "api.stackexchange.com": 2.0,
}
MAX_RETRIES = 4
BACKOFF_BASE = 2.0

# per-app review budget; 5,000 reaches ~45 days on a high-volume app
REVIEWS_PER_APP = 5000
REVIEW_PAGE = 200


##############################################################################
# src/fetch.py
# polite HTTP: rate limits, backoff, cache, ledger, anonymisation
##############################################################################

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
        max_age: float | None = None) -> bytes | None:
    """Fetch a URL politely. Returns ``None`` if it could not be retrieved.

    Returning None rather than raising is deliberate: one dead feed among forty
    should degrade the dataset, not abort the collection. Every failure is
    counted in the ledger and surfaced in the source audit.
    """
    host = _host(url)
    path = _cache_path(url)
    if use_cache and path.exists():
        fresh = max_age is None or (time.time() - path.stat().st_mtime) < max_age
        if fresh:
            LEDGER.cached[host] += 1
            return gzip.decompress(path.read_bytes())

    for attempt in range(MAX_RETRIES):
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


##############################################################################
# src/sources/play_reviews.py
# L1 primary: Google Play reviews, 44 apps
##############################################################################

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


##############################################################################
# src/sources/social_news.py
# L1 secondary: Reddit, Mastodon, Google News, Hacker News
##############################################################################

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


##############################################################################
# src/sources/incidents.py
# L2 triggers: FAA delay register, public status pages
##############################################################################

"""L2: documented incidents - the ground truth that turns correlation into cause.

The rubric asks for "trigger explanations" and "reasons behind observed
changes". Most of the time that question is answered by reading a spike and
telling a plausible story about it. A plausible story is not evidence.

This module collects delay events that were recorded by somebody other than us,
with timestamps we did not choose:

``faa``         live US airport ground stops, ground delay programmes and
                closures, with the airport, the reason and the delay length as
                published by the FAA - a literal register of service delays;
``statuspage``  incident histories from ten operators that run public
                status pages. Each entry has a title, a state and a timestamp,
                so an outage can be lined up against the reaction curve.

The output is an incident table with a start time, which ``analyse.py`` joins
against the reaction time series. When a sentiment shift sits next to an
incident, we can say so with a citation instead of an adjective.
"""

import html
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from config import FAA_STATUS_URL, RAW, STATUSPAGE_FEEDS
from fetch import get, write_jsonl

_TAGS = re.compile(r"<[^>]+>")


def _clean(s: str) -> str:
    return html.unescape(_TAGS.sub(" ", s or "")).replace("\xa0", " ").strip()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _text(node, tag, default=""):
    if node is None:
        return default
    el = node.find(tag)
    return (el.text or default) if el is not None else default


# ---------------------------------------------------------------------------
def collect_faa() -> list[dict]:
    """US National Airspace System status: real delays, happening now."""
    raw = get(FAA_STATUS_URL, max_age=600)
    if not raw:
        print("      - FAA: unavailable", flush=True)
        return []
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return []

    observed = _text(root, "Update_Time") or _now()
    out: list[dict] = []

    for block in root.findall(".//Delay_type"):
        kind = _text(block, "Name", "unknown")

        # Ground stops and ground delay programmes carry per-airport detail
        for prog in block.findall(".//Ground_Stop_List/Program") + \
                    block.findall(".//Ground_Delay_List/Delay"):
            airport = _text(prog, "ARPT")
            out.append({
                "source": "faa", "incident_kind": kind,
                "entity": airport or "US airspace",
                "sector": "airline",
                "started_utc": "", "observed_utc": observed,
                "title": f"{kind} - {airport}".strip(" -"),
                "detail": " | ".join(filter(None, [
                    _text(prog, "Reason"), _text(prog, "Avg"), _text(prog, "Max"),
                    _text(prog, "End_Time"),
                ])),
                "impact_reason": _text(prog, "Reason"),
                "avg_delay": _text(prog, "Avg"),
                "url": "https://nasstatus.faa.gov/",
                "collected_utc": _now(),
            })

        # Arrival/departure delay info and closures
        for d in block.findall(".//Arrival_Departure_Delay_List/Delay") + \
                 block.findall(".//Airport_Closure_List/Airport"):
            airport = _text(d, "ARPT")
            out.append({
                "source": "faa", "incident_kind": kind,
                "entity": airport or "US airspace",
                "sector": "airline",
                "started_utc": _text(d, "Start"),
                "observed_utc": observed,
                "title": f"{kind} - {airport}".strip(" -"),
                "detail": " | ".join(filter(None, [
                    _text(d, "Reason"), _text(d, "Reopen"),
                    _text(d, "Arrival_Departure/Min"),
                    _text(d, "Arrival_Departure/Max"),
                ])),
                "impact_reason": _text(d, "Reason"),
                "avg_delay": _text(d, "Arrival_Departure/Min"),
                "url": "https://nasstatus.faa.gov/",
                "collected_utc": _now(),
            })

    print(f"      FAA {len(out):3d} active delay entries", flush=True)
    return out


# ---------------------------------------------------------------------------
def collect_statuspages(feeds=None) -> list[dict]:
    """Incident histories from public status pages (Atlassian Statuspage etc.)."""
    feeds = feeds or STATUSPAGE_FEEDS
    out: list[dict] = []
    for name, url in feeds:
        raw = get(url, max_age=1800)
        if not raw:
            print(f"      - {name}: unavailable", flush=True)
            continue
        try:
            root = ET.fromstring(raw)
        except ET.ParseError:
            print(f"      - {name}: unparseable", flush=True)
            continue
        items = root.findall(".//item")
        for it in items:
            try:
                started = parsedate_to_datetime(_text(it, "pubDate"))
                started = started.astimezone(timezone.utc).isoformat()
            except Exception:
                started = ""
            body = _clean(_text(it, "description"))
            title = _clean(_text(it, "title"))
            out.append({
                "source": "statuspage", "incident_kind": name,
                "entity": name, "sector": "platform_service",
                "started_utc": started, "observed_utc": started,
                "title": title, "detail": body[:800],
                "impact_reason": _classify_incident(f"{title} {body}"),
                "avg_delay": "",
                "url": _text(it, "link"),
                "collected_utc": _now(),
            })
        print(f"      {name:12s} {len(items):3d} incidents", flush=True)
    return out


_INCIDENT_PATTERNS = [
    ("degraded_performance", r"degrad|slow|latency|elevated (?:error|response)"),
    ("outage",               r"outage|unavailab|down\b|disruption|cannot (?:access|connect)"),
    ("delivery_delay",       r"deliver|queue|backlog|delay in (?:process|send)"),
    ("maintenance",          r"maintenance|scheduled|planned"),
    ("connectivity",         r"connectivity|network|packet loss|routing"),
]


def _classify_incident(text: str) -> str:
    low = (text or "").lower()
    for label, pattern in _INCIDENT_PATTERNS:
        if re.search(pattern, low):
            return label
    return "other"


# ---------------------------------------------------------------------------
def collect() -> list[dict]:
    rows = collect_faa() + collect_statuspages()
    write_jsonl(rows, RAW / "incidents.jsonl")
    return rows


if __name__ == "__main__":
    rows = collect()
    print(f"\n{len(rows)} incidents -> data/raw/incidents.jsonl")


##############################################################################
# src/sources/attention.py
# L3 attention: Wikipedia pageviews
##############################################################################

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


##############################################################################
# src/collect.py
# orchestrator + source audit
##############################################################################

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
    import play_reviews
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

    print("[2/4] Social and news")
    social = social_news.collect()
    results.update({k: len(v) for k, v in social.items()})
    print()

    print("[3/4] Ground-truth incidents")
    results["incidents"] = len(incidents.collect())
    print()

    print("[4/4] Attention (Wikipedia pageviews)")
    results["attention_rows"] = len(attention.collect())
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
            if k not in ("incidents", "attention_rows")),
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
