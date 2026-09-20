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
from __future__ import annotations

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
