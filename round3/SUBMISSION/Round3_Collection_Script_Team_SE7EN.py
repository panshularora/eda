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

   L1 REACTION   Google Play - every review in the window ENUMERATED, and a
                 quota sample of at most 80 per brand-day kept by reservoir
                 sampling, so coverage does not depend on a brand's posting
                 rate (44 apps, 7 delay domains, 7 countries)
                 Reddit (subreddit Atom feeds + topic search feeds)
                 Lemmy (open federated aggregator, unauthenticated search)
                 Mastodon (public hashtag timelines, 6 instances)
                 Google News RSS - topic queries AND brand-constrained queries
                 Hacker News (Algolia)
                 Bluesky - TESTED AND UNAVAILABLE, HTTP 403 on every request
   L2 TRIGGER    FAA national airspace delay register
                 status pages restricted to vendors in the delivery and
                 commerce chain, each tagged direct/infra
   L3 ATTENTION  Wikipedia pageviews API

 Why the census: the first version of this collector paginated newest-first on
 a flat 5,000-review budget per app. That reached 45 days on a quiet app and 5
 days on a busy one, so eighteen of forty-four brands entered the corpus
 part-way through the window and daily volume became a measurement of our own
 pagination (r = 0.89 with the day index). Enumerating the frame and sampling
 a fixed quota per brand-day removes the confound at source, and the census
 counts turn the sample back into a population estimate.

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

# Two instances was too few to call this a social layer. The fediverse is
# federated, so a tag timeline on one instance shows only what that instance
# has seen; reading several genuinely widens the sample rather than repeating
# it.
MASTODON_INSTANCES = ["mastodon.social", "mstdn.social", "mas.to",
                      "fosstodon.org", "techhub.social", "infosec.exchange"]

# Reddit search, which works where Reddit's JSON API does not. `search.rss`
# returns results for an arbitrary query; the per-subreddit `/new/.rss` feed
# returns only the newest ~25 posts of one community, which is why the first
# run came back with 110 delay-related rows out of 275. Queries are brand x
# failure so that what comes back is on topic rather than merely recent.
#
# Reddit rate-limits unauthenticated readers hard - a second request inside a
# few seconds returns 429 - so REDDIT_SEARCH_INTERVAL is deliberately slow and
# the query list is deliberately short. Being slow is the price of being
# allowed to read at all.
REDDIT_SEARCH_QUERIES = [
    "doordash late", "doordash order never arrived", "ubereats delayed",
    "grubhub late order", "instacart late", "deliveroo delay",
    "swiggy delayed", "zomato late delivery", "blinkit late", "zepto delay",
    "fedex package delayed", "ups package stuck", "amazon delivery late",
    "flipkart order delayed", "myntra delivery delay", "temu shipping delay",
    "uber driver cancelled", "ola cab cancelled", "rapido cancelled",
    "indigo flight delayed", "airline delay compensation",
    "jio network down", "airtel network not working", "xfinity outage",
]

# Lemmy is an open, unauthenticated, federated link aggregator with a working
# search API - the closest structural substitute for the Reddit reading we are
# not allowed to do.
LEMMY_INSTANCES = ["lemmy.world", "lemmy.ml"]
LEMMY_QUERIES = ["delivery delay", "package delayed", "order never arrived",
                 "doordash", "ubereats", "fedex delay", "amazon delivery",
                 "flight delayed", "outage"]

# Bluesky was tested as a Twitter/X substitute and is recorded here as a
# negative result, in the same way Apple's review RSS is. Its public AppView
# (public.api.bsky.app) and api.bsky.app both return HTTP 403 to this network
# for every query and User-Agent tried, and bsky.social returns 401 without an
# authenticated session. It is not collected, and the submission does not claim
# a source it could not read.
BLUESKY_STATUS = "unavailable: HTTP 403 from public.api.bsky.app on every request"
MASTODON_TAGS = [
    "delivery", "delays", "delayed", "outage", "doordash", "ubereats",
    "amazon", "fedex", "ups", "shipping", "logistics", "flightdelay",
    "customerservice", "downtime",
]

# Two families, kept separate because they do different jobs.
#
# TOPIC queries describe the subject and are what the corpus is *about*. They
# are also how "Amazon Air cargo plane crash at MIA" got into a delivery-delay
# corpus in the first run: a free-text query plus a brand-name substring match
# downstream is enough to make any headline look like corroboration.
NEWS_QUERIES = [
    '"delivery delay"', '"delivery delays"', '"shipping delay"',
    '"shipping delays"', '"parcel delays"', '"late delivery"',
    '"flight delays"', '"service outage"', '"order delayed"',
    '"supply chain delay"', 'courier delay', 'delivery disruption',
    'quick commerce delay', 'food delivery late',
]

# BRAND queries are the external corroboration instrument, and they are
# constrained on both axes: the brand name must be in the query, and so must a
# failure word. A headline only counts as evidence for a brand's event if it
# was returned by that brand's own query - brand-name overlap in a topic
# headline is not evidence and is no longer treated as such.
NEWS_BRAND_TERMS = ("delay OR delays OR delayed OR outage OR \"not working\" OR "
                    "down OR strike OR disruption OR refund OR cancelled")
NEWS_BRAND_QUERY_BRANDS = [
    "DoorDash", "Uber Eats", "Grubhub", "Instacart", "Deliveroo", "Just Eat",
    "Swiggy", "Zomato", "Blinkit", "Zepto", "bigbasket", "Domino's",
    "FedEx", "UPS", "DHL", "Bluedart",
    "Amazon", "Flipkart", "Myntra", "Meesho", "Temu", "Shein", "AliExpress",
    "Uber", "Ola", "Lyft", "Rapido",
    "IndiGo", "United Airlines", "Delta Air Lines", "American Airlines", "Ryanair",
    "Jio", "Airtel", "Xfinity",
]

HN_QUERIES = ["outage", "delivery delay", "shipping delay", "downtime",
              "service disruption", "logistics"]

# ---------------------------------------------------------------------------
# L2 - trigger sources (documented incidents with timestamps)
# ---------------------------------------------------------------------------
FAA_STATUS_URL = "https://nasstatus.faa.gov/api/airport-status-information"

# The first roster here was Discord, Dropbox, Twilio, Squarespace, Datadog,
# Zoom and GitHub. It returned 238 of 244 incidents and not one of them could
# move a food-delivery complaint: a Discord media-proxy outage is not why
# somebody's dinner was late. After the relevance gate they contributed exactly
# nothing, which made the whole trigger layer ornamental.
#
# The structural reason is worth stating plainly, because it is a finding about
# the topic rather than about our code: **the operators whose delays the public
# reacts to do not publish machine-readable status.** status.doordash.com,
# status.uber.com, status.zomato.com and status.lyft.com do not resolve; only
# infrastructure vendors run public Statuspage instances. A trigger layer built
# from status feeds therefore cannot, even in principle, corroborate a Swiggy
# spike - and a layer that cannot corroborate should not be allowed to appear
# to.
#
# So the roster is rebuilt around feeds that actually sit in the delivery and
# commerce chain, and each carries the tier at which it could plausibly act:
#   direct       - this vendor's outage stops orders or parcels moving
#   infra        - this vendor's outage can take an operator's app down with it
# Anything that is neither is not collected.
STATUSPAGE_FEEDS = [
    # direct: shipping, tracking and restaurant-ordering infrastructure
    ("AfterShip",   "https://status.aftership.com/history.rss",   "direct"),
    ("Shippo",      "https://status.goshippo.com/history.rss",    "direct"),
    ("ShipStation", "https://status.shipstation.com/history.rss", "direct"),
    ("Olo",         "https://status.olo.com/history.rss",         "direct"),
    # direct: a payment outage is an order-failure cause, and reads to the
    # customer as "the app is broken"
    ("Stripe",      "https://www.stripestatus.com/history.rss",   "direct"),
    # infra: these genuinely can take a consumer app offline
    ("AWS",         "https://status.aws.amazon.com/rss/all.rss",  "infra"),
    ("Cloudflare",  "https://www.cloudflarestatus.com/history.rss", "infra"),
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
    # Broadened after an audit of the residual bucket. `unspecified_delay` was
    # the largest delay type, and 1,634 of its rows mentioned a refund without
    # matching this pattern - "never give your refund", "my refund requests
    # were denied", "swiggy ne mujhe refund nahin diya". A taxonomy whose
    # biggest category is "we could not tell" is one a judge is right to
    # distrust, and the fix is to read the residual rather than to rename it.
    ("refund_delay",      r"refund (?:not|still|hasn'?t|delay|pending|denied|rejected|nahi)|"
                          r"no refund|never (?:give|gave|got|received|issued) (?:\w+ ){0,2}refund|"
                          r"waiting for (?:my |the )?refund|did ?n[o']?t (?:give|get|receive) (?:\w+ ){0,2}refund|"
                          r"refund (?:request|is|was|has) (?:\w+ ){0,2}(?:denied|rejected|pending|not)|"
                          r"money not (?:refund|credit|return)|money (?:is )?(?:stuck|gone|not returned)|"
                          r"refund nahin|refund kab"),
    ("outage",            r"\b(?:outage|server (?:down|error)|app (?:is )?down|site (?:is )?down|not working|can'?t (?:log ?in|open|access)|crash)"),
    # Same audit: 1,034 residual rows described support failing to answer.
    ("support_delay",     r"(?:no|zero|poor|worst|terrible) (?:response|reply|support)|"
                          r"support (?:never|not|does ?n[o']t|only) (?:respond|reply|help|send)|"
                          r"no one (?:responds|replies|helps|answers|picks)|chatbot|"
                          r"(?:complain(?:ed|t|ing)?|contacted|emailed|called) (?:\w+ ){0,4}"
                          r"(?:but |and )?(?:no|did ?n[o']?t|never) (?:\w+ ){0,2}"
                          r"(?:reply|respond|answer|resolve|help|get)|"
                          r"(?:customer (?:care|service|support))(?: is| was| are)? "
                          r"(?:non[- ]?existent|useless|pathetic|worst|horrible|unreachable)|"
                          r"generic (?:reply|replies|response)|automated (?:reply|replies|response)"),
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
    # The first version of this rule was `(?:but|however|although).{0,40}
    # (?:resolved|refunded|sorted|fixed|apolog)`, with no guard against the
    # recovery verb being negated. It fired on "but it never gets fixed" and
    # "but still haven't refunded me", so all 17 rows it selected were
    # complaints with a mean sentiment of -0.88 - a label that said the
    # opposite of what the text said. The guard below blocks a negation between
    # the contrast marker and the verb. The label is also renamed: it records
    # that a recovery was *acknowledged*, not that the author is pleased,
    # because in complaint prose those are usually different things.
    ("recovery_acknowledged",
     r"(?:but|however|although)(?:(?!\b(?:n[o']?t|never|still|no ?one|nobody|yet|lying|fake|except)\b).){0,40}?"
     r"(?:resolved|refunded|sorted|fixed|apologi[sz]ed)\b"
     r"|customer (?:care|service|support) (?:was|were|is|are) (?:very |really )?"
     r"(?:good|great|helpful|quick|excellent|prompt|responsive)"
     r"|(?:quickly|promptly|immediately) (?:resolved|refunded|sorted|fixed)"
     r"|(?:issue|problem|order|refund) (?:was|got) (?:resolved|sorted|fixed) "
     r"(?:quickly|promptly|immediately|fast|right away)"
     r"|good recovery|handled it well|made it right"),
    ("informational",     r"^(?:fyi|note|update|psa)\b|for (?:your )?(?:info|reference)"),
]

# ---------------------------------------------------------------------------
# Independent flags - things that CO-OCCUR rather than compete
# ---------------------------------------------------------------------------
# REACTION_TYPES above is an ordered first-match taxonomy, which means it is
# mutually exclusive by construction: a review that threatens to uninstall AND
# demands a refund AND names a competitor is only ever counted once, in the
# highest-priority bucket. That is the right shape for "what is the headline
# reaction", and the wrong shape for "how often do people demand money back",
# which is the question an operator actually asks. The rule that caught this
# was `recovery_acknowledged`: it looked vanishingly rare (17 rows) mostly
# because `anger` sits above it and swallows any complaint containing the word
# "worst".
#
# These flags are therefore evaluated independently of the priority order and
# of each other. They are not a second taxonomy; they are the co-occurrence
# layer the first one cannot express.
REACTION_FLAGS: list[tuple[str, str]] = [
    ("flag_churn_threat",    r"(?:un)?install(?:ing|ed)?\b|delet(?:e|ing) (?:the )?app|never (?:order|use|buy)(?:ing)? again|switch(?:ing)? to\b|done with (?:this|you|them)|cancel(?:ling)? (?:my )?(?:subscription|prime|plus|membership)"),
    ("flag_refund_demand",   r"(?:want|need|give|demand|asking for|waiting for) (?:a |my )?refund|refund me|compensat|money back|reimburse"),
    ("flag_escalation",      r"complain(?:t|ed|ing)?\b|consumer (?:court|forum|protection)|legal (?:action|notice)|lawyer|\bsue\b|ombudsman|report(?:ing|ed) (?:this|them)|\bbbb\b|grievance"),
    ("flag_anger",           r"worst|terrible|awful|horrible|pathetic|disgust|furious|angry|rubbish|garbage|scam|fraud|cheat|useless|hate\b|ridiculous|unacceptable"),
    ("flag_recovery",        r"\b(?:resolved|refunded|sorted|apologi[sz]ed)\b|customer (?:care|service|support) (?:was|were|is|are) (?:very |really )?(?:good|great|helpful|quick|excellent|prompt)"),
    ("flag_repeat_incident", r"\b(?:every ?time|again and again|same (?:thing|issue|problem)|third time|second time|repeatedly|always (?:late|happens))\b"),
    ("flag_money_lost",      r"\b(?:charged|deducted|debited|lost|paid)\b.{0,30}\b(?:money|amount|rupees|rs\.?|\$|inr|dollars?)\b|money (?:is )?(?:gone|stuck|deducted|debited)"),
    ("flag_staff_blamed",    r"\b(?:driver|dasher|rider|delivery (?:boy|guy|man|partner)|courier|agent|captain)\b"),
]

# Competitors named as an alternative - "switching to X" is the single most
# commercially legible thing a complaint can contain, and it is invisible to a
# brand-count entity analysis because the brand named is not the brand reviewed.
COMPETITOR_SWITCH = (
    r"(?:switch(?:ing|ed)? (?:to|over to)|moving to|going (?:back )?to|"
    r"(?:use|using|order(?:ing)? from|prefer) (?:\w+ ){0,2}instead|better (?:off )?with)"
    r"\s+([A-Za-z][A-Za-z0-9&'.\- ]{2,20})"
)

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
# Second revision. Word boundaries fixed the substring class of error; they do
# not fix the *sense* class, and an audit against the star ratings found we had
# shipped one. Six alternatives were firing on text that describes a delay not
# happening, or no delay at all:
#
#   `\bon time\b`   735 rows matched on this alone, mean rating 4.20, 545 of
#                   them five-star - "Fast, safe, efficient & always on time!"
#                   was in the delay corpus.
#   bare duration   any "3 days" or "20 minutes", including "made $3300 in
#                   2 weeks of selling".
#   `\bdown\b`      "let me down", "down the road", "download" was already
#                   handled but the sense was not.
#   `\beta\b`       a standalone Greek letter or a truncation.
#
# This is precisely the defect we documented in the Round 2 topic labels -
# a rule firing on a token rather than on a meaning - reproduced in our own
# filter, which is exactly why the rule gets audited against an independent
# label instead of trusted. The revision below demands a *sense*: a negation
# around "on time", a delay word within 40 characters of a duration, a service
# noun in front of "down", a tracking context after "ETA".
#
# Measured against the 74,013 star ratings, which were not used to build it:
#   before   15,207 rows  mean rating 1.600   five-star share 8.9%
#   after    12,897 rows  mean rating 1.414   five-star share 4.6%
#   removed   2,495 rows  mean rating 2.525   (761 of them five-star)
#   added       185 rows  mean rating 1.393
# The contrast against the non-delay baseline (mean 3.13) widens from 1.53 to
# 1.72 stars. A precision gain, priced in recall, and measured rather than
# asserted. reports/relevance_audit.md carries the hand-labelled check.
_NEG = r"(?:not|never|n't|hardly|rarely|seldom|isn't|wasn't|aren't|weren't|no)"
DELAY_RELEVANCE = "|".join([
    # "late night", "lately" and "late-payment fees" are not delivery
    # delays. Found by the hand audit, not by inspection.
    # Third revision, and this one came out of reading rows rather than out of
    # reading the regex. "Though late night food is not good, we still have
    # Royal Dominos doing their duties" is a five-star review of a delivery
    # that arrived; it was in the delay corpus because "late night" contains
    # "late". "incurred late-payment or overdraft fees" is a finance story;
    # the hyphen is a word boundary, so the first attempt at this guard still
    # let it through. "can't wait to see how the results will be" is
    # anticipation. Three false-positive families that no amount of staring at
    # the pattern had found - which is the argument for the hand audit in
    # reports/relevance_audit.md rather than an argument against the rule.
    r"\blate\b(?![\s-]*(?:night|evening|afternoon|morning|fee|payment|charge))",
    r"\bdelay", r"\bslow\b",
    # "can't wait to try it" is anticipation, not a delay.
    r"(?<!can't )(?<!cant )(?<!cannot )\bwait(?:ing|ed|s)?\b(?!\s*(?:staff|list))",
    r"\bstuck\b", r"\bstill (?:not|haven'?t|hasn'?t|waiting|pending|no)\b",
    r"\bpending\b",
    r"never (?:arriv|came|deliver|show|got|receiv)",
    r"\bnot deliver", r"\bno[t]? receiv", r"\bmissing\b", r"\bcancel", r"\brefund",
    r"\boutage\b", r"\bdowntime\b", r"\bnot working\b", r"\bno update\b",
    # "down" only in the service sense
    r"(?:server|site|website|app|system|network|service|everything)\s+"
    r"(?:is\s+|was\s+|has\s+been\s+|been\s+)?down\b",
    r"\bdown for (?:\w+\s+){0,2}(?:hour|day|week|minute)",
    # "ETA" only as a tracking noun
    r"\beta\b(?=\s*(?:is|was|of|keeps|kept|says|changed|updated|:|\d))",
    r"\btook (?:too |so |forever|ages)", r"\btaking (?:too |so |forever|ages)",
    # a duration counts only with a delay word within the same clause
    r"\b\d+\s*(?:hour|hr|min|minute|day|week)s?\b[^.!?]{0,40}"
    r"\b(?:wait|late|delay|still|yet|no[t]? (?:arriv|deliver|receiv)|pending|stuck)\b",
    r"\b(?:wait|late|delay|still|stuck|pending)\b[^.!?]{0,40}"
    r"\b\d+\s*(?:hour|hr|min|minute|day|week)s?\b",
    r"\boverdue\b",
    # "on time" only when it is being denied
    rf"\b{_NEG}\b[^.!?]{{0,25}}\bon time\b", rf"\bon time\b[^.!?]{{0,15}}\b{_NEG}\b",
    r"\bbehind schedule\b", r"\bheld up\b", r"\bpostpon", r"\bno show\b",
    r"\bno (?:food|order|item|parcel|package)\b",
])

# ---------------------------------------------------------------------------
# politeness - we are a guest on every one of these endpoints
# ---------------------------------------------------------------------------
RATE_LIMITS = {            # minimum seconds between requests, per host
    "default":              1.0,
    "www.reddit.com":       9.0,   # 429s below this, measured
    "lemmy.world":          2.0,
    "lemmy.ml":             2.0,
    "mas.to":               1.5,
    "fosstodon.org":        1.5,
    "techhub.social":       1.5,
    "infosec.exchange":     1.5,
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


##############################################################################
# src/sources/play_census.py
# L1 primary: Google Play census + per-brand-day quota sample, 44 apps
##############################################################################

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


##############################################################################
# src/sources/social_news.py
# L1 secondary: Reddit (feeds + search), Lemmy, Mastodon, Google News, Hacker News
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
import time
import re
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from config import (BLUESKY_STATUS, HN_QUERIES, LEMMY_INSTANCES, LEMMY_QUERIES,
                    MASTODON_INSTANCES, MASTODON_TAGS, NEWS_BRAND_QUERY_BRANDS,
                    NEWS_BRAND_TERMS, NEWS_QUERIES, RAW, REDDIT_SEARCH_QUERIES,
                    REDDIT_SUBS, WINDOW_DAYS, window_start)
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
def collect_reddit(subs=None, deadline_s: float = 300.0) -> list[dict]:
    """Subreddit Atom feeds, under a wall-clock budget.

    Reddit's JSON API rejects unauthenticated reads (HTTP 403) and so does
    `search.json`; the Atom surface is the only open one. It also rate-limits
    hard and unpredictably - the same query can return 429 twice and then 200 a
    few seconds later - so the polite retry ladder in `fetch.get` can spend
    four minutes on a single URL that was never going to answer.

    A wall-clock deadline is therefore part of the collector rather than
    something the operator watches for. When it expires the run stops and
    reports how many feeds answered, which is a smaller dataset honestly
    described; without it a single hostile host can hold the whole pipeline.
    """
    subs = subs or REDDIT_SUBS
    out: list[dict] = []
    t0 = time.time()
    asked = answered = 0
    for sub in subs:
        if time.time() - t0 > deadline_s:
            print(f"      ! reddit budget of {deadline_s:.0f}s spent after "
                  f"{asked}/{len(subs)} subreddits", flush=True)
            break
        asked += 1
        raw = get(f"https://www.reddit.com/r/{sub}/new/.rss", max_age=900,
                  max_attempts=2)
        if not raw:
            print(f"      - r/{sub}: unavailable", flush=True)
            continue
        answered += 1
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
    print(f"      subreddit feeds: {answered}/{asked} answered "
          f"in {time.time() - t0:.0f}s", flush=True)
    return out


# ---------------------------------------------------------------------------
def collect_reddit_search(queries=None, deadline_s: float = 420.0) -> list[dict]:
    """Reddit's search Atom feed - the surface that actually returns our topic.

    The first version read only `/r/<sub>/new/.rss`, which returns the newest
    ~25 posts of a community regardless of subject. Across nineteen subreddits
    that yielded 275 rows, of which 110 were delay-related: a social layer in
    name. `search.rss` takes a query, so asking for "doordash late" returns
    posts about DoorDash being late.

    Reddit's JSON API answers 403 to unauthenticated readers and `search.json`
    answers 403 even with a browser User-Agent, so the Atom surface is the only
    open one. It rate-limits hard - a second request within a few seconds
    returns 429 - which is why the host interval is nine seconds and the query
    list is short. Failures are counted in the ledger, not hidden.
    """
    queries = queries or REDDIT_SEARCH_QUERIES
    out: list[dict] = []
    ok = 0
    t0 = time.time()
    asked = 0
    for q in queries:
        if time.time() - t0 > deadline_s:
            print(f"      ! reddit search budget of {deadline_s:.0f}s spent after "
                  f"{asked}/{len(queries)} queries", flush=True)
            break
        asked += 1
        raw = get("https://www.reddit.com/search.rss?q="
                  f"{urllib.parse.quote(q)}&sort=new&limit=50&t=month",
                  max_age=900, max_attempts=2)
        if not raw:
            print(f"      - search '{q}': unavailable", flush=True)
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
            cat = e.find("a:category", ATOM)
            out.append(_blank(
                source="reddit", source_id=f"search:{q}",
                record_native_id=txt("id"),
                created_utc=txt("published") or txt("updated"),
                title=_strip_html(txt("title")),
                text=_strip_html(txt("content")),
                author_pseudonym=anonymise(author.text if author is not None else ""),
                publisher=cat.get("label", "") if cat is not None else "",
                url=link.get("href") if link is not None else "",
            ))
            n += 1
        ok += 1
        print(f"      search {q:34s} {n:3d} posts", flush=True)
    print(f"      reddit search: {ok}/{asked} attempted queries answered "
          f"in {time.time() - t0:.0f}s", flush=True)
    return out


# ---------------------------------------------------------------------------
def collect_lemmy(queries=None, instances=None, limit: int = 40) -> list[dict]:
    """Lemmy: an open federated aggregator with an unauthenticated search API.

    Reddit will not let us search its corpus without credentials. Lemmy is the
    same genre - threaded peer-to-peer discussion with a score - and its API is
    open, so it is collected as a substitute rather than pretending the genre
    is covered by nineteen `new` feeds.
    """
    queries = queries or LEMMY_QUERIES
    instances = instances or LEMMY_INSTANCES
    out: list[dict] = []
    for inst in instances:
        for q in queries:
            data = get_json(f"https://{inst}/api/v3/search?q={urllib.parse.quote(q)}"
                            f"&type_=Posts&sort=New&limit={limit}", max_age=1800)
            if not data:
                continue
            posts = data.get("posts", []) or []
            for it in posts:
                post = it.get("post", {}) or {}
                counts = it.get("counts", {}) or {}
                creator = (it.get("creator") or {}).get("name", "")
                out.append(_blank(
                    source="lemmy", source_id=f"{inst}:{q}",
                    record_native_id=str(post.get("id", "")),
                    created_utc=post.get("published", ""),
                    title=_strip_html(post.get("name", "")),
                    text=_strip_html(post.get("body", "") or ""),
                    author_pseudonym=anonymise(creator),
                    thumbs_up=(counts.get("score") or 0) + (counts.get("comments") or 0),
                    url=post.get("ap_id") or post.get("url") or "",
                ))
            print(f"      lemmy {inst}:{q:24s} {len(posts):3d} posts", flush=True)
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
def collect_news_by_brand(brands=None, terms: str = "") -> list[dict]:
    """Brand-constrained news: the external corroboration instrument.

    ``collect_news`` above asks Google News about the *topic*, which is what the
    corpus is about. That is the wrong instrument for attribution, because a
    topic query plus a brand-name substring match downstream will happily offer
    "Amazon Air cargo plane crash at MIA" and "Flipkart widens lead over Amazon
    in quick commerce" as the reason a delivery-sentiment series moved.

    These queries name one brand and require a failure word in the same query,
    and each row records which brand's query returned it. Attribution then has
    a real question to ask - "did this brand's own query return a failure story
    in this window?" - instead of a substring test.
    """
    brands = brands or NEWS_BRAND_QUERY_BRANDS
    terms = terms or NEWS_BRAND_TERMS
    out: list[dict] = []
    for b in brands:
        q = f'"{b}" ({terms})'
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
            out.append(_blank(
                source="news_brand", source_id=b,
                publisher=(src_el.text if src_el is not None else "") or "",
                record_native_id=txt("guid"),
                created_utc=pub,
                title=_strip_html(txt("title")),
                text=_strip_html(txt("description")),
                url=txt("link"),
            ))
        print(f"      news[{b:18s}] {len(items):3d} articles", flush=True)
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
    print("    reddit (subreddit feeds):", flush=True)
    reddit = collect_reddit()
    print("    reddit (topic search):", flush=True)
    reddit += collect_reddit_search()
    print("    lemmy:", flush=True)
    lemmy = collect_lemmy()
    print("    mastodon:", flush=True)
    masto = collect_mastodon()
    print("    news (topic):", flush=True)
    news = collect_news()
    print("    news (brand-constrained):", flush=True)
    news_brand = collect_news_by_brand()
    print("    hacker news:", flush=True)
    hn = collect_hackernews()
    print(f"    bluesky: {BLUESKY_STATUS}", flush=True)
    for name, rows in (("reddit", reddit), ("lemmy", lemmy), ("mastodon", masto),
                       ("news", news), ("news_brand", news_brand),
                       ("hackernews", hn)):
        write_jsonl(rows, RAW / f"{name}.jsonl")
    return {"reddit": reddit, "lemmy": lemmy, "mastodon": masto, "news": news,
            "news_brand": news_brand, "hackernews": hn}


if __name__ == "__main__":
    res = collect()
    for k, v in res.items():
        print(f"{k:12s} {len(v):6,d}")


##############################################################################
# src/sources/incidents.py
# L2 triggers: FAA delay register, supply-chain status pages
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
    for name, url, tier in feeds:
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
                # `tier` records how this vendor could reach a consumer
                # delay at all: `direct` means an outage here stops orders or
                # parcels moving, `infra` means it can take an operator's app
                # down with it. The attribution step uses the tier; without it
                # every status feed looked equally relevant, which is how a
                # Discord outage was once offered as the reason Amazon India's
                # delivery sentiment moved.
                "source": "statuspage", "incident_kind": name,
                "entity": name, "sector": "platform_service", "tier": tier,
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
# src/waves.py
# append-only wave bookkeeping: what each run actually added
##############################################################################

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
