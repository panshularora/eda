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
