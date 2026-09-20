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
from __future__ import annotations

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
