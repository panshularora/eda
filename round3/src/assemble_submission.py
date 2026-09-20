"""Stage the Round 3 deliverables into ``round3/SUBMISSION``.

The form takes one file per slot, so this produces exactly one file per slot,
checks each against the upload budget, and fails loudly rather than letting an
oversized file reach the upload page.

The scraping-code slot takes a single script, but the collector is a package.
Rather than submit one file and hide the rest, the whole collection layer is
concatenated into one runnable, readable script with its module boundaries
preserved as banners - so the judge reads everything that actually ran.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from config import NOTEBOOKS, PROCESSED, REPORTS, ROUND3, SUBMISSION, TOPIC

MAX_MB = 10.0

COLLECTION_MODULES = [
    ("src/config.py", "configuration: topic, source roster, taxonomies"),
    ("src/fetch.py", "polite HTTP: rate limits, backoff, cache, ledger, anonymisation"),
    ("src/sources/play_reviews.py", "L1 primary: Google Play reviews, 44 apps"),
    ("src/sources/social_news.py", "L1 secondary: Reddit, Mastodon, Google News, Hacker News"),
    ("src/sources/incidents.py", "L2 triggers: FAA delay register, public status pages"),
    ("src/sources/attention.py", "L3 attention: Wikipedia pageviews"),
    ("src/collect.py", "orchestrator + source audit"),
]

HEADER = f'''"""
=============================================================================
 DATA VORTEX A'26 - ROUND 3 - DATA COLLECTION / SCRAPING SCRIPT
 Team SE7EN: Tanmay Singh, Panshul Arora
 Topic: {TOPIC}
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

'''


def build_collection_script() -> Path:
    parts = [HEADER]
    for rel, desc in COLLECTION_MODULES:
        path = ROUND3 / rel
        if not path.exists():
            continue
        bar = "#" * 78
        parts.append(f"\n\n{bar}\n# {rel}\n# {desc}\n{bar}\n\n")
        parts.append(path.read_text(encoding="utf-8"))
    out = ROUND3 / "Round3_Collection_Script_Team_SE7EN.py"
    out.write_text("".join(parts), encoding="utf-8")
    return out


def main() -> dict:
    script = build_collection_script()

    deliverables = [
        ("1. Self-Collected Structured Dataset (CSV/JSON)",
         SUBMISSION / "Round3_Delay_Reactions_Dataset_Team_SE7EN.csv",
         "Round3_Delay_Reactions_Dataset_Team_SE7EN.csv"),
        ("2. Data Collection / Scraping Code (Python)",
         script, "Round3_Collection_Script_Team_SE7EN.py"),
        ("3. Analysis Notebook",
         NOTEBOOKS / "Round3_Analysis_Team_SE7EN.ipynb",
         "Round3_Analysis_Notebook_Team_SE7EN.ipynb"),
        ("4. Round 3 Analytical Report (PDF)",
         REPORTS / "Round3_Analytical_Report_Team_SE7EN.pdf",
         "Round3_Analytical_Report_Team_SE7EN.pdf"),
    ]

    SUBMISSION.mkdir(parents=True, exist_ok=True)
    rows, problems, missing = [], [], []
    for slot, source, target_name in deliverables:
        target = SUBMISSION / target_name
        if not source.exists():
            missing.append((slot, str(source)))
            continue
        if source.resolve() != target.resolve():
            shutil.copy(source, target)
        mb = target.stat().st_size / 1e6
        if mb > MAX_MB:
            problems.append(f"{slot}: {target_name} is {mb:.2f} MB (cap {MAX_MB})")
        rows.append((slot, target_name, mb))

    # companion material that does not occupy a form slot
    extras = [
        (PROCESSED / "DATA_DICTIONARY.md", "DATA_DICTIONARY.md"),
        (PROCESSED / "round3_incidents.csv", "companion_incidents.csv"),
        (PROCESSED / "round3_attention.csv", "companion_attention.csv"),
    ]
    for src, name in extras:
        if src.exists():
            shutil.copy(src, SUBMISSION / name)

    width = max((len(r[1]) for r in rows), default=44)
    print(f"{'Form slot':46s} {'File':{width}s}  Size")
    for slot, name, mb in rows:
        print(f"{slot:46s} {name:{width}s}  {mb:6.2f} MB")
    for slot, path in missing:
        print(f"{slot:46s} {'-- not built --':{width}s}")

    lines = [
        "# Round 3 submission - Team SE7EN", "",
        f"**Topic:** {TOPIC}", "",
        "Tanmay Singh &middot; Panshul Arora &middot; Data Vortex A'26", "",
        "Upload one file per Google Form slot, in this order.", "",
        "| # | Form slot | File | Size |", "|---|---|---|---|",
    ]
    for i, (slot, name, mb) in enumerate(rows, 1):
        lines.append(f"| {i} | {slot.split('. ', 1)[1]} | `{name}` | {mb:.2f} MB |")
    lines += ["", "Companion material (not a form slot): `DATA_DICTIONARY.md`, "
                  "`companion_incidents.csv`, `companion_attention.csv`.", "",
              "Everything is regenerated by `python round3/run_round3.py`.", ""]
    (SUBMISSION / "README.md").write_text("\n".join(lines), encoding="utf-8")

    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print("  -", p)
        raise SystemExit(1)
    print(f"\n{len(rows)}/4 deliverables staged in {SUBMISSION}")
    if missing:
        print(f"({len(missing)} not built yet: "
              + ", ".join(s.split('. ', 1)[1] for s, _ in missing) + ")")
    return {"staged": rows, "missing": missing}


if __name__ == "__main__":
    main()
