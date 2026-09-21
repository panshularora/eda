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

import pathlib
import re
import shutil
from pathlib import Path

from config import NOTEBOOKS, PROCESSED, REPORTS, ROUND3, SUBMISSION, TOPIC

MAX_MB = 10.0

COLLECTION_MODULES = [
    ("src/config.py", "configuration: topic, source roster, taxonomies"),
    ("src/fetch.py", "polite HTTP: rate limits, backoff, cache, ledger, anonymisation"),
    ("src/sources/play_census.py",
     "L1 primary: Google Play census + per-brand-day quota sample, 44 apps"),
    ("src/sources/social_news.py",
     "L1 secondary: Reddit (feeds + search), Lemmy, Mastodon, Google News, Hacker News"),
    ("src/sources/incidents.py",
     "L2 triggers: FAA delay register, supply-chain status pages"),
    ("src/sources/attention.py", "L3 attention: Wikipedia pageviews"),
    ("src/waves.py", "append-only wave bookkeeping: what each run actually added"),
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

'''


def build_collection_script() -> Path:
    """Concatenate the collection package into one runnable script.

    ``from __future__`` imports must be the first statement in a file, so every
    module's copy is stripped out of the bodies and a single one is hoisted to
    the top. Without this the concatenated file raises SyntaxError on the second
    module - which is exactly what a judge would hit on first run, and is why
    this function verifies the result compiles before returning it.
    """
    future = re.compile(r"^from __future__ import .*$\n?", re.M)

    parts = [HEADER, "from __future__ import annotations\n"]
    for rel, desc in COLLECTION_MODULES:
        path = ROUND3 / rel
        if not path.exists():
            continue
        body = future.sub("", path.read_text(encoding="utf-8"))
        bar = "#" * 78
        parts.append(f"\n\n{bar}\n# {rel}\n# {desc}\n{bar}\n\n")
        parts.append(body)

    out = ROUND3 / "Round3_Collection_Script_Team_SE7EN.py"
    source = "".join(parts)
    compile(source, str(out), "exec")          # fail here, not on the judge's machine
    out.write_text(source, encoding="utf-8")
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
        # The census is what makes the sample a survey rather than a
        # convenience sample, so it travels with the dataset: 44 brands x 45
        # days of enumerated review counts, ~600 KB, and the thing a judge
        # needs in order to check the population estimate themselves.
        (PROCESSED / "round3_play_census.csv", "companion_play_census.csv"),
        (REPORTS / "relevance_audit.md", "relevance_audit.md"),
        (REPORTS / "relevance_audit_sample.csv", "relevance_audit_sample.csv"),
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
    for slot, path in missing:
        lines.append(f"| {len(rows) + 1} | {slot.split('. ', 1)[1]} | "
                     f"**NOT BUILT** — expected at `{pathlib.Path(path).name}` | — |")
    lines += ["", "Companion material (not a form slot):", "",
              "| file | what it is |", "|---|---|",
              "| `DATA_DICTIONARY.md` | every column, its type, completeness and provenance |",
              "| `companion_play_census.csv` | enumerated review counts per brand-day — the sampling frame, and what makes the population estimate checkable |",
              "| `companion_incidents.csv` | documented incidents with start times, each tagged `direct` or `infra` |",
              "| `companion_attention.csv` | daily Wikipedia pageviews per brand |",
              "| `relevance_audit.md` + `relevance_audit_sample.csv` | the hand-adjudicated topic-filter audit and the 120 rows it read |",
              "", "Everything is regenerated by `python round3/run_round3.py`.", ""]
    (SUBMISSION / "README.md").write_text("\n".join(lines), encoding="utf-8")

    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print("  -", p)
        raise SystemExit(1)
    print(f"\n{len(rows)}/4 deliverables staged in {SUBMISSION}")
    if missing:
        # Loud, not a footnote. A form slot with no file is an incomplete
        # submission, and the first version of this assembler printed the
        # absence as a parenthetical that was easy to read past.
        print()
        print("!" * 66)
        for slot, path in missing:
            print(f"!! FORM SLOT NOT FILLED: {slot}")
            print(f"!!   expected: {path}")
        print("!" * 66)
    return {"staged": rows, "missing": missing}


if __name__ == "__main__":
    main()
