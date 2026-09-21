"""Render README.md from the template, filling every number from CLAIMS.json.

A README is where numbers go stale. The pipeline is re-run, a filter is
tightened, a dedupe rule is fixed - and the prose still quotes the figure from
two runs ago, because nothing connects them. Round 2 solved this by reading
every reported number out of `metrics.json`. This is the same discipline: the
template carries `{{claim}}` placeholders, `claims.py` writes what the run
actually measured, and a placeholder with no matching claim **fails the build**
rather than shipping as literal `{{...}}` in the submitted repository.
"""
from __future__ import annotations

from config import ROUND3
from claims import build, render

TEMPLATE = ROUND3 / "README_TEMPLATE.md"
OUT = ROUND3 / "README.md"


def main() -> str:
    if not TEMPLATE.exists():
        raise SystemExit(f"template not found: {TEMPLATE}")
    claims = build()
    text = render(TEMPLATE.read_text(encoding="utf-8"), claims)
    OUT.write_text(text, encoding="utf-8")
    print(f"  README.md rendered from {len(claims)} verified claims")
    return text


if __name__ == "__main__":
    main()
