"""Collect the four round-2 deliverables into ``round2/SUBMISSION``.

The Google Form accepts exactly one file per slot and caps each at 10 MB, so
this step also checks the sizes and fails loudly rather than letting a silent
overflow reach the upload page.
"""
from __future__ import annotations

import shutil

from config import MODEL_BUNDLE, REPORTS, ROUND2, SUBMISSION

MAX_MB = 10.0

DELIVERABLES = [
    ("1. NLP Model Script/Notebook",
     ROUND2 / "notebooks" / "Round2_NLP_Team_SE7EN.ipynb",
     "Round2_NLP_Model_Notebook_Team_SE7EN.ipynb"),
    ("2. Trained Model Files (optional)",
     MODEL_BUNDLE,
     "Round2_Trained_Models_Team_SE7EN.pkl"),
    ("3. Evaluation Metrics Report (PDF)",
     REPORTS / "Evaluation_Metrics_Report_Team_SE7EN.pdf",
     "Round2_Evaluation_Metrics_Report_Team_SE7EN.pdf"),
    ("4. Round 2 Technical Report (PDF)",
     REPORTS / "Round2_Technical_Report_Team_SE7EN.pdf",
     "Round2_Technical_Report_Team_SE7EN.pdf"),
]


def main() -> dict:
    SUBMISSION.mkdir(parents=True, exist_ok=True)
    # clear stale copies so the folder always matches the current build
    for old in SUBMISSION.glob("*"):
        if old.is_file() and old.name != "README.md":
            old.unlink()

    rows, problems = [], []
    for slot, source, target_name in DELIVERABLES:
        if not source.exists():
            problems.append(f"{slot}: missing {source}")
            continue
        target = SUBMISSION / target_name
        shutil.copy(source, target)
        size_mb = target.stat().st_size / 1e6
        if size_mb > MAX_MB:
            problems.append(f"{slot}: {target_name} is {size_mb:.2f} MB (limit {MAX_MB})")
        rows.append((slot, target_name, size_mb))

    width = max(len(r[1]) for r in rows) if rows else 40
    print(f"{'Form slot':38s} {'File':{width}s}  Size")
    for slot, name, mb in rows:
        print(f"{slot:38s} {name:{width}s}  {mb:6.2f} MB")

    lines = [
        "# Round 2 submission - Team SE7EN",
        "",
        "Tanmay Singh &middot; Panshul Arora &middot; Data Vortex A'26",
        "",
        "Upload one file per Google Form slot, in this order.",
        "",
        "| # | Form slot | File | Size |",
        "|---|---|---|---|",
    ]
    for i, (slot, name, mb) in enumerate(rows, 1):
        lines.append(f"| {i} | {slot.split('. ', 1)[1]} | `{name}` | {mb:.2f} MB |")
    lines += [
        "",
        "Every file is regenerated from the raw CSV by `python round2/run_round2.py`.",
        "",
    ]
    (SUBMISSION / "README.md").write_text("\n".join(lines), encoding="utf-8")

    if problems:
        print("\nPROBLEMS:")
        for prob in problems:
            print("  -", prob)
        raise SystemExit(1)
    print(f"\n{len(rows)}/4 deliverables staged in {SUBMISSION}")
    return {"files": rows}


if __name__ == "__main__":
    main()
