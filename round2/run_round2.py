"""One command that rebuilds every round-2 artefact from the raw CSV.

    python round2/run_round2.py            # audit -> train -> evaluate -> PDFs
    python round2/run_round2.py --sweep    # also re-run the hyperparameter sweep

Steps are ordered by dependency; each prints what it wrote. Nothing here reads
the held-out slice except the single scoring pass inside ``evaluate``.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))


def banner(n, total, title):
    print(f"\n{'=' * 74}\n[{n}/{total}]  {title}\n{'=' * 74}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true",
                    help="re-run the hyperparameter sweep (slow; does not change HPARAMS)")
    args = ap.parse_args(argv)

    steps = [("Audit the labels", "audit_labels"),
             ("Train and select models", "train"),
             ("Evaluate on the held-out slice", "evaluate"),
             ("Build the evaluation metrics report", "build_metrics_report"),
             ("Build the technical report", "build_technical_report"),
             ("Execute the walkthrough notebook", "build_notebook"),
             ("Assemble the submission folder", "assemble_submission")]
    if args.sweep:
        steps.insert(1, ("Sweep hyperparameters", "sweep"))

    started = time.time()
    for i, (title, module) in enumerate(steps, 1):
        banner(i, len(steps), title)
        __import__(module).main()
    print(f"\nall done in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
