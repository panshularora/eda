"""One command that rebuilds every Round 3 artefact from the live web.

    python round3/run_round3.py              # full run, re-collects everything
    python round3/run_round3.py --skip-play  # reuse the reviews already on disk
    python round3/run_round3.py --offline    # analyse what is on disk, fetch nothing

Ordered by dependency. Collection is the only step that touches the network;
everything after it is a pure function of what landed in ``data/raw``, so the
analysis is reproducible from the published raw files even after the live
endpoints have moved on - which, for a round about live data, is the whole
point.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "sources"))


def banner(i, n, title):
    print(f"\n{'=' * 72}\n[{i}/{n}]  {title}\n{'=' * 72}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-play", action="store_true",
                    help="reuse data/raw/play_reviews.jsonl instead of re-collecting")
    ap.add_argument("--offline", action="store_true",
                    help="skip collection entirely and analyse what is on disk")
    args = ap.parse_args(argv)

    steps = [
        ("Collect from public sources", "collect"),
        ("Normalise into one schema", "normalise"),
        ("Label delays, reactions and Round 2 sentiment", "classify"),
        ("Validate Round 2 transfer against star ratings", "validate"),
        ("Detect shifts and spikes, attribute triggers", "analyse"),
        ("Render figures", "figures"),
        ("Publish dataset, dictionary and quality report", "build_dataset"),
        ("Execute the analysis notebook", "build_notebook"),
        ("Stage the submission folder", "assemble_submission"),
    ]
    if args.offline:
        steps = [s for s in steps if s[1] != "collect"]

    t0 = time.time()
    for i, (title, module) in enumerate(steps, 1):
        banner(i, len(steps), title)
        mod = __import__(module)
        if module == "collect":
            mod.main(skip_play=args.skip_play)
        else:
            mod.main()
    print(f"\nall done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
