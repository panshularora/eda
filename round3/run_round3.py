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
    ap.add_argument("--wave", action="store_true",
                    help="record a collection wave without rebuilding the analysis")
    args = ap.parse_args(argv)

    if args.wave:
        import waves
        w = waves.record_wave()
        print(f"wave {w['wave']}: {w['total_new_this_wave']:,} records not seen before")
        waves.report()
        return 0

    steps = [
        ("Collect from public sources", "collect"),
        ("Normalise into one schema", "normalise"),
        ("Label delays, reactions and Round 2 sentiment", "classify"),
        ("Validate Round 2 transfer against star ratings", "validate"),
        ("Detect shifts and spikes, attribute triggers", "analyse"),
        ("Classify the brand-days that are events", "cases"),
        ("Render figures", "figures"),
        ("Publish dataset, dictionary and quality report", "build_dataset"),
        ("Execute the analysis notebook", "build_notebook"),
        ("Collect every reported number into CLAIMS.json", "claims"),
        ("Render README.md from the verified claims", "build_readme"),
        ("Stage the submission folder", "assemble_submission"),
        ("Pre-flight: check the pack the way a judge would", "preflight"),
    ]
    if args.offline:
        steps = [s for s in steps if s[1] != "collect"]

    t0 = time.time()
    for i, (title, module) in enumerate(steps, 1):
        banner(i, len(steps), title)
        mod = __import__(module)
        if module == "collect":
            mod.main(skip_play=args.skip_play)
        elif module == "preflight":
            # Reports rather than aborts: the pipeline has already produced
            # everything by this point, and a failed check is information the
            # operator needs on screen, not an exception that hides it.
            rc = mod.main()
            if rc:
                print("pre-flight found problems - see the FAIL lines above")
        else:
            mod.main()
    print(f"\nall done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
