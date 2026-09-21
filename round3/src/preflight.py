"""Check the submission the way a sceptical judge would, before uploading it.

Every failure this catches is one that actually happened in the first version
of this work:

* a form slot with no file behind it (the analytical report);
* a notebook whose cells were not all executed, so a judge reads code with no
  output under it;
* a notebook that prints a number contradicting the paragraph above it
  (`significant_after_fdr` was assigned the wrong count);
* a README citing `reports/relevance_audit.md`, which did not exist;
* prose quoting figures from two runs ago, because nothing connected the two;
* an effect size reported as "large" that was computed on fourteen daily
  averages rather than on people;
* a dataset column containing a raw author handle.

So each of those is an assertion here. The script exits non-zero if any fails,
which makes "is this ready to upload" a command rather than a judgement.

    python round3/src/preflight.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROCESSED, REPORTS, ROUND3, SUBMISSION      # noqa: E402

MAX_UPLOAD_MB = 10.0
SLOTS = {
    "1. dataset": "Round3_Delay_Reactions_Dataset_Team_SE7EN.csv",
    "2. collection script": "Round3_Collection_Script_Team_SE7EN.py",
    "3. analysis notebook": "Round3_Analysis_Notebook_Team_SE7EN.ipynb",
    "4. analytical report": "Round3_Analytical_Report_Team_SE7EN.pdf",
}

checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    checks.append((name, bool(ok), detail))
    return bool(ok)


def _load(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> int:
    # ---- form slots ------------------------------------------------------
    for slot, fname in SLOTS.items():
        p = SUBMISSION / fname
        if p.exists():
            mb = p.stat().st_size / 1e6
            check(f"slot present: {slot}", True, f"{fname}  {mb:.2f} MB")
            check(f"slot within upload cap: {slot}", mb <= MAX_UPLOAD_MB,
                  f"{mb:.2f} MB of {MAX_UPLOAD_MB} MB")
        else:
            check(f"slot present: {slot}", False, f"missing {fname}")

    # ---- the collection script must compile ------------------------------
    script = SUBMISSION / SLOTS["2. collection script"]
    if script.exists():
        try:
            compile(script.read_text(encoding="utf-8"), str(script), "exec")
            check("collection script compiles", True)
        except SyntaxError as e:
            check("collection script compiles", False, str(e))

    # ---- the notebook must be fully executed -----------------------------
    nb_path = SUBMISSION / SLOTS["3. analysis notebook"]
    if nb_path.exists():
        nb = _load(nb_path)
        code = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
        unrun = [i for i, c in enumerate(code) if c.get("execution_count") is None]
        check("every notebook code cell executed", not unrun,
              f"{len(code) - len(unrun)}/{len(code)} executed"
              + (f"; unrun at {unrun[:6]}" if unrun else ""))
        errs = [o for c in code for o in c.get("outputs", [])
                if o.get("output_type") == "error"]
        check("no notebook cell raised", not errs,
              f"{len(errs)} error outputs" if errs else "")
        empty = [i for i, c in enumerate(code)
                 if not c.get("outputs") and "display(" not in "".join(c.get("source", []))
                 and "import" not in "".join(c.get("source", []))[:40]]
        check("notebook cells produce output", len(empty) <= 2,
              f"{len(empty)} silent cells")

    # ---- README has no unfilled placeholders -----------------------------
    readme = ROUND3 / "README.md"
    if readme.exists():
        text = readme.read_text(encoding="utf-8")
        left = re.findall(r"\{\{[a-zA-Z0-9_]+\}\}", text)
        check("README has no unfilled placeholders", not left,
              ", ".join(sorted(set(left))[:6]))
        # every file the README links to must exist
        links = re.findall(r"\]\((?!http)([^)]+)\)", text)
        broken = [l for l in links if not (ROUND3 / l).exists()]
        check("README links resolve", not broken, ", ".join(broken[:6]))

    # ---- every file the docs cite must exist -----------------------------
    cited = ["reports/relevance_audit.md", "reports/CLAIMS.json",
             "reports/analysis.json", "reports/case_studies.json",
             "src/audit_relevance.py", "src/waves.py", "src/panel.py"]
    for rel in cited:
        check(f"cited file exists: {rel}", (ROUND3 / rel).exists())

    # ---- internal consistency of the numbers -----------------------------
    analysis = _load(REPORTS / "analysis.json")
    scan = analysis.get("shift_scan") or {}
    shifts = analysis.get("sentiment_shifts") or []
    fdr = analysis.get("sentiment_shifts_scan_fdr") or []
    check("FDR survivor count matches the FDR list",
          scan.get("significant_after_fdr") == len(fdr),
          f"reported {scan.get('significant_after_fdr')}, list has {len(fdr)}")
    check("FDR count is not the PELT count",
          not (shifts and scan.get("significant_after_fdr") == len(shifts)
               and len(fdr) != len(shifts)),
          "the two were conflated in the first version")

    # ---- effect sizes are reported honestly ------------------------------
    if shifts:
        has_both = all("cohens_d_records" in s and "cohens_d_daily" in s
                       for s in shifts)
        check("every shift reports both effect sizes", has_both)
        bare = [s for s in shifts if "cohens_d" in s]
        check("no bare 'cohens_d' left to be misread", not bare,
              f"{len(bare)} shifts still carry an unqualified cohens_d")

    # ---- spikes are typed and, if endorsement, characterised -------------
    spikes = analysis.get("engagement_spikes") or []
    if spikes:
        check("every spike declares its kind",
              all("kind" in s for s in spikes))
        end = [s for s in spikes if s.get("kind") == "endorsement"]
        check("endorsement spikes carry concentration",
              all("concentration" in s for s in end) if end else True,
              f"{len(end)} endorsement spikes")

    # ---- the rulebook minimum --------------------------------------------
    n_sent_shifts = len(shifts) + len(fdr)
    check("rulebook: at least 2 sentiment shifts", n_sent_shifts >= 2,
          f"{len(shifts)} change points + {len(fdr)} FDR survivors")
    check("rulebook: at least 1 engagement spike", len(spikes) >= 1,
          f"{len(spikes)} spikes")
    check("rulebook: key entities reported",
          bool(analysis.get("entities")) and bool(analysis.get("entity_analysis")))
    check("rulebook: reasons behind the changes reported",
          bool(analysis.get("explained_events")))

    # ---- the composition control ran -------------------------------------
    diag = (analysis.get("coverage") or {}).get("panel_diagnostics") or {}
    check("composition control present", bool(diag),
          f"{diag.get('brands_in_panel')} of {diag.get('brands_total')} brands in panel")
    if diag:
        r = abs(diag["balanced_panel"]["pearson_r_with_day_index"])
        check("balanced-panel volume is not a pagination curve", r < 0.6,
              f"r = {diag['balanced_panel']['pearson_r_with_day_index']:+.3f} "
              f"vs all-brand {diag['all_brands']['pearson_r_with_day_index']:+.3f}")

    # ---- no raw identifiers in the published dataset ----------------------
    ds = SUBMISSION / SLOTS["1. dataset"]
    if ds.exists():
        head = ds.read_text(encoding="utf-8", errors="replace").split("\n", 1)[0]
        cols = [c.strip().strip('"') for c in head.split(",")]
        forbidden = [c for c in cols
                     if c in {"userName", "author", "username", "handle",
                              "email", "user_name", "author_name"}]
        check("no raw identifier column in the dataset", not forbidden,
              ", ".join(forbidden))
        check("dataset carries the pseudonym instead",
              "author_pseudonym" in cols)
        check("dataset carries the label evidence spans",
              "delay_type_evidence" in cols and "reaction_type_evidence" in cols)

    # ---- the relevance filter's regression tests --------------------------
    try:
        from config import DELAY_RELEVANCE                     # noqa: PLC0415
        rx = re.compile(DELAY_RELEVANCE, re.I)
        cases = {"late night food is not good": False,
                 "incurred late-payment or overdraft fees": False,
                 "can't wait to see how the results will be": False,
                 "chocolate and translate and download": False,
                 "my parcel is 3 days late": True,
                 "still waiting for my refund": True}
        bad = [t for t, want in cases.items() if bool(rx.search(t)) != want]
        check("relevance filter regression tests pass", not bad,
              "; ".join(bad))
    except Exception as e:
        check("relevance filter regression tests pass", False, str(e))

    # ---- report ----------------------------------------------------------
    width = max(len(n) for n, _, _ in checks)
    failed = 0
    print()
    for name, ok, detail in checks:
        mark = "PASS" if ok else "FAIL"
        failed += not ok
        print(f"  [{mark}] {name:{width}s}  {detail}")
    print()
    print(f"{len(checks) - failed}/{len(checks)} checks passed")
    if failed:
        print(f"\n{failed} FAILED - this is not ready to upload")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
