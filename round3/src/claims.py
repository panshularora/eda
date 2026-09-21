"""Every number the prose is allowed to use, written by the pipeline.

Why
---
A README and a report are where numbers go stale. The pipeline is re-run, a
filter is tightened, a dedupe rule is fixed - and the prose still quotes the
figure from two runs ago, because nothing connects them. Round 2 handled this
by reading every reported number out of `metrics.json`; this is the same
discipline for Round 3, made checkable.

`build()` collects every claim the write-up is entitled to make into a single
`reports/CLAIMS.json`, keyed by a short name. `render()` then substitutes
`{{claim_name}}` in a Markdown template, and raises on any placeholder it
cannot fill. A number that no longer exists breaks the build instead of
quietly surviving into the submission.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from config import PROCESSED, REPORTS, ROUND3


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def build() -> dict:
    analysis = _load(REPORTS / "analysis.json")
    transfer = _load(REPORTS / "round2_transfer.json")
    quality = _load(REPORTS / "data_quality.json")
    collection = _load(PROCESSED / "collection_audit.json")
    normal = _load(PROCESSED / "normalise_audit.json")
    census = _load(ROUND3 / "data" / "raw" / "play_census_audit.json")
    cases = _load(REPORTS / "case_studies.json")

    c: dict = {}

    def put(key, value, fmt=None):
        if value is None:
            return
        if fmt == ",":
            c[key] = f"{value:,}"
        elif fmt == "%":
            c[key] = f"{value:.1%}"
        elif fmt == "2f":
            c[key] = f"{value:.2f}"
        elif fmt == "3f":
            c[key] = f"{value:.3f}"
        elif fmt == "4f":
            c[key] = f"{value:.4f}"
        else:
            c[key] = value

    # --- collection --------------------------------------------------------
    put("reviews_enumerated", census.get("reviews_enumerated"), ",")
    put("reviews_kept", census.get("reviews_kept"), ",")
    put("apps_total", census.get("apps"))
    put("apps_complete_window", census.get("apps_with_complete_window"))
    put("quota_per_brand_day", census.get("per_brand_day_quota"))
    if census.get("by_app"):
        put("play_pages", sum(a.get("pages", 0) for a in census["by_app"]), ",")
    if census.get("reviews_enumerated"):
        put("sampling_fraction",
            census["reviews_kept"] / census["reviews_enumerated"], "%")

    put("raw_rows", normal.get("input_rows"), ",")
    put("normalised_rows", normal.get("output_rows"), ",")
    put("dropped_duplicate_text", normal.get("dropped_duplicate_text"), ",")
    put("window_span_days", normal.get("span_days"))
    if normal.get("window_start"):
        c["window_start"] = str(normal["window_start"])[:10]
        c["window_end"] = str(normal["window_end"])[:10]
    if normal.get("by_source"):
        c["by_source"] = normal["by_source"]

    http = (collection.get("http") or {})
    # cross-pass HTTP totals are deliberately not claimed; see collection_audit.json

    # --- corpus ------------------------------------------------------------
    put("n_records", analysis.get("n_records"), ",")
    put("n_delay", analysis.get("n_delay_related"), ",")
    if analysis.get("n_records"):
        put("delay_share", analysis["n_delay_related"] / analysis["n_records"], "%")

    # --- the composition control ------------------------------------------
    diag = (analysis.get("coverage") or {}).get("panel_diagnostics") or {}
    if diag:
        put("brands_total", diag.get("brands_total"))
        put("brands_in_panel", diag.get("brands_in_panel"))
        put("r_all_brands", diag["all_brands"]["pearson_r_with_day_index"], "3f")
        put("r_panel", diag["balanced_panel"]["pearson_r_with_day_index"], "3f")
        put("cv_all_brands", diag["all_brands"]["cv"], "2f")
        put("cv_panel", diag["balanced_panel"]["cv"], "2f")
        put("panel_mean_daily", diag["balanced_panel"]["mean_daily_n"], "2f")
        put("all_mean_daily", diag["all_brands"]["mean_daily_n"], "2f")

    pe = analysis.get("population_estimate") or {}
    put("population_enumerated", pe.get("reviews_enumerated"), ",")

    # --- shifts and spikes -------------------------------------------------
    shifts = analysis.get("sentiment_shifts") or []
    put("n_shifts", len(shifts))
    put("n_shifts_survive_panel",
        sum(1 for s in shifts
            if (s.get("balanced_panel_check") or {}).get("survives") is True))
    if shifts:
        top = shifts[0]
        c["top_shift_date"] = top.get("date")
        c["top_shift_scope"] = top.get("scope")
        put("top_shift_d_records", top.get("cohens_d_records"), "3f")
        put("top_shift_d_daily", top.get("cohens_d_daily"), "2f")
    scan = analysis.get("shift_scan") or {}
    put("scan_tests", scan.get("candidates_tested"), ",")
    put("scan_survivors", scan.get("significant_after_fdr"))

    spikes = analysis.get("engagement_spikes") or []
    put("n_spikes", len(spikes))
    put("n_spikes_volume", sum(1 for s in spikes if s.get("kind") == "volume"))
    put("n_spikes_endorsement",
        sum(1 for s in spikes if s.get("kind") == "endorsement"))
    end = [s for s in spikes if s.get("kind") == "endorsement"]
    if end:
        t = end[0]
        c["top_spike_date"] = t.get("date")
        put("top_spike_ratio", t.get("ratio_to_median"), "2f")
        conc = t.get("concentration") or {}
        if conc.get("top1_share") is not None:
            put("top_spike_top1_share", conc["top1_share"], "%")
            put("top_spike_top5_share", conc["top5_share"], "%")

    # --- the case-study classifier -----------------------------------------
    if cases:
        put("n_case_events", cases.get("n_events"))
        for sig, n in (cases.get("by_signature") or {}).items():
            put(f"n_case_{sig}", n)
        profs = cases.get("profiles") or []
        if profs:
            c["case_labels"] = [p["label"] for p in profs]
            for i, pr in enumerate(profs[:4], 1):
                c[f"case{i}_label"] = pr["label"]
                put(f"case{i}_rating_before", pr["rating_before"], "2f")
                put(f"case{i}_rating_during", pr["rating_during"], "2f")
                put(f"case{i}_d", pr["cohens_d_records"], "2f")
                put(f"case{i}_delay_before", pr["delay_share_before"], "%")
                put(f"case{i}_delay_during", pr["delay_share_during"], "%")
                put(f"case{i}_outage_before", pr["outage_language_before"], "%")
                put(f"case{i}_outage_during", pr["outage_language_during"], "%")

    # --- the Round 2 model -------------------------------------------------
    put("transfer_n", transfer.get("n"), ",")
    put("transfer_accuracy", transfer.get("accuracy"), "4f")
    put("transfer_macro_f1", transfer.get("macro_f1"), "4f")
    put("transfer_kappa", transfer.get("cohen_kappa"), "4f")
    po = transfer.get("polarity_only") or {}
    put("polarity_n", po.get("n"), ",")
    put("polarity_accuracy", po.get("accuracy"), "4f")
    put("polarity_kappa", po.get("cohen_kappa"), "3f")
    g7 = transfer.get("accuracy_at_conf_ge_0.7") or {}
    put("gate70_accuracy", g7.get("accuracy"), "3f")
    put("gate70_coverage", g7.get("coverage"), "%")

    # per-class transfer, so the README never hand-types a recall again
    cm = transfer.get("confusion_matrix")
    labels = transfer.get("labels") or []
    if cm and "Neutral" in labels:
        i = labels.index("Neutral")
        tp = cm[i][i]
        support = sum(cm[i])
        predicted = sum(r[i] for r in cm)
        put("neutral_recall", tp / support if support else None, "2f")
        put("neutral_precision", tp / predicted if predicted else None, "2f")
        put("neutral_true", support, ",")
        put("neutral_predicted", predicted, ",")
        if support:
            put("neutral_over_emission", predicted / support, "2f")

    rc = transfer.get("recalibration") or {}
    if rc.get("available"):
        put("recal_tau", rc.get("tau_fitted"), "2f")
        put("recal_n_test", rc.get("n_test"), ",")
        put("argmax_accuracy", rc["argmax"]["accuracy"], "4f")
        put("argmax_kappa", rc["argmax"]["cohen_kappa"], "4f")
        put("argmax_neutral_predicted", rc["argmax"]["predicted_neutral"], ",")
        put("recal_accuracy", rc["recalibrated"]["accuracy"], "4f")
        put("recal_kappa", rc["recalibrated"]["cohen_kappa"], "4f")
        put("recal_neutral_predicted", rc["recalibrated"]["predicted_neutral"], ",")
        put("recal_true_neutral", rc.get("true_neutral_in_test"), ",")
        put("recal_accuracy_gain",
            rc["recalibrated"]["accuracy"] - rc["argmax"]["accuracy"], "4f")
        put("recal_kappa_gain",
            rc["recalibrated"]["cohen_kappa"] - rc["argmax"]["cohen_kappa"], "4f")

    rt = analysis.get("round2_topic_replay") or {}
    if rt.get("available"):
        put("topic_rule_agreement",
            rt.get("agreement_with_recovered_substring_rule"), "%")

    # --- the fields the first pass never opened ---------------------------
    rb = analysis.get("company_reply_behaviour") or {}
    if rb.get("available"):
        put("reply_rate", rb.get("overall_reply_rate"), "%")
    sp = analysis.get("severity_profile") or {}
    if sp.get("available"):
        put("duration_share", sp.get("share_of_delay_rows"), "%")
        put("duration_median_hours", sp.get("median_hours"), "2f")

    # --- data quality ------------------------------------------------------
    sub = (quality.get("submission_csv") or {})
    put("submission_rows", sub.get("rows"), ",")
    put("submission_mb", sub.get("size_mb"), "2f")
    put("n_columns", quality.get("n_columns"))

    (REPORTS / "CLAIMS.json").write_text(
        json.dumps(c, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return c


_PLACEHOLDER = re.compile(r"\{\{([a-zA-Z0-9_]+)\}\}")


def render(template: str, claims: dict | None = None) -> str:
    """Substitute {{claim}} placeholders, and refuse to ship an unfilled one."""
    claims = claims or _load(REPORTS / "CLAIMS.json")
    missing: list[str] = []

    def sub(m):
        key = m.group(1)
        if key not in claims:
            missing.append(key)
            return m.group(0)
        return str(claims[key])

    out = _PLACEHOLDER.sub(sub, template)
    if missing:
        raise SystemExit("claims missing from reports/CLAIMS.json: "
                         + ", ".join(sorted(set(missing))))
    return out


def main() -> dict:
    c = build()
    print(f"  {len(c)} verified claims -> reports/CLAIMS.json")
    return c


if __name__ == "__main__":
    main()
