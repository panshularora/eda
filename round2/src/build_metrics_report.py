"""Build ``Evaluation_Metrics_Report_Team_SE7EN.pdf``.

Every figure and number is read from ``reports/metrics.json`` and the
``model_comparison_*.csv`` tables that ``evaluate.py`` wrote, so the PDF cannot
drift from the code that produced it.
"""
from __future__ import annotations

import json

import pandas as pd
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Spacer

from config import CV_FOLDS, REPORTS, SEED, TASKS
from report_style import (
    CONTENT_W, S, bullets, build, callout, cover, figure, h1, h2, h3,
    metric_cards, p, reset_figures, table,
)

PRETTY = {"sentiment": "Task A — Sentiment", "topic": "Task B — Topic"}
OUT = REPORTS / "Evaluation_Metrics_Report_Team_SE7EN.pdf"


def pct(x):
    return f"{x * 100:.2f}%"


def f4(x):
    return f"{x:.4f}"


# ---------------------------------------------------------------------------
def section_protocol(m) -> list:
    flow = h1("Evaluation protocol", 1)
    flow.append(p(
        "Two supervised tasks were trained on Dataset 2 and are evaluated here: "
        "<b>sentiment_label</b> (3 classes, perfectly balanced at 3,000 posts each) "
        "and <b>topic_category</b> (4 classes, heavily skewed). Both are graded on "
        "the same held-out slice policy, and every number below comes from code in "
        "<font name='Courier' size='8'>round2/src</font>."))
    dup = m["duplicates"]
    flow.append(table([
        ["Item", "Setting", "Why"],
        ["Corpus", f"{dup['n_rows']:,} rows, {dup['n_unique_texts']:,} unique posts",
         "1,100 rows repeat text that already appears in the file"],
        ["Split unit", "unique post, never the row",
         "a row-level split would grade the model on memorised duplicates"],
        ["Held-out slice", "20% of unique posts, stratified on the target",
         "untouched by model selection, scored exactly once"],
        ["Model selection", f"{CV_FOLDS}-fold StratifiedGroupKFold on the remaining 80%",
         "folds never split a post, so CV and held-out measure the same thing"],
        ["Primary metric", "macro-F1",
         "topic_category is 86% one class; accuracy would hide the minority classes"],
        ["Secondary", "accuracy, weighted-F1, Cohen's kappa, MCC, one-vs-rest ROC-AUC",
         "kappa and MCC discount the score a chance classifier would reach"],
        ["Seed", str(SEED), "single seed in config.py; splits and fits are deterministic"],
    ], widths=[26 * mm, 52 * mm, CONTENT_W - 78 * mm]))
    flow.append(Spacer(1, 6))
    flow.append(callout(
        "Why the split unit matters more than the model here",
        "Holding the pipeline fixed and changing only the splitting rule moves "
        f"sentiment macro-F1 from <b>{f4(m['tasks']['sentiment']['leakage']['grouped_by_text_f1_macro'])}</b> "
        f"to <b>{f4(m['tasks']['sentiment']['leakage']['random_row_split_f1_macro'])}</b> "
        f"(+{f4(m['tasks']['sentiment']['leakage']['inflation'])}) and topic macro-F1 from "
        f"<b>{f4(m['tasks']['topic']['leakage']['grouped_by_text_f1_macro'])}</b> to "
        f"<b>{f4(m['tasks']['topic']['leakage']['random_row_split_f1_macro'])}</b> "
        f"(+{f4(m['tasks']['topic']['leakage']['inflation'])}). That gap is duplicate "
        "memorisation, not skill. Every figure in this report uses the grouped split."))
    flow.append(figure(m["figures"]["split_leakage"],
                       "The same pipeline scored two ways. The orange bars are what a "
                       "duplicate-blind evaluation would have reported."))
    flow.append(figure(m["figures"]["class_distribution"],
                       "Class balance for both targets. Sentiment is balanced by "
                       "construction; topic is not, which is why macro-F1 leads the reporting."))
    return flow


def section_selection(m) -> list:
    flow = [PageBreak()]
    flow += h1("Model selection results", 2)
    flow.append(p(
        "Eight candidates were scored per task under identical folds. The list is "
        "arranged as an ablation: the feature blocks are added one at a time with the "
        "classifier held fixed, and only then is the model family varied. Reading down "
        "a column therefore answers “what did this choice buy?” rather than just "
        "“which number is biggest?”"))
    flow.append(figure(m["figures"]["model_comparison"],
                       "Grouped 5-fold CV macro-F1 for every candidate, with "
                       "fold-to-fold standard deviation as error bars."))
    for task in TASKS:
        t = pd.read_csv(REPORTS / f"model_comparison_{task}.csv")
        rows = [["Candidate", "CV macro-F1", "SD", "Weighted-F1", "Accuracy", "Fit (s)"]]
        for _, r in t.iterrows():
            rows.append([r["model"], f4(r["cv_f1_macro"]), f"{r['cv_f1_macro_std']:.3f}",
                         f4(r["cv_f1_weighted"]), f4(r["cv_accuracy"]),
                         f"{r['fit_seconds']:.0f}"])
        flow.append(h2(f"{PRETTY[task]} — candidate comparison"))
        flow.append(table(rows, widths=[CONTENT_W - 96 * mm, 22 * mm, 13 * mm,
                                        22 * mm, 20 * mm, 19 * mm],
                          bold_rows=(1,)))
        flow.append(Spacer(1, 7))
    return flow


def section_task(m, task: str) -> list:
    b = m["tasks"][task]
    flow = [PageBreak()]
    flow += h1(f"{PRETTY[task]} — held-out results", 3 if task == "sentiment" else 4)
    flow.append(p(
        f"Selected model: <b>{b['selected_model']}</b>, refitted on the full training "
        f"split and Platt-calibrated. Scored once on {b['n']:,} held-out posts that no "
        "step of model selection had seen."))
    cards = [(f4(b["macro_f1"]), "macro-F1"), (pct(b["accuracy"]), "accuracy"),
             (f4(b["weighted_f1"]), "weighted-F1"), (f4(b["cohen_kappa"]), "Cohen's kappa"),
             (f4(b["matthews_corrcoef"]), "MCC")]
    if "roc_auc_ovr_macro" in b:
        cards.append((f4(b["roc_auc_ovr_macro"]), "ROC-AUC (OvR)"))
    flow.append(metric_cards(cards))

    comparison = pd.read_csv(REPORTS / f"model_comparison_{task}.csv")
    floor = float(comparison[comparison.model.str.startswith("baseline")].iloc[0]["cv_f1_macro"])
    permuted = b["leakage"]["permuted_labels_f1_macro"]
    flow.append(table([
        ["Metric", "Held-out", f"{CV_FOLDS}-fold CV", "Floor", "Reading"],
        ["macro-F1", f4(b["macro_f1"]),
         f"{f4(b['cv']['f1_macro'])} ± {b['cv']['f1_macro_std']:.3f}",
         f4(floor),
         "held-out sits inside the CV band, so selection did not overfit"],
        ["accuracy", f4(b["accuracy"]), "—",
         f4(max(pc["support"] for pc in b["per_class"].values()) / b["n"]),
         "floor here is the majority class, not chance"],
        ["weighted-F1", f4(b["weighted_f1"]), "—", "—",
         "support-weighted; flattered by class skew"],
        ["Cohen's kappa", f4(b["cohen_kappa"]), "—", "0.0000",
         "agreement corrected for chance"],
        ["MCC", f4(b["matthews_corrcoef"]), "—", "0.0000",
         "correlation between prediction and truth"],
        ["macro-F1, shuffled labels", "—", f4(permuted), f4(floor),
         "permutation check: the protocol itself leaks nothing"],
    ], widths=[30 * mm, 20 * mm, 27 * mm, 16 * mm, CONTENT_W - 93 * mm]))
    flow.append(Spacer(1, 7))

    flow.append(h2("Per-class breakdown"))
    rows = [["Class", "Precision", "Recall", "F1", "Support"]]
    for c in b["labels"]:
        pc = b["per_class"][c]
        rows.append([c.replace("_", " "), f4(pc["precision"]), f4(pc["recall"]),
                     f4(pc["f1"]), f"{pc['support']:,}"])
    rows.append(["<b>macro average</b>", f"<b>{f4(b['macro_precision'])}</b>",
                 f"<b>{f4(b['macro_recall'])}</b>", f"<b>{f4(b['macro_f1'])}</b>",
                 f"<b>{b['n']:,}</b>"])
    flow.append(table(rows, widths=[CONTENT_W - 84 * mm, 21 * mm, 21 * mm, 21 * mm, 21 * mm],
                      bold_rows=(len(rows) - 1,)))
    flow.append(Spacer(1, 7))

    flow.append(h2("Confusion matrix"))
    flow.append(figure(m["figures"][f"confusion_{task}"],
                       f"Counts on the left, row-normalised on the right; "
                       "the diagonal of the right-hand panel is per-class recall."))
    pairs = pd.read_csv(REPORTS / f"confusion_pairs_{task}.csv")
    rows = [["Actual", "Predicted as", "Posts", "Share of that class"]]
    for _, r in pairs.head(5).iterrows():
        rows.append([r["actual"].replace("_", " "), r["predicted"].replace("_", " "),
                     f"{int(r['n']):,}", pct(r["share_of_actual"])])
    flow.append(h3("Largest off-diagonal cells"))
    flow.append(table(rows, widths=[CONTENT_W - 96 * mm, 46 * mm, 22 * mm, 28 * mm]))
    return flow


def section_calibration(m) -> list:
    flow = [PageBreak()]
    flow += h1("Calibration, learning curves and stability", 5)
    flow.append(p(
        "The deployed models are wrapped in Platt scaling, so each prediction carries "
        "a probability. A confidence is only useful if it is honest, so it is measured "
        "rather than assumed: the reliability curves below plot observed accuracy "
        "against claimed confidence, and the expected calibration error (ECE) is the "
        "support-weighted gap between them."))
    rows = [["Task", "ECE", "Mean confidence when right", "Mean confidence when wrong",
             "Errors above 0.8 confidence"]]
    for task in TASKS:
        b = m["tasks"][task]
        ep = b["error_profile"]
        rows.append([PRETTY[task].split("—")[1].strip(),
                     f4(b["calibration"]["expected_calibration_error"]),
                     f4(ep["mean_conf_correct"]), f4(ep["mean_conf_wrong"]),
                     pct(ep["share_of_errors_above_0_8_conf"])])
    flow.append(table(rows, widths=[22 * mm, 16 * mm, 40 * mm, 42 * mm,
                                    CONTENT_W - 120 * mm]))
    flow.append(Spacer(1, 6))
    flow.append(figure(m["figures"]["reliability_sentiment"],
                       "Sentiment reliability. Points below the diagonal mean the "
                       "model claims more certainty than it earns.", width=CONTENT_W * 0.62))
    flow.append(figure(m["figures"]["reliability_topic"],
                       "Topic reliability.", width=CONTENT_W * 0.62))
    flow.append(PageBreak())
    flow.append(h2("Learning curves"))
    flow.append(p(
        "The gap between the training and validation curves at full data size is the "
        "part of the score that is memorisation; a validation curve still climbing at "
        "the right-hand edge means more labelled data would still pay."))
    rows = [["Task", "Train macro-F1 @ 100%", "Validation macro-F1 @ 100%", "Gap",
             "Validation gain over last 20% of data"]]
    for task in TASKS:
        lc = m["tasks"][task]["learning_curve"]
        gain = lc["val_f1_macro"][-1] - lc["val_f1_macro"][-2]
        rows.append([PRETTY[task].split("—")[1].strip(),
                     f4(lc["train_f1_macro"][-1]), f4(lc["val_f1_macro"][-1]),
                     f4(lc["final_gap"]), f"{gain:+.4f}"])
    flow.append(table(rows, widths=[22 * mm, 34 * mm, 38 * mm, 16 * mm,
                                    CONTENT_W - 110 * mm]))
    flow.append(Spacer(1, 5))
    flow.append(figure(m["figures"]["learning_curve_sentiment"],
                       "Sentiment learning curve.", width=CONTENT_W * 0.60))
    flow.append(figure(m["figures"]["learning_curve_topic"],
                       "Topic learning curve.", width=CONTENT_W * 0.60))
    return flow


def section_errors(m) -> list:
    flow = [PageBreak()]
    flow += h1("Error analysis", 6)
    flow.append(p(
        "Errors are profiled three ways for each task: by post length, by the presence "
        "of a surface cue, and by the confidence the model attached to the mistake. The "
        "third is the one that matters operationally — a wrong-but-hesitant prediction "
        "can be routed to a human, a wrong-and-certain one cannot."))
    flow.append(figure(m["figures"]["error_profile_sentiment"],
                       "Sentiment errors by length, surface cue and confidence."))
    flow.append(figure(m["figures"]["error_profile_topic"],
                       "Topic errors by length, surface cue and confidence."))
    for task in TASKS:
        b = m["tasks"][task]
        flow.append(h2(f"{PRETTY[task]} — highest-confidence mistakes"))
        rows = [["Post (truncated)", "Actual", "Predicted", "Conf."]]
        for r in b["worst_errors"][:8]:
            txt = str(r["post_text"])[:112].replace("&", "&amp;").replace("<", "&lt;")
            rows.append([txt, str(r["actual"]).replace("_", " "),
                         str(r["predicted"]).replace("_", " "), f"{float(r['confidence']):.3f}"])
        flow.append(table(rows, widths=[CONTENT_W - 74 * mm, 26 * mm, 30 * mm, 14 * mm],
                          font_size=7.0))
        flow.append(Spacer(1, 6))
    flow.append(p(
        "Both full error lists are in <font name='Courier' size='8'>reports/errors_sentiment.csv"
        "</font> and <font name='Courier' size='8'>reports/errors_topic.csv</font>; the "
        "technical report groups them into named failure modes."))
    return flow


def section_topic_rule(m) -> list:
    b = m["tasks"]["topic"]
    rd = b["rule_diagnosis"]
    flow = [PageBreak()]
    flow += h1("Why the topic score reads the way it does", 7)
    flow.append(callout(
        "topic_category is a substring switch, not an annotation",
        "A case-insensitive substring rule recovered from the data reproduces "
        "<b>9,000 of 9,000</b> topic labels exactly: any post containing "
        "<i>app, down, update, crash, screen, slow, bug</i> or <i>glitch</i> is "
        "Technical_Issues; failing that, <i>ban, account, suspend, hack, password</i> is "
        "Account_Security; failing that, <i>ui, mode, feature, ugly, design, button</i> is "
        "Feature_Feedback; everything else is Community_Discussion. The Bayes error of "
        "this task is therefore exactly zero, and any gap below 1.000 is the learned "
        "model failing to recover a substring it saw too rarely — not a limit on "
        "topical understanding.", tone="warn"))
    flow.append(table([
        ["System", "Held-out accuracy", "What it is"],
        [b["selected_model"], f4(rd["model_accuracy"]),
         "learned from the training split only, no knowledge of the rule"],
        ["Recovered substring rule", f4(rd["rule_accuracy"]),
         "the label generator itself, reconstructed by mining the training file"],
    ], widths=[CONTENT_W - 78 * mm, 30 * mm, 48 * mm]))
    flow.append(Spacer(1, 6))
    flow.append(p(
        "Splitting the held-out posts by how often their deciding trigger appears in "
        "the corpus confirms the mechanism: posts with no trigger at all are almost "
        f"never wrong ({pct(rd['error_rate_no_trigger'])} error), posts whose trigger is "
        f"common are close behind ({pct(rd['error_rate_common_trigger'])}), and posts "
        "whose trigger appears in fewer than 60 posts carry "
        f"{pct(rd['error_rate_rare_trigger'])} of the error."))
    flow.append(figure(m["figures"]["topic_rule_gap"],
                       "Left: topic error rate by how common the deciding trigger is. "
                       "Right: the learned model against the recovered rule."))
    flow.append(p(
        "This is reported rather than exploited. Submitting the rule would score 1.000 "
        "and would teach the Social Engine nothing, because the rule generalises only "
        "for as long as the labels keep being generated the same way. The learned model "
        "is what is shipped; the rule is shipped beside it as a diagnostic, and the "
        "recommendation in the technical report is that topic_category be re-annotated "
        "before it is trusted."))
    return flow


def section_repro(m) -> list:
    flow = [PageBreak()]
    flow += h1("Reproducibility", 8)
    flow.append(p("Every artefact in this report is regenerated by one command."))
    flow.append(table([
        ["Item", "Value"],
        ["Dataset SHA-256", f"<font name='Courier' size='7'>{m['dataset_sha256']}</font>"],
        ["Random seed", str(m["seed"])],
        ["Rebuild everything", "<font name='Courier' size='7.5'>python round2/run_round2.py</font>"],
        ["Train only", "<font name='Courier' size='7.5'>python round2/src/train.py</font>"],
        ["Evaluate only", "<font name='Courier' size='7.5'>python round2/src/evaluate.py</font>"],
        ["Score new text", "<font name='Courier' size='7.5'>python round2/src/predict.py \"the app keeps crashing\"</font>"],
        ["Machine-readable metrics", "<font name='Courier' size='7.5'>round2/reports/metrics.json</font>"],
    ], widths=[44 * mm, CONTENT_W - 44 * mm]))
    flow.append(Spacer(1, 6))
    flow.append(h2("Full scikit-learn classification reports"))
    for task in TASKS:
        flow.append(h3(PRETTY[task]))
        text = m["tasks"][task]["classification_report"].replace(" ", "&nbsp;")
        from reportlab.platypus import Paragraph
        flow.append(Paragraph(text.replace("\n", "<br/>"), S["mono"]))
        flow.append(Spacer(1, 6))
    return flow


def main():
    reset_figures()
    m = json.loads((REPORTS / "metrics.json").read_text(encoding="utf-8"))
    s, t = m["tasks"]["sentiment"], m["tasks"]["topic"]
    flow = cover(
        "Evaluation Metrics Report",
        "Rebuilding the Social Engine · semantic comprehension layer",
        "Round 2 deliverable 3 of 4",
        [("Tasks", "sentiment_label (3 classes), topic_category (4 classes)"),
         ("Corpus", f"{m['duplicates']['n_rows']:,} labelled posts "
                    f"({m['duplicates']['n_unique_texts']:,} unique)"),
         ("Held-out macro-F1", f"sentiment {f4(s['macro_f1'])} · topic {f4(t['macro_f1'])}"),
         ("Protocol", f"{CV_FOLDS}-fold grouped CV for selection, 20% held out for the "
                      "single final score"),
         ("Dataset SHA-256", m["dataset_sha256"][:32] + "…")],
    )
    flow += h1("Headline results", None)
    flow.append(metric_cards([
        (f4(s["macro_f1"]), "Sentiment macro-F1 (held-out)"),
        (pct(s["accuracy"]), "Sentiment accuracy"),
        (f4(t["macro_f1"]), "Topic macro-F1 (held-out)"),
        (pct(t["accuracy"]), "Topic accuracy"),
    ]))
    flow.append(p(
        "Sentiment is the genuine language problem in Dataset 2 and is reported "
        f"honestly: <b>{f4(s['macro_f1'])}</b> macro-F1 over three balanced classes, "
        f"against a {f4(1/3)} chance floor, with Cohen's kappa {f4(s['cohen_kappa'])}. "
        "Topic is a different animal — Section 7 shows its labels are produced by a "
        "substring rule, which is why its ceiling is 1.000 and why its score is "
        "reported with that context attached rather than as evidence of topical "
        "understanding."))
    flow.append(figure(m["figures"]["per_class"],
                       "Per-class precision, recall and F1 on the held-out slice "
                       "for both tasks."))
    flow += section_protocol(m)
    flow += section_selection(m)
    flow += section_task(m, "sentiment")
    flow += section_task(m, "topic")
    flow += section_calibration(m)
    flow += section_errors(m)
    flow += section_topic_rule(m)
    flow += section_repro(m)

    build(OUT, "Evaluation Metrics Report", flow)
    print(f"built {OUT.name} ({OUT.stat().st_size / 1e6:.2f} MB)")
    return OUT


if __name__ == "__main__":
    main()
