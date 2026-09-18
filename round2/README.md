# Rebuilding the Social Engine: Data Vortex · Round 2

**Team SE7EN**: Tanmay Singh · Panshul Arora
Data Vortex A'26 @ Aaruush, SRM Institute of Science & Technology

Round 1 restored the intake pipeline and the analytical core. Round 2 rebuilds the
**semantic comprehension layer**: reading 9,000 short social posts and returning both
how the author feels and what the post is about.

---

## Submission

| # | Form slot | File |
|---|---|---|
| 1 | NLP Model Script/Notebook | [`SUBMISSION/Round2_NLP_Model_Notebook_Team_SE7EN.ipynb`](SUBMISSION/Round2_NLP_Model_Notebook_Team_SE7EN.ipynb) |
| 2 | Trained Model Files | [`SUBMISSION/Round2_Trained_Models_Team_SE7EN.pkl`](SUBMISSION/Round2_Trained_Models_Team_SE7EN.pkl) |
| 3 | Evaluation Metrics Report | [`SUBMISSION/Round2_Evaluation_Metrics_Report_Team_SE7EN.pdf`](SUBMISSION/Round2_Evaluation_Metrics_Report_Team_SE7EN.pdf) |
| 4 | Round 2 Technical Report | [`SUBMISSION/Round2_Technical_Report_Team_SE7EN.pdf`](SUBMISSION/Round2_Technical_Report_Team_SE7EN.pdf) |

---

## Results

| | Sentiment (3 classes) | Topic (4 classes) |
|---|---|---|
| Selected model | word + char 3-5gram + surface → LinearSVC | char 2-3gram → LinearSVC |
| Grouped 5-fold CV macro-F1 | see `reports/model_comparison_sentiment.csv` | see `reports/model_comparison_topic.csv` |
| Held-out macro-F1 | see `reports/metrics.json` | see `reports/metrics.json` |

Every number in both PDFs is read from `reports/metrics.json`, which is written by
`src/evaluate.py`. Nothing in the reports is typed by hand.

---

## The finding that shaped the submission

`topic_category` is **not an annotation**. A case-insensitive substring switch,
recovered from the data by greedy minimum-cover mining, reproduces **9,000 of 9,000**
topic labels exactly:

```
if any of [app, down, update, crash, screen, slow, bug, glitch] in text.lower():
    Technical_Issues
elif any of [ban, account, suspend, hack, password]:
    Account_Security
elif any of [ui, mode, feature, ugly, design, button]:
    Feature_Feedback
else:
    Community_Discussion
```

Reproduce it with `python round2/src/audit_labels.py`.

Because it matches **substrings** rather than words, roughly one post in eight gets a
topic from a trigger buried in an unrelated word:

| Post | Trigger | Assigned topic |
|---|---|---|
| *That Janet Jackson… Happy Friday friends!* | `app` inside **happy** | Technical_Issues |
| *…guitarist in the British heavy metal band JUDAS PRIEST* | `ban` inside **band** | Account_Security |
| *Trina was my role model lmfao* | `mode` inside **model** | Feature_Feedback |
| *Jurassic Park is screening at the Actors Playhouse* | `screen` inside **screening** | Technical_Issues |

Three consequences:

1. The **Bayes error of the topic task is exactly zero** — a perfect score is
   attainable and meaningless.
2. A topic model is rewarded for **recovering substrings**, not topics.
3. The column is not safe to route a support queue on until it is re-annotated.

We report this rather than submitting the rule for a free 1.000. The learned model is
what ships; the rule ships beside it as a diagnostic.

The model-selection tables show the same thing from the other direction: on topic, the
character block *alone* wins and adding the word block **costs** ~10 macro-F1 points,
and the winning character n-gram range is **2–3** — the length of `ui`, `ban`, `app`,
`bug`. The hyperparameter that wins is describing what the label is made of.

Sentiment shows no such structure. Cue words are graded ("love" is ~81% Positive, not
100%), which is what human annotation looks like — so that is where the modelling
effort went.

---

## Evaluation protocol

1,100 of the 9,000 rows repeat text that appears elsewhere (7,900 unique posts), and
every repeat is label-consistent. A random **row** split therefore grades the model on
memorised duplicates, so everything here splits on the **unique post**:

- 20% of unique posts held out, stratified, scored exactly once at the end.
- Model selection by 5-fold `StratifiedGroupKFold` on the remaining 80%, with
  vectorisers fitted inside each fold.
- The shortcut is measured rather than merely avoided: `src/train.py` scores the same
  pipeline both ways and records the inflation.
- A label-permutation check confirms the protocol itself leaks nothing.

---

## Reproduce

```bash
pip install -r round2/requirements.txt
python round2/run_round2.py
```

That runs: label audit → train and select → held-out evaluation and figures → both
PDFs → executed notebook → submission folder. Add `--sweep` to re-run the
hyperparameter search.

Score new text:

```bash
python round2/src/predict.py "the app keeps crashing after the update"
```

---

## Repository map

| Path | Contents |
|---|---|
| `data/` | the provided corpus, unmodified (SHA-256 recorded in every report) |
| `src/config.py` | paths, seed, task definitions — the single place to change anything |
| `src/preprocess.py` | de-serialisation repair, platform-noise normalisation, negation scope |
| `src/features.py` | the three feature blocks (word, char, surface) |
| `src/dataio.py` | loading, duplicate report, group-aware splitting |
| `src/audit_labels.py` | recovery of the topic labelling rule |
| `src/sweep.py` | hyperparameter search → `reports/hparam_sweep.md` |
| `src/train.py` | candidate comparison, leakage + permutation audits, calibration |
| `src/evaluate.py` | held-out metrics, figures, error tables → `reports/metrics.json` |
| `src/build_*.py` | the two PDFs and the executed notebook |
| `src/predict.py` | inference on new text |
| `notebooks/` | executed end-to-end walkthrough |
| `models/` | both task models in one bundle |
| `reports/` | metrics, CSV tables, 15 figures, both PDFs |
| `SUBMISSION/` | the four files to upload, nothing else |
