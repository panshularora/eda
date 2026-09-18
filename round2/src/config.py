"""Shared paths, constants and the one random seed used everywhere.

Every script in round2/src imports from here, so a single edit changes the
whole pipeline and re-running it reproduces byte-identical artefacts.
"""
from __future__ import annotations

from pathlib import Path

ROUND2 = Path(__file__).resolve().parents[1]
DATA = ROUND2 / "data"
REPORTS = ROUND2 / "reports"
FIGURES = REPORTS / "figures"
MODELS = ROUND2 / "models"
SUBMISSION = ROUND2 / "SUBMISSION"

RAW_CSV = DATA / "Labeled_Social_NLP_Training_Data.csv"
MODEL_BUNDLE = MODELS / "social_engine_nlp_team_se7en.pkl"

SEED = 42
TEST_SIZE = 0.20          # held-out slice, split by unique text (never by row)
CV_FOLDS = 5

TEXT_COL = "post_text"
ID_COL = "text_id"
TASKS = {
    "sentiment": "sentiment_label",
    "topic": "topic_category",
}
SENTIMENT_CLASSES = ["Negative", "Neutral", "Positive"]
TOPIC_CLASSES = [
    "Account_Security",
    "Community_Discussion",
    "Feature_Feedback",
    "Technical_Issues",
]

TEAM = "Team SE7EN"
MEMBERS = "Tanmay Singh \u00b7 Panshul Arora"
EVENT = "Data Vortex A'26 \u00b7 Round 2 \u00b7 Rebuilding the Social Engine"

for _d in (REPORTS, FIGURES, MODELS, SUBMISSION):
    _d.mkdir(parents=True, exist_ok=True)
