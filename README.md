# 🌀 Data Vortex — Social Engine Recovery
### Aaruush '26 · Round 1 · Phase 1

**Theme:** Rebuilding the Social Engine  
**Event:** Data Vortex @ Aaruush '26, SRM Institute of Science & Technology  
**Author:** Panshul Arora

---

## 📁 Repository Structure

| File | Description |
|------|-------------|
| `cleaning_pipeline.py` | Documented Python cleaning script |
| `EDA_Report_DataVortex_Phase1.pdf` | Full EDA report with charts and insights |
| `Social_Engine_Posts_Cleaned.csv` | Cleaned posts dataset (12,000 rows) |
| `Social_Engine_Users_Cleaned.csv` | Standardised users dataset (1,500 rows) |

---

## 🧩 Problem Statement

The Social Engine suffered a critical failure corrupting its data intake pipeline.  
The raw `Social_Engine_Posts_Corrupted.csv` contained 7 categories of corruption that needed to be identified and fixed before analysis could proceed.

---

## 🔍 Corruptions Found & Fixed

| # | Corruption | Count | Fix |
|---|-----------|-------|-----|
| 1 | Duplicate `post_id`s | 360 | `drop_duplicates(keep='first')` |
| 2 | Mixed timestamp formats (Unix / DD-MM-YYYY / ISO) | ~7,410 | Normalized to ISO 8601 |
| 3 | Missing `platform` | 1,846 | Mode imputation |
| 4 | Missing `likes` | 1,858 | Median imputation (2,498) |
| 5 | Negative `likes` | 525 | `abs()` — sign-flip correction |
| 6 | HTML entities in text (`&amp;`, `&lt;`, etc.) | 341 | `html.unescape()` |
| 7 | Missing `text_content` | 1,746 | `'[CONTENT UNAVAILABLE]'` |

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/panshularora/eda
cd eda

# Install dependencies
pip install pandas numpy matplotlib seaborn reportlab

# Place raw files in the same directory, then run:
python cleaning_pipeline.py
```

**Input files required:**
- `Social_Engine_Posts_Corrupted.csv`
- `Social_Engine_Users.csv`

**Output files produced:**
- `Social_Engine_Posts_Cleaned.csv`
- `Social_Engine_Users_Cleaned.csv`

---

## 📊 Key EDA Insights

1. **Midnight Anomaly** — Peak posting at 00:00 across all platforms → likely scheduled/bot activity
2. **YouTube Engagement Premium** — Highest avg likes despite not being top platform by volume
3. **Symmetric Likes Distribution** — Mean ≈ Median (2,493 vs 2,498) → synthetic data signature
4. **Asia-Pacific Language Dominance** — Chinese (zh) is top user language
5. **May 2024 Volume Spike** — Highest monthly post count, possible launch event
6. **Perfect Referential Integrity** — All 1,500 user_ids in Posts map to Users with zero orphans

---

## ⚙️ Assumptions

- Negative likes are sign-flip errors (not intentional downvotes)
- Missing platform is Missing At Random (MAR) — mode imputation defensible
- Unix timestamps assumed UTC
- DD-MM-YYYY inferred when day > 12 (unambiguous)
- No data fabricated — all values statistically derived

---

## 🛠 Tech Stack

`Python 3` · `pandas` · `numpy` · `matplotlib` · `seaborn` · `reportlab`
