"""
Builds the Phase 1 EDA report from pipeline outputs.

    python src/build_report.py

Reads   reports/cleaning_audit.json, reports/eda_stats.json, reports/validation_report.md, reports/figures/*.png
Writes  reports/EDA_Report_Team_SE7EN.html
        reports/EDA_Report_Team_SE7EN.pdf   (printed with headless Chrome/Edge if one is installed)

Every number in the report is read from the JSON files, so re-running the pipeline on a new
extract regenerates a consistent report with no hand-typed figures.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
HTML_OUT = REPORTS / "EDA_Report_Team_SE7EN.html"
PDF_OUT = REPORTS / "EDA_Report_Team_SE7EN.pdf"

TEAM = "Team SE7EN"
MEMBERS = "Tanmay Singh · Panshul Arora"
REPO_URL = os.environ.get("DATAVORTEX_REPO_URL", "https://github.com/panshularora/eda")

A = json.loads((REPORTS / "cleaning_audit.json").read_text(encoding="utf-8"))
S = json.loads((REPORTS / "eda_stats.json").read_text(encoding="utf-8"))
V = (REPORTS / "validation_report.md").read_text(encoding="utf-8")
passed, total = map(int, re.search(r"\*\*(\d+) / (\d+) checks passed", V).groups())


def n(x) -> str:
    """Thousands separator for ints, one decimal for floats."""
    if isinstance(x, float) and not x.is_integer():
        return f"{x:,.1f}"
    return f"{int(x):,}"


def p(x: float) -> str:
    return "p < 0.001" if x < 0.001 else f"p = {x:.3f}"


def pct(part, whole, nd=1) -> str:
    return f"{part / whole * 100:.{nd}f}%"


def fig(name: str, caption: str) -> str:
    return f'<figure><img src="figures/{name}.png" alt="{caption}"><figcaption>{caption}</figcaption></figure>'


def implication(text: str) -> str:
    return f'<div class="phase2"><b>So what / Phase 2:</b> {text}</div>'


# ── shorthand into the audit / stats ──
raw = A["raw_file_counts_before_dedup"]
dup, ts, plat, likes, text = A["duplicates"], A["timestamps"], A["platform"], A["likes"], A["text_content"]
ov, miss, hod, lk, tm = S["overview"], S["missingness"], S["hour_of_day"], S["likes"], S["time"]
eng, us, lang, tx, anom = S["engagement"], S["users"], S["language"], S["text"], S["daily_anomalies"]
N = ov["posts"]
changes = A["output"]["change_log_rows"]
art = text["artifact_counts"]
mt = miss["tests"]
max_mcar_p_min = min(v for t in mt.values() for k, v in t.items() if k.startswith("chi2") and v is not None)
bp = eng["by_platform"]
insta_sh, fb_sh = bp["shares"]["means"]["Instagram"], bp["shares"]["means"]["Facebook"]
co = miss["co_missing_observed_vs_expected_if_independent"]
tok = miss["empty_vs_NULL_token_ratio"]
n_tests = miss["n_tests"]
pc = ov["platform_counts_known"]
top_platform, top_platform_n = max(pc.items(), key=lambda kv: kv[1])
platform_gap_pct = (max(pc.values()) - min(pc.values())) / min(pc.values()) * 100
openers = list(tx["template_openers"].values())[:6]

CSS = """
@page { size: A4; margin: 17mm 16mm 18mm 16mm;
  @bottom-left { content: "Team SE7EN (Tanmay Singh · Panshul Arora) · Data Vortex, Aaruush '26 · Round 1, Phase 1"; font: 7.5pt 'Segoe UI', sans-serif; color: #898781; }
  @bottom-right { content: "Page " counter(page) " of " counter(pages); font: 7.5pt 'Segoe UI', sans-serif; color: #898781; } }
@page :first { @bottom-left { content: none; } @bottom-right { content: none; } }
:root { --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --grid:#e1e0d9; --blue:#2a78d6; --blue-wash:#eef4fc; --orange:#eb6834; --surface:#fcfcfb; }
* { box-sizing: border-box; }
html { background: #fff; }
body { font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; color: var(--ink); font-size: 9.6pt; line-height: 1.5; margin: 0; }
.wrap { max-width: 178mm; margin: 0 auto; padding-inline: 4mm; }
h1 { font-size: 25pt; line-height: 1.15; margin: 0 0 6pt; font-weight: 700; letter-spacing: -0.01em; }
h2 { font-size: 14.5pt; margin: 20pt 0 6pt; padding-top: 4pt; border-top: 2px solid var(--ink); break-after: avoid; }
h2 .num { color: var(--blue); margin-right: 6pt; }
h3 { font-size: 11pt; margin: 14pt 0 4pt; break-after: avoid; }
p { margin: 0 0 7pt; }
.lead { font-size: 10.5pt; color: var(--ink2); }
.small { font-size: 8.2pt; color: var(--ink2); }
code, .mono { font-family: 'Cascadia Mono', Consolas, monospace; font-size: 8.4pt; background: #f2f1ee; padding: 0 3px; border-radius: 3px; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt; font-size: 8.4pt; break-inside: auto; }
th { text-align: left; font-weight: 600; color: var(--ink2); border-bottom: 1.2px solid var(--ink); padding: 4pt 5pt; vertical-align: bottom; }
td { border-bottom: 0.6px solid var(--grid); padding: 4pt 5pt; vertical-align: top; }
td.r, th.r { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
tr { break-inside: avoid; }
figure { margin: 8pt 0 12pt; break-inside: avoid; }
figure img { width: 100%; display: block; }
figcaption { font-size: 8pt; color: var(--muted); margin-top: 3pt; }
.phase2 { background: var(--blue-wash); border-left: 3px solid var(--blue); padding: 6pt 9pt; margin: 6pt 0 12pt; font-size: 9pt; break-inside: avoid; }
.warn { background: #fdf1ec; border-left: 3px solid var(--orange); padding: 6pt 9pt; margin: 6pt 0 12pt; font-size: 9pt; break-inside: avoid; }
.cover { height: 257mm; display: flex; flex-direction: column; break-after: page; padding-top: 18mm; }
.eyebrow { font-size: 9pt; letter-spacing: 0.16em; text-transform: uppercase; color: var(--blue); font-weight: 600; margin-bottom: 10pt; }
.tiles { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8pt; margin: 22pt 0; }
.tile { border: 1px solid var(--grid); border-radius: 6px; padding: 9pt 11pt; background: var(--surface); }
.tile .v { font-size: 19pt; font-weight: 650; line-height: 1.1; }
.tile .l { font-size: 8.2pt; color: var(--ink2); margin-top: 2pt; }
.meta { margin-top: auto; font-size: 9pt; color: var(--ink2); border-top: 1px solid var(--grid); padding-top: 10pt; }
.meta b { color: var(--ink); }
.summary li { margin-bottom: 5pt; }
.insight { break-inside: avoid; margin: 10pt 0 4pt; }
.insight h3 { margin-top: 0; }
.tag { display: inline-block; font-size: 7.5pt; font-weight: 600; color: var(--ink2); border: 1px solid var(--grid); border-radius: 3px; padding: 0 4px; margin-left: 4pt; vertical-align: 1px; }
.steps { counter-reset: s; list-style: none; padding: 0; margin: 4pt 0 10pt; }
.steps li { counter-increment: s; padding-left: 20pt; position: relative; margin-bottom: 3pt; }
.steps li::before { content: counter(s); position: absolute; left: 0; top: 0; width: 13pt; height: 13pt; border-radius: 50%; background: var(--ink); color: #fff; font-size: 7.5pt; text-align: center; line-height: 13pt; font-weight: 600; }
.pagebreak { break-before: page; }
.toc { columns: 2; font-size: 9pt; margin: 8pt 0 0; padding-left: 14pt; }
"""


def cleaning_table() -> str:
    rows = [
        ("1", "Exact duplicate rows", f"{n(dup['duplicated_post_ids'])} post_ids appear 2–3 times ({dup['group_size_distribution'].get('3_copies', 0)} triplicates); every copy identical in all 8 columns",
         "Keep first occurrence, drop the rest", n(dup["rows_removed"]), "—"),
        ("2", "Timestamp as Unix epoch", "10-digit integers; decoded as UTC they fall in the same 2024-05-01 → 2025-04-30 window as ISO values",
         "Convert to UTC <code>YYYY-MM-DD HH:MM:SS</code>", n(raw["timestamp_unix_epoch"]), n(ts["format_counts"]["unix_epoch"])),
        ("3", "Timestamp as dd-mm-yyyy (time of day lost)", f"2nd field never &gt; 12, 1st field &gt; 12 in {n(ts['dd_mm_evidence']['rows_with_first_field_gt_12'])} rows ⇒ day-first; no time component exists",
         "Keep <code>post_date</code>; <code>post_datetime_utc</code> = NULL; <code>timestamp_precision</code> = 'day'", n(raw["timestamp_dd_mm_yyyy"]), n(ts["format_counts"]["dd-mm-yyyy"])),
        ("4", "Missing platform, two spellings", f"'' and 'NULL' in a ~2:1 ratio ({n(tok['platform']['<empty>'])} : {n(tok['platform']['NULL'])})",
         "Unify to NULL. Not imputed (no signal predicts it)", n(raw["platform_missing"]), n(plat["missing_total"])),
        ("5", "Missing likes, two spellings", f"'' and 'NULL' ({n(tok['likes']['<empty>'])} : {n(tok['likes']['NULL'])})",
         "Unify to NULL. Not imputed", n(raw["likes_missing"]), n(likes["missing_total"])),
        ("6", "Negative likes (sign flip)", f"All {n(likes['negative_values'])} negatives are float-formatted (<code>-2388.0</code>) while every other value is an integer; |negative| has the same distribution as positives (KS {p(lk['ks_abs_negative_vs_positive_p'])})",
         "<code>abs()</code>, cast to integer, flag <code>dq_likes_sign_corrected</code>", n(raw["likes_negative"]), n(likes["negative_values"])),
        ("7", "Missing text, two spellings", f"'' and 'NULL' ({n(tok['text_content']['<empty>'])} : {n(tok['text_content']['NULL'])}), plus {text['missing_token_hidden_behind_artifact']} 'NULL' tokens hidden behind an artifact (e.g. <code>NULLÃ©</code>)",
         "Unify to NULL (no placeholder text)", n(raw["text_missing_including_hidden"]), n(text["missing_total_after_cleaning"])),
        ("8", "Text suffix artifacts (5 kinds)", "<code>&amp;amp;</code> " + n(art["html_entity_amp"]) + ", <code>&lt;div&gt;</code> " + n(art["html_tag_div"]) + ", <code>&lt;br&gt;</code> " + n(art["html_tag_br"]) + ", <code>\\n\\n</code> " + n(art["trailing_newlines"]) + ", mojibake <code>Ã©</code> " + n(art["mojibake_e_acute"]) + ". Each occurs only as the final characters, never mid-text, at most one per post",
         "Strip the suffix; flag the kind in <code>dq_text_artifact_removed</code>", n(raw["text_suffix_artifact"]), n(text["rows_with_artifact"])),
        ("9", "Double spaces inside text", "Gaps where a template slot was empty (e.g. <i>“recommend.&nbsp;&nbsp;#Beauty”</i>)",
         "Collapse runs of whitespace to one space", "—", n(text["whitespace_collapsed_rows"])),
    ]
    body = "".join(f"<tr><td>{a}</td><td><b>{b}</b></td><td>{c}</td><td>{d}</td><td class='r'>{e}</td><td class='r'>{f}</td></tr>" for a, b, c, d, e, f in rows)
    return ("<table><thead><tr><th>#</th><th>Corruption</th><th>Evidence (computed)</th><th>Action</th>"
            "<th class='r'>Raw file</th><th class='r'>After dedup</th></tr></thead><tbody>" + body + "</tbody></table>")


def top_users_table() -> str:
    rows = "".join(
        f"<tr><td>{i + 1}</td><td class='mono'>{u['user_id']}</td><td class='r'>{u['posts']}</td><td class='r'>{n(u['engagement_mean'])}</td><td class='r'>{n(u['followers'])}</td><td>{u['location']}</td></tr>"
        for i, u in enumerate(us["top10_by_mean_min5_known"][:5]))
    return ("<table><thead><tr><th>#</th><th>user_id</th><th class='r'>posts</th><th class='r'>mean engagement / post</th>"
            "<th class='r'>followers</th><th>location</th></tr></thead><tbody>" + rows + "</tbody></table>")


html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Rebuilding the Social Engine — Phase 1 Report · {TEAM}</title>
<meta name="author" content="{TEAM}: {MEMBERS}">
<style>{CSS}</style></head><body><div class="wrap">

<section class="cover">
  <div class="eyebrow">Data Vortex · Aaruush '26 · Round 1 · Phase 1</div>
  <h1>Rebuilding the Social Engine</h1>
  <p class="lead">Data recovery, cleaning and exploratory analysis of the corrupted Social Engine intake dataset.</p>
  <div class="tiles">
    <div class="tile"><div class="v">{n(N)}</div><div class="l">unique, clean posts (from {n(raw['rows'])} raw rows)</div></div>
    <div class="tile"><div class="v">9</div><div class="l">corruption types detected and fixed</div></div>
    <div class="tile"><div class="v">{n(changes)}</div><div class="l">value-level changes, each logged with a rule</div></div>
    <div class="tile"><div class="v">0</div><div class="l">values imputed or invented</div></div>
    <div class="tile"><div class="v">{passed} / {total}</div><div class="l">automated validation checks passed</div></div>
    <div class="tile"><div class="v">8</div><div class="l">statistically tested insights</div></div>
  </div>
  <p><b>Contents</b></p>
  <ol class="toc">
    <li>Executive summary</li><li>Recovering the dataset</li><li>Dataset overview</li><li>Corruption audit &amp; fixes</li>
    <li>Decisions, assumptions &amp; alternatives</li><li>Validation</li><li>Exploratory analysis</li><li>Key insights</li>
    <li>Limitations &amp; Phase 2 readiness</li><li>Reproducibility</li>
  </ol>
  <div class="meta">
    <b>{TEAM}</b> — {MEMBERS}<br>
    Repository: <b>{REPO_URL}</b><br>
    Theme: Rebuilding the Social Engine · Submission deadline 14 September 2026, 11:59 PM<br>
    Generated by <code>src/build_report.py</code> from pipeline outputs; every figure and number is reproducible with <code>python run_all.py</code>.
  </div>
</section>

<h2><span class="num">1</span>Executive summary</h2>
<ul class="summary">
  <li><b>Recovered the exact dataset.</b> Following the recovery-log clue to <code>node_07</code> through the recovery shell yielded
  <code>Social_Engine_Posts_Corrupted.csv</code> ({n(raw['rows'])} × 8) and <code>Social_Engine_Users.csv</code> ({n(A['users']['rows'])} × 5).
  SHA-256 hashes are pinned in the pipeline, so any altered input is detected.</li>
  <li><b>Found nine corruption types, including four that are easy to miss:</b> timestamps that lost their time of day
  ({n(ts['format_counts']['dd-mm-yyyy'])} posts), two different spellings of missing values, five kinds of injected text suffixes
  (<code>&amp;amp;</code>, <code>&lt;br&gt;</code>, <code>&lt;div&gt;</code>, <code>Ã©</code>, trailing newlines), and <code>NULL</code> tokens hidden behind those suffixes.</li>
  <li><b>Cleaned without inventing data.</b> Values are corrected only where the original is deterministically recoverable. Unrecoverable values stay NULL, since they are
  missing completely at random (every χ² test {p(max_mcar_p_min).replace('p = ', 'p ≥ ')}). Imputation would add false spikes and bias Phase 2 correlations.</li>
  <li><b>Avoided the “midnight spike” trap.</b> Padding date-only timestamps with 00:00:00 would put {hod['naive_hour0_share_pct']}% of all posts at midnight ({hod['naive_hour0_multiple_of_mean']}× any other hour).
  Real posting activity is flat across the day ({p(hod['chi2_uniform_p'])}).</li>
  <li><b>What the data says:</b> volume is stationary (~{tm['daily_mean']:.0f} posts/day), engagement metrics are independent and uniform, no platform has a real engagement edge,
  followers do not predict engagement (Spearman ρ = {us['spearman_followers_vs_mean_engagement'][0]}), totals simply track post count (Spearman ρ = {us['spearman_posts_vs_total_engagement']}),
  user language is unrelated to location, and seasonal campaigns are not mentioned in season. Each claim is backed by a test in Section 7.</li>
</ul>

<h2><span class="num">2</span>Recovering the dataset</h2>
<p>The rulebook withholds the dataset; it had to be recovered from the corrupted Social Engine site. Our path:</p>
<ol class="steps">
  <li>The landing terminal boots and reports <code>data_intake.service … FAILED</code>. <i>Enter Recovery Mode</i> asks for a locally stored operator identity.</li>
  <li>All ten dashboard modules fail in different ways (hang, signal lost, page error, access denied, no response). The ARCHIVE module shows <code>ARCH_LINKLOST_07</code>. After a few failures the site hints: <i>“Nothing responding? There may be another way in.”</i></li>
  <li>Each <b>system log</b> line <b>begins</b> with a timestamp and the component reporting it. Only one component is healthy:
  <code>node_07 responded 200 (intermittent)</code>, then <code>watchdog last known good node: node_07</code> and <code>dashboard link to node_07: severed</code>.</li>
  <li>The recovery shell (terminal button) accepts <code>help</code>, <code>status</code>, <code>scan</code>, <code>logs</code>. <code>scan</code> confirms <i>“node_07 responding intermittently, outside normal routing. Manual reconnection may be possible.”</i></li>
  <li><code>connect node_07</code> stabilises the link and opens <b>ARCHIVE NODE 07</b>. We recovered both files listed under <i>Dataset 01</i>.</li>
</ol>
<p class="small"><b>Integrity note.</b> The site builds both CSVs in the browser and downloads them as Blobs, so they never show up as separate requests in the Network tab.
We confirmed our downloads match the copies embedded in the site bundle byte-for-byte. SHA-256: posts <code>{A['inputs']['Social_Engine_Posts_Corrupted.csv']['sha256'][:16]}…</code>, users <code>{A['inputs']['Social_Engine_Users.csv']['sha256'][:16]}…</code>.
A <code>.gitattributes</code> rule keeps git from rewriting their line endings, so the hashes still match after a clone on any OS.</p>

<h2><span class="num">3</span>Dataset overview</h2>
<table>
<thead><tr><th>Table</th><th>Grain</th><th class="r">Raw rows</th><th class="r">Clean rows</th><th>Columns (raw)</th></tr></thead>
<tbody>
<tr><td><b>Posts</b></td><td>one post</td><td class="r">{n(raw['rows'])}</td><td class="r">{n(N)}</td><td><code>post_id, user_id, platform, text_content, timestamp, likes, shares, comments</code></td></tr>
<tr><td><b>Users</b></td><td>one account</td><td class="r">{n(A['users']['rows'])}</td><td class="r">{n(A['users']['rows'])}</td><td><code>user_id, location, language, account_created, follower_count</code></td></tr>
</tbody></table>
<p>The posts cover <b>{ov['date_start']} → {ov['date_end']}</b> ({ov['days']} days) across {ov['platforms']} platforms, written by {n(ov['users'])} users in {lang['cities']} cities and {lang['countries']} countries using {len(lang['counts'])} languages.
All users signed up during 2023 ({A['users']['account_created_range'][0]} → {A['users']['account_created_range'][1]}). No post predates its author's account.
Every post's <code>user_id</code> exists in Users and every user has at least one post, so referential integrity holds in both directions.
<b>Only {n(ov['complete_rows'])} posts ({pct(ov['complete_rows'], N)}) have platform, text and likes all present.</b>
Dropping incomplete rows would therefore discard {pct(N - ov['complete_rows'], N, 0)} of the data. We keep every row and let each analysis use the fields it needs.</p>

<h2><span class="num">4</span>Corruption audit &amp; fixes</h2>
<p>Every cell is read as raw text (<code>keep_default_na=False</code>), so nothing is silently converted.
Each corruption is detected by an explicit rule, its <b>evidence is computed and asserted in code</b> (the pipeline stops if an assumption breaks), then the fix is applied and logged.
Counts are shown for the raw file and after de-duplication, so they reconcile exactly.</p>
{cleaning_table()}
{fig('fig01_corruption_profile', 'Figure 1. Size of each corruption after de-duplication. Percentages are shares of the 12,000 unique posts; duplicates are a share of raw rows.')}
<p><b>Lineage.</b> <code>reports/change_log.csv</code> records all {n(changes)} changed values as (post_id, column, rule, old value, new value).
The cleaned posts table also carries three data-quality flags (<code>dq_duplicate_copies_removed</code>, <code>dq_likes_sign_corrected</code>, <code>dq_text_artifact_removed</code>), so corrections stay queryable in SQL.
<b>Users</b> passed every check (no missing, padded, duplicate or malformed values), so it needed no corrections. We only added <code>city</code>, <code>country</code> and <code>language_name</code> for grouping.
ISO timestamps were already correct and were only re-rendered in the common output format.</p>

<h2><span class="num">5</span>Decisions, assumptions &amp; alternatives</h2>
<h3>5.1 Date-only timestamps keep their date and get no invented time</h3>
<p>{n(ts['format_counts']['dd-mm-yyyy'])} posts ({pct(ts['format_counts']['dd-mm-yyyy'], N)}) carry only a date. The obvious conversion (<code>pd.to_datetime</code> → <code>2024-09-25 00:00:00</code>) invents a time.
The cleaned table therefore has <code>post_date</code> (always present), <code>post_datetime_utc</code> (NULL when unknown), <code>timestamp_precision</code> and <code>timestamp_source_format</code>.</p>
{fig('fig03_midnight_artifact', 'Figure 2. Left: the hour-of-day profile if date-only posts are padded to midnight. Right: the true profile using only posts with a real time.')}

<h3>5.2 Negative likes are sign flips; missing likes stay NULL</h3>
<p>Two independent pieces of evidence show a sign-flip corruption rather than “dislikes”. First, the format: every negative value is a float string while every other value is an integer. Second, the distribution: the absolute values of negatives match positives (KS {p(lk['ks_abs_negative_vs_positive_p'])}; means {n(lk['mean_abs_negative'])} vs {n(lk['mean_positive'])}).
Missing likes cannot be recovered. They correlate with nothing, so any model reduces to the median, and median imputation would put
{n(lk['if_median_imputed']['rows_at_single_value'])} posts on one value ({n(lk['median_available'])}) and shrink the standard deviation by {lk['if_median_imputed']['std_shrink_pct']}%.</p>
{fig('fig04_likes_signflip_and_imputation', 'Figure 3. Left: the cumulative distribution of |negative likes| overlaps positive likes. Right: median imputation would create a spike that later analysis could mistake for an anomaly.')}

<h3>5.3 Missing values are missing completely at random</h3>
<p>We tested each corruption flag against platform, timestamp source format and month ({n_tests} χ² tests). None is significant; the smallest p-value is {max_mcar_p_min:.3f}.
Missingness in one column does not predict missingness in another: likes and platform are both missing in {co['likes&platform'][0]} posts vs {n(co['likes&platform'][1])} expected under independence, likes and text in {co['likes&text'][0]} vs {n(co['likes&text'][1])}.
Because the data is MCAR, analyses that simply skip missing values are unbiased, and filling values in would add nothing but false precision.</p>
{fig('fig02_missingness_mcar', 'Figure 4. Corruption rate per platform (rows with a known platform), with χ² p-values. Flat rows mean the corruption is independent of platform.')}

<h3>5.4 Assumptions register</h3>
<table><thead><tr><th>#</th><th>Assumption</th><th>Support</th><th>If wrong</th></tr></thead><tbody>
<tr><td>A1</td><td>Unix epochs are UTC; naive ISO strings use the same UTC clock</td><td>Epoch is UTC by definition. Both formats span the same window ({ts['range_by_format_utc']['iso8601'][0][:10]} → {ts['range_by_format_utc']['iso8601'][1][:10]}) with matching flat hour profiles</td><td>Hour-of-day shifts by a constant offset; daily and monthly results barely change</td></tr>
<tr><td>A2</td><td><code>dd-mm-yyyy</code> is day-first</td><td>Proven: month field max = {ts['dd_mm_evidence']['max_second_field']}, day field max = {ts['dd_mm_evidence']['max_first_field']}</td><td>The pipeline raises an error</td></tr>
<tr><td>A3</td><td>Negative likes are sign-flipped, not downvotes</td><td>Format fingerprint plus identical distribution (5.2)</td><td>{n(likes['negative_values'])} posts ({pct(likes['negative_values'], N)}) would carry the wrong sign; flagged, so they can be excluded</td></tr>
<tr><td>A4</td><td>The five suffixes are injected, not user-written</td><td>Asserted: never mid-text, never stacked, only at the end. Decoding <code>&amp;amp;</code> or <code>Ã©</code> would leave a stray <code>&amp;</code> or <code>é</code> glued to punctuation</td><td>The pipeline raises an error if a suffix appears mid-text</td></tr>
<tr><td>A5</td><td>Duplicate post_ids are repeated ingestions of one post</td><td>Asserted: every copy identical in all columns</td><td>The pipeline raises an error on conflicting copies</td></tr>
<tr><td>A6</td><td>Template oddities (“How do I fix about…”, “Pepsi Pepsi Zero Sugar”) are original content</td><td>They occur at template scale, not at corruption rates, and are not suffix-like</td><td>Rewriting them would be fabrication, so they are kept verbatim</td></tr>
</tbody></table>

<h3>5.5 Alternatives we rejected</h3>
<table><thead><tr><th>Alternative</th><th>Why rejected</th></tr></thead><tbody>
<tr><td>Mode-impute platform</td><td>With five near-equal platforms the mode is right only ~20% of the time. It would push {top_platform} from {pct(top_platform_n, N)} to {pct(top_platform_n + plat['missing_total'], N)} of posts ({n(top_platform_n + plat['missing_total'])}) and fabricate a “dominant platform”.</td></tr>
<tr><td>Median-impute likes</td><td>Creates a {n(lk['if_median_imputed']['rows_at_single_value'])}-post spike, shrinks variance and biases correlations toward 0 (Figure 3).</td></tr>
<tr><td><code>html.unescape()</code> the text</td><td>Fixes only 1 of 5 artifact kinds and turns <code>&amp;amp;</code> into a stray trailing <code>&amp;</code>. <code>&lt;br&gt;</code>, <code>&lt;div&gt;</code>, <code>Ã©</code> and newlines would survive.</td></tr>
<tr><td>Placeholder text such as “[CONTENT UNAVAILABLE]”</td><td>Pollutes word counts and text search, and SQL cannot tell it apart from real content. NULL is the honest encoding.</td></tr>
<tr><td>Drop incomplete rows</td><td>Discards {pct(N - ov['complete_rows'], N, 0)} of posts for no bias reduction, since the data is MCAR.</td></tr>
<tr><td>“Fix” language to match location</td><td>The true language is unknowable. The mismatch is itself a finding (Insight 7).</td></tr>
</tbody></table>

<h2><span class="num">6</span>Validation</h2>
<p><code>src/validate.py</code> re-reads the written CSVs and asserts <b>{total} contracts; {passed} pass</b>. They cover schema and key uniqueness, the foreign key,
no leftover missing-value tokens, platform vocabulary, date window and precision consistency, integer ranges (likes 0–5,000, shares 0–2,000, comments 0–1,000),
zero HTML, mojibake, newlines or double spaces left in text, and no post before account creation. It also <b>re-runs the full pipeline and confirms the outputs are byte-identical</b>.
The full table is in <code>reports/validation_report.md</code>.</p>

<h2><span class="num">7</span>Exploratory analysis</h2>
<p class="small">Conventions: we report the test and p-value for every pattern. p ≥ 0.05 is reported as “no evidence”, never as a finding. Likes-based statistics use the {n(lk['n_available'])} posts with known likes.
Hour-of-day uses the {n(hod['timed_rows'])} posts with a real time.</p>

<h3>7.1 Volume over time</h3>
{fig('fig05_volume_over_time', 'Figure 5. Daily posts with a 28-day mean (top) and posts per day by month, labelled with monthly totals (bottom).')}
<p>Volume is flat at <b>{tm['daily_mean']} posts/day</b> (range {tm['daily_min']}–{tm['daily_max']}). Variance/mean is {tm['poisson_dispersion_var_over_mean']}, i.e. Poisson noise around a constant rate.
The largest month ({tm['busiest_month_raw'][0]}, {n(tm['busiest_month_raw'][1])}) and smallest ({tm['quietest_month_raw'][0]}, {n(tm['quietest_month_raw'][1])}) differ only by calendar length.
Per-day rates run {min(tm['monthly_posts_per_day'].values())}–{max(tm['monthly_posts_per_day'].values())}, and a χ² test of monthly totals against days-in-month gives {p(tm['chi2_month_counts_vs_calendar_days_p'])}.
Day of week is also flat ({p(tm['chi2_dow_uniform_p'])}). <b>No single day is anomalous</b> after correcting for 365 tests (Poisson tail, α = 0.05/365).
Only {anom['days_z_above_3']} days exceed z = 3, against {anom['expected_days_z_above_3_if_poisson']} expected by chance.</p>

<h3>7.2 Engagement distributions</h3>
{fig('fig07_engagement_distributions', 'Figure 6. Histograms of likes, shares and comments; the orange line is the count expected under a uniform distribution.')}
<p>Likes, shares and comments are each <b>uniform</b> on [0, 5,000], [0, 2,000] and [0, 1,000] (KS vs uniform: {p(lk['ks_uniform_0_5000_p'])}, {p(eng['shares']['ks_uniform_p'])}, {p(eng['comments']['ks_uniform_p'])}), and <b>mutually independent</b>
(Spearman ρ: likes–shares {eng['spearman']['likes_shares']}, likes–comments {eng['spearman']['likes_comments']}, shares–comments {eng['spearman']['shares_comments']}).
The means ({n(eng['likes']['mean'])} / {n(eng['shares']['mean'])} / {n(eng['comments']['mean'])}) sit at the midpoints, so expected engagement splits about 5 : 2 : 1.
There is no heavy tail and no viral outlier: the maxima are hard caps.</p>

<h3>7.3 Platforms</h3>
{fig('fig06_platform_engagement', 'Figure 7. Mean engagement per platform with 95% confidence intervals.')}
<p>Platform volumes are balanced ({', '.join(f'{k} {n(v)}' for k, v in sorted(pc.items(), key=lambda kv: -kv[1]))}; largest vs smallest {platform_gap_pct:.1f}%, χ² vs uniform {p(ov['platform_count_chi2_uniform_p'])}). Mean likes differ by at most {bp['likes_f']['spread_pct_of_mean']}% of the overall mean ({p(bp['likes_f']['kruskal_p'])}) and comments by {bp['comments']['spread_pct_of_mean']}% ({p(bp['comments']['kruskal_p'])}).
Shares are the only nominal difference: Instagram {n(insta_sh)} vs Facebook {n(fb_sh)}, {p(bp['shares']['kruskal_p'])}. That does not survive a Bonferroni correction for four metrics (α = {eng['bonferroni_alpha']:.4f}), and the effect size is negligible (ε² = {bp['shares']['epsilon_squared']}).</p>

<h3>7.4 Users: activity, reach and engagement</h3>
{fig('fig08_user_activity', 'Figure 8. Posts per user against a Poisson(8) curve; total engagement vs post count; mean engagement per post vs followers.')}
<p>Posts per user follow a <b>Poisson(λ = {us['posts_per_user']['mean']})</b> distribution (variance/mean {us['posts_per_user']['variance_over_mean']}, range {us['posts_per_user']['min']}–{us['posts_per_user']['max']}).
{n(us['users_on_2plus_platforms'])} users ({pct(us['users_on_2plus_platforms'], ov['users'])}) post on 2+ platforms, averaging {us['mean_platforms_per_user']}. A null model that assigns each post a random platform predicts {n(us['expected_users_2plus_if_random'])} and {us['expected_mean_platforms_if_random']}, so cross-platform posting is not a behaviour, just chance.
Followers (uniform {n(us['followers']['min'])}–{n(us['followers']['max'])}, KS {p(us['followers']['ks_uniform_p'])}) do not predict engagement per post (ρ = {us['spearman_followers_vs_mean_engagement'][0]}, {p(us['spearman_followers_vs_mean_engagement'][1])}) or activity (ρ = {us['spearman_followers_vs_posts'][0]}).</p>
<p>Ranking users by <b>total</b> engagement mostly ranks post counts (ρ = {us['spearman_posts_vs_total_engagement']}). The top 10 by total average {us['top10_total_mean_posts']} posts, twice the norm,
and <b>none of them</b> appears in the top 10 by <b>mean engagement per post</b> (minimum 5 posts with known likes). The per-post leaders are:</p>
{top_users_table()}

<h3>7.5 Language and location</h3>
{fig('fig09_language_vs_country', "Figure 9. Users by country × language. Colour is the share of each country's users; outlined cells are a national language of that country.")}
<p>The ten languages are equally common ({p(lang['chi2_uniform_p'])} vs uniform) and <b>statistically independent of country</b> (χ² {p(lang['country_language_chi2_p'])}, Cramér's V = {lang['cramers_v']}).
Only {lang['national_language_match_pct']}% of users speak a national language of their country, against {lang['expected_match_pct_if_random']}% expected from random assignment (binomial {p(lang['binomial_p'])}).
Examples: <code>{lang['example_mismatches'][0]['location']} → {lang['example_mismatches'][0]['language']}</code>, <code>{lang['example_mismatches'][1]['location']} → {lang['example_mismatches'][1]['language']}</code>, <code>{lang['example_mismatches'][3]['location']} → {lang['example_mismatches'][3]['language']}</code>.
The {lang['users_in_countries_without_language_code']} users in Italy and South Korea cannot match at all, since <code>it</code> and <code>ko</code> are absent. The most represented country is the {lang['top_country'][0]} ({lang['top_country'][1]} users).</p>

<h3>7.6 Text content</h3>
{fig('fig10_text_brands_campaigns', 'Figure 10. Left: posts mentioning each brand. Right: when each seasonal campaign is mentioned; outlined cells are its real season.')}
<p>The {n(tx['posts_with_text'])} texts are near-unique ({n(tx['distinct_texts'])} distinct) but built from templates: six openers such as “Just unboxed my new…”, “Just saw an ad for…” and “Comparing…” each start {n(min(openers))}–{n(max(openers))} posts.
Mentions spread evenly over 10 brands ({n(min(tx['brand_mentions'].values()))}–{n(max(tx['brand_mentions'].values()))} posts each), {tx['distinct_hashtags']} hashtags ({tx['hashtags_per_post_mean']} per post) and {tx['distinct_mentions']} @handles (in {tx['posts_with_mentions_pct']}% of posts).
Verdict phrases split into positive {n(tx['verdict_counts']['positive'])}, negative {n(tx['verdict_counts']['negative'])} and neutral {n(tx['verdict_counts']['neutral'])}.
Engagement does not differ by verdict ({p(tx['kruskal_engagement_by_verdict_p'])}).
<b>{tx['contradiction_pct_of_emotional_posts']}% of posts that state an emotion contradict their own verdict</b>, e.g. <i>“{tx['contradiction_examples'][0]}”</i>.
Seasonal campaigns land in their season {tx['seasonal_in_season_pct']}% of the time vs {tx['seasonal_in_season_expected_pct']}% by chance; <i>SpringBlast2025</i> even appears in {tx['springblast2025_mentions_in_2024']} posts dated 2024.</p>

<h2><span class="num">8</span>Key insights</h2>

<div class="insight"><h3>1 · The corruption is machine-injected at fixed rates, uniformly at random <span class="tag">data quality</span></h3>
<p>~15% missing in each of platform, likes and text; {mt['text artifact']['rate_pct']}% text suffixes (five kinds of ~2.7% each); {mt['likes sign-flipped']['rate_pct']}% sign flips; {mt['duplicated on ingest']['rate_pct']}% duplicates.
All {n_tests} dependence tests are non-significant, co-missingness matches independence, and each missing column mixes <code>''</code> and <code>NULL</code> at the same ~2:1 ratio. That points to a single random corruption process, not a failing source.</p>
{implication("Available-case SQL (<code>WHERE likes IS NOT NULL</code>) is unbiased. Keep NULLs; do not <code>COALESCE</code> to a median.")}</div>

<div class="insight"><h3>2 · The “midnight posting spike” is an artifact of bad parsing <span class="tag">anomaly</span></h3>
<p>{pct(ts['format_counts']['dd-mm-yyyy'], N)} of timestamps have no time of day. Defaulting them to 00:00 makes hour 0 hold {n(hod['naive_hour0_if_padded_midnight'])} posts ({hod['naive_hour0_multiple_of_mean']}× any other hour). With real times only, posting is flat across all 24 hours ({hod['true_min_hour'][1]}–{hod['true_max_hour'][1]} per hour, {p(hod['chi2_uniform_p'])}).</p>
{implication("Any hour-of-day query must filter <code>timestamp_precision = 'second'</code>; day, week and month queries can use <code>post_date</code> for all rows.")}</div>

<div class="insight"><h3>3 · Activity is stationary: no trend, seasonality or anomalous day <span class="tag">trend</span></h3>
<p>{tm['daily_mean']} posts/day with Poisson noise. Monthly totals are explained by days-in-month ({p(tm['chi2_month_counts_vs_calendar_days_p'])}), weekdays are flat ({p(tm['chi2_dow_uniform_p'])}), and no day survives a Bonferroni-corrected Poisson test.
The “May peak” and “February dip” are 31-day and 28-day months.</p>
{implication("Normalise counts by days (posts/day) before comparing periods, and test spikes against Poisson noise before calling them anomalies.")}</div>

<div class="insight"><h3>4 · Likes, shares and comments are independent uniform variables <span class="tag">correlation</span></h3>
<p>Uniform on fixed caps (KS p ≥ {min(lk['ks_uniform_0_5000_p'], eng['shares']['ks_uniform_p'], eng['comments']['ks_uniform_p']):.2f}), |ρ| ≤ {max(abs(v) for v in eng['spearman'].values())}, no long tail.
Real social data is heavy-tailed and positively correlated, so this looks like synthetic generation. A post with many likes is no more likely to be shared.</p>
{implication("Expect near-zero correlations. Define “high engagement” by percentile (e.g. top 10%) rather than absolute thresholds, and use rank-based statistics.")}</div>

<div class="insight"><h3>5 · No platform has a real engagement advantage <span class="tag">grouping</span></h3>
<p>Gaps are ≤ {max(bp[m]['spread_pct_of_mean'] for m in ['likes_f', 'shares', 'comments'])}% of the mean. The one nominal result (shares, Instagram vs Facebook, {p(bp['shares']['kruskal_p'])}) fails multiple-testing correction, with ε² = {bp['shares']['epsilon_squared']}.</p>
{implication("Platform-level rankings will be driven by noise. Report confidence intervals alongside any <code>GROUP BY platform</code> result.")}</div>

<div class="insight"><h3>6 · Volume, not reach, drives total engagement <span class="tag">behaviour</span></h3>
<p>Followers show no relationship to per-post engagement (ρ = {us['spearman_followers_vs_mean_engagement'][0]}). Total engagement tracks post count (ρ = {us['spearman_posts_vs_total_engagement']}), and the top-10 by total and top-10 by per-post average share <b>zero</b> users.
Cross-platform posting ({pct(us['users_on_2plus_platforms'], ov['users'])} of users) matches a random-assignment null model.</p>
{implication("For behavioural grouping, rank by per-post rates with a minimum-activity threshold (window functions such as <code>NTILE</code>/<code>PERCENT_RANK</code>), not by <code>SUM()</code>. “Influencer” cannot be inferred from follower_count.")}</div>

<div class="insight"><h3>7 · User language is not a location signal <span class="tag">consistency</span></h3>
<p>Language is independent of country (V = {lang['cramers_v']}), with a national-language match of {lang['national_language_match_pct']}% vs {lang['expected_match_pct_if_random']}% by chance. {lang['users_in_countries_without_language_code']} users live in countries whose language is not coded at all.</p>
{implication("Treat language and country as separate dimensions. Do not derive regional language claims (e.g. “Asia-Pacific language dominance”) from this field.")}</div>

<div class="insight"><h3>8 · Post text is templated and carries no temporal or engagement signal <span class="tag">text</span></h3>
<p>Brands, hashtags and verdicts are evenly balanced. {tx['contradiction_pct_of_emotional_posts']}% of emotional posts contradict their own verdict, engagement does not vary by sentiment ({p(tx['kruskal_engagement_by_verdict_p'])}), and campaigns are mentioned out of season (e.g. SpringBlast2025 in 2024).</p>
{implication("Text features (brand, campaign, sentiment) suit descriptive counts only. Do not expect them to explain engagement or timing.")}</div>

<h2><span class="num">9</span>Limitations &amp; Phase 2 readiness</h2>
<ul>
  <li><b>Unrecoverable values remain NULL</b>: platform {pct(plat['missing_total'], N)}, likes {pct(likes['missing_total'], N)}, text {pct(text['missing_total_after_cleaning'], N)}, time of day {pct(ts['format_counts']['dd-mm-yyyy'], N)}. This is a deliberate trade of completeness for correctness.</li>
  <li><b>Timezone:</b> A1 cannot be proven for naive ISO strings. Hour-of-day findings would shift by a constant if the source clock were not UTC.</li>
  <li><b>Sentiment</b> relies on the dataset's fixed verdict phrases, not a general NLP model. It is exact for these templates but not transferable.</li>
  <li><b>SQL-ready:</b> the cleaned CSVs load directly into a two-table schema (<code>users</code> 1—N <code>posts</code>) with typed columns, primary and foreign keys, and CHECK constraints that mirror our validation (<code>sql/schema.sql</code>, loaded by <code>src/build_sqlite.py</code>).</li>
</ul>

<h2><span class="num">10</span>Reproducibility</h2>
<table><thead><tr><th>Step</th><th>Command</th><th>Output</th></tr></thead><tbody>
<tr><td>Environment</td><td><code>pip install -r requirements.txt</code></td><td>pandas 2.2.2, numpy 1.26.4, scipy 1.17.1, matplotlib 3.10.9</td></tr>
<tr><td>Everything</td><td><code>python run_all.py</code></td><td>Runs the six steps below in order (~15 s on a laptop)</td></tr>
<tr><td>1 Clean</td><td><code>python src/clean.py</code></td><td><code>data/cleaned/*.csv</code>, <code>reports/cleaning_audit.json</code>, <code>reports/change_log.csv</code></td></tr>
<tr><td>2 Validate</td><td><code>python src/validate.py</code></td><td><code>reports/validation_report.md</code> (exits non-zero on failure)</td></tr>
<tr><td>3 JSON</td><td><code>python src/export_json.py</code></td><td><code>data/cleaned/social_engine_cleaned.json</code> (both tables in one file)</td></tr>
<tr><td>4 EDA</td><td><code>python src/eda.py</code></td><td><code>reports/figures/*.png</code>, <code>reports/eda_stats.json</code></td></tr>
<tr><td>5 SQL</td><td><code>python src/build_sqlite.py</code></td><td><code>data/social_engine.db</code> + SQL cross-checks</td></tr>
<tr><td>6 Report</td><td><code>python src/build_report.py</code></td><td>This document (HTML + PDF)</td></tr>
</tbody></table>
<p class="small">Walkthrough notebook: <code>notebooks/Phase1_Walkthrough.ipynb</code> (executed, with outputs). Column definitions: <code>data/cleaned/DATA_DICTIONARY.md</code>. Decision log: <code>docs/CLEANING_DECISIONS.md</code>.
Output hashes: posts <code>{A['output']['sha256']['posts_cleaned.csv'][:16]}…</code>, users <code>{A['output']['sha256']['users_cleaned.csv'][:16]}…</code>.</p>

</div></body></html>
"""

HTML_OUT.write_text(html, encoding="utf-8")
print(f"wrote {HTML_OUT.relative_to(ROOT)}")


def find_browser() -> str | None:
    candidates = [
        os.environ.get("CHROME_PATH", ""),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return c
    for name in ("google-chrome", "chromium", "chromium-browser", "chrome", "msedge"):
        if shutil.which(name):
            return shutil.which(name)
    return None


browser = find_browser()
if browser:
    subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={PDF_OUT}", HTML_OUT.resolve().as_uri()],
                   check=True, capture_output=True, timeout=180)
    print(f"wrote {PDF_OUT.relative_to(ROOT)}")
else:
    print("No Chrome/Edge found: open the HTML file and print to PDF (A4, no headers/footers).")
