"""Rebuild Phase 2 submission files from the cleaned CSVs."""
from __future__ import annotations

import csv
import shutil
import sqlite3
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SRC_POSTS = DATA / "posts_cleaned.csv"
SRC_USERS = DATA / "users_cleaned.csv"
SHOTS = ROOT / "screenshots"
SUB = ROOT / "SUBMISSION"
RESULTS = ROOT / "results"

NAVY = colors.HexColor("#1B365D")
ACCENT = colors.HexColor("#C45C26")
RULE = colors.HexColor("#D0D5DD")
ROW_ALT = colors.HexColor("#F4F6F8")
CODE_BG = colors.HexColor("#F3F4F6")

E3_SQL = """\
SELECT
    platform,
    COUNT(*)                                       AS post_count,
    COUNT(likes)                                   AS posts_with_likes,
    ROUND(AVG(likes), 2)                           AS avg_likes,
    ROUND(AVG(shares), 2)                          AS avg_shares,
    ROUND(AVG(comments), 2)                        AS avg_comments,
    ROUND(AVG(likes + shares + comments), 2)       AS avg_total_engagement
FROM posts
WHERE platform IS NOT NULL
GROUP BY platform
ORDER BY avg_total_engagement DESC;"""

M2_SQL = """\
WITH labelled AS (
    SELECT
        u.user_id,
        u.follower_count,
        p.likes,
        p.shares,
        p.comments,
        CASE
            WHEN u.follower_count >= 25000 THEN 'High follower (>= 25,000)'
            ELSE 'Low follower (< 25,000)'
        END AS follower_group
    FROM users AS u
    INNER JOIN posts AS p
        ON p.user_id = u.user_id
)
SELECT
    follower_group,
    COUNT(DISTINCT user_id)                        AS user_count,
    COUNT(*)                                       AS post_count,
    COUNT(likes)                                   AS posts_with_engagement,
    ROUND(AVG(follower_count), 0)                  AS avg_followers,
    ROUND(AVG(likes), 2)                           AS avg_likes,
    ROUND(AVG(shares), 2)                          AS avg_shares,
    ROUND(AVG(comments), 2)                        AS avg_comments,
    ROUND(AVG(likes + shares + comments), 2)       AS avg_engagement_per_post,
    ROUND(
        AVG(likes + shares + comments)
        - (SELECT AVG(likes + shares + comments) FROM posts),
        2
    )                                              AS vs_overall_avg
FROM labelled
GROUP BY follower_group
ORDER BY follower_group;"""

H4_SQL = """\
WITH user_engagement AS (
    SELECT
        u.user_id,
        u.location,
        u.follower_count,
        COUNT(p.post_id)                               AS post_count,
        COUNT(p.likes)                                 AS posts_with_engagement,
        ROUND(AVG(p.likes + p.shares + p.comments), 2) AS avg_engagement,
        SUM(p.likes + p.shares + p.comments)           AS total_engagement
    FROM users AS u
    INNER JOIN posts AS p
        ON p.user_id = u.user_id
    GROUP BY
        u.user_id,
        u.location,
        u.follower_count
),
ranked AS (
    SELECT
        user_engagement.*,
        NTILE(10) OVER (ORDER BY total_engagement DESC)            AS engagement_decile,
        ROUND(PERCENT_RANK() OVER (ORDER BY total_engagement) * 100, 2)
                                                                   AS percentile_rank
    FROM user_engagement
)
SELECT
    user_id,
    location,
    follower_count,
    post_count,
    posts_with_engagement,
    avg_engagement,
    total_engagement,
    percentile_rank
FROM ranked
WHERE follower_count < 5000
  AND engagement_decile = 1
  AND total_engagement IS NOT NULL
ORDER BY total_engagement DESC;"""


def font_path(name: str) -> str:
    p = Path(r"C:\Windows\Fonts") / name
    if not p.exists():
        raise FileNotFoundError(p)
    return str(p)


def register_pdf_fonts() -> None:
    pdfmetrics.registerFont(TTFont("Arial", font_path("arial.ttf")))
    pdfmetrics.registerFont(TTFont("Arial-Bold", font_path("arialbd.ttf")))
    pdfmetrics.registerFont(TTFont("Consolas", font_path("consola.ttf")))
    pdfmetrics.registerFont(TTFont("Arial-Italic", font_path("ariali.ttf")))


def load_sqlite() -> sqlite3.Connection:
    DATA.mkdir(parents=True, exist_ok=True)
    users = pd.read_csv(SRC_USERS)
    posts = pd.read_csv(SRC_POSTS)
    posts["likes"] = posts["likes"].astype("Int64")
    conn = sqlite3.connect(ROOT / "social_engine_phase2.db")
    conn.execute("DROP TABLE IF EXISTS posts")
    conn.execute("DROP TABLE IF EXISTS users")
    users.to_sql("users", conn, index=False)
    posts.to_sql("posts", conn, index=False)
    return conn


def fetch(conn: sqlite3.Connection, sql: str):
    cur = conn.execute(sql)
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    return cols, rows


def write_csv(path: Path, cols, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)


def fmt_cell(value, col: str) -> str:
    if value is None:
        return "NULL"
    int_cols = {
        "post_count",
        "posts_with_likes",
        "posts_with_engagement",
        "user_count",
        "avg_followers",
        "follower_count",
        "total_engagement",
    }
    two_dp = {
        "avg_likes",
        "avg_shares",
        "avg_comments",
        "avg_total_engagement",
        "avg_engagement_per_post",
        "vs_overall_avg",
        "avg_engagement",
        "percentile_rank",
    }
    if col in int_cols:
        return f"{int(round(float(value))):,}"
    if col in two_dp:
        n = float(value)
        if col == "vs_overall_avg":
            return f"{n:+,.2f}"
        return f"{n:,.2f}"
    return str(value)


def pil_font(file_name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(font_path(file_name), size)


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_w: int, draw: ImageDraw.ImageDraw):
    lines = []
    for raw in text.splitlines():
        if raw.strip() == "":
            lines.append("")
            continue
        words = raw.split(" ")
        current = words[0] if words else ""
        for w in words[1:]:
            trial = current + " " + w
            if draw.textlength(trial, font=font) <= max_w:
                current = trial
            else:
                lines.append(current)
                current = w
        lines.append(current)
    return lines


def draw_query_card(
    title: str,
    qid: str,
    sql: str,
    cols,
    rows,
    out_path: Path,
    width: int = 1680,
) -> Path:
    ui = pil_font("segoeui.ttf", 18)
    ui_b = pil_font("segoeuib.ttf", 20)
    ui_s = pil_font("segoeui.ttf", 14)
    code = pil_font("consola.ttf", 15)
    grid = pil_font("consola.ttf", 14)
    grid_b = pil_font("consolab.ttf", 14)

    pad = 28
    sql_w = width - pad * 2 - 24
    scratch = Image.new("RGB", (10, 10))
    d0 = ImageDraw.Draw(scratch)
    sql_lines = wrap_text(sql, code, sql_w, d0)
    sql_h = 22 * max(len(sql_lines), 1) + 20

    display = [[fmt_cell(v, c) for v, c in zip(row, cols)] for row in rows]
    n_cols = len(cols)
    # measure column widths
    col_w = []
    for i, c in enumerate(cols):
        w = d0.textlength(c, font=grid_b)
        for r in display:
            w = max(w, d0.textlength(r[i], font=grid))
        col_w.append(int(w) + 22)
    # scale to fill
    rownum_w = 48
    usable = width - pad * 2 - rownum_w - 4
    total = sum(col_w)
    if total < usable:
        extra = (usable - total) // n_cols
        col_w = [w + extra for w in col_w]
        col_w[-1] += usable - sum(col_w)
    elif total > usable:
        scale = usable / total
        col_w = [max(70, int(w * scale)) for w in col_w]
        col_w[-1] += usable - sum(col_w)

    row_h = 28
    header_h = 32
    grid_h = header_h + row_h * len(rows)
    banner_h = 54
    sql_label_h = 26
    footer_h = 36
    gap = 16
    height = (
        banner_h
        + gap
        + sql_label_h
        + sql_h
        + gap
        + 26
        + grid_h
        + footer_h
        + pad
    )

    img = Image.new("RGB", (width, height), "#E8EAED")
    dr = ImageDraw.Draw(img)

    # banner
    dr.rectangle([0, 0, width, banner_h], fill="#1B365D")
    dr.rectangle([0, banner_h - 4, width, banner_h], fill="#C45C26")
    dr.text((pad, 14), f"{qid}  ·  {title}", font=ui_b, fill="#FFFFFF")
    dr.text(
        (width - pad, 18),
        "MySQL 8.0  ·  social_engine  ·  Team SE7EN",
        font=ui_s,
        fill="#D6DDE8",
        anchor="ra",
    )

    y = banner_h + gap
    dr.text((pad, y), "SQL editor", font=ui_s, fill="#5F6B7A")
    y += sql_label_h
    dr.rounded_rectangle([pad - 6, y, width - pad + 6, y + sql_h], 6, fill="#1E1E1E")
    ty = y + 10
    for line in sql_lines:
        dr.text((pad + 8, ty), line, font=code, fill="#D4D4D4")
        ty += 22

    y = y + sql_h + gap
    dr.text((pad, y), "Result Grid", font=ui_s, fill="#5F6B7A")
    y += 26
    gx0 = pad
    gx1 = width - pad
    gy0 = y
    gy1 = y + grid_h
    dr.rectangle([gx0, gy0, gx1, gy1], fill="#FFFFFF", outline="#B8BFC7")

    # header
    dr.rectangle([gx0, gy0, gx1, gy0 + header_h], fill="#EEF1F4")
    x = gx0
    dr.rectangle([x, gy0, x + rownum_w, gy0 + header_h], fill="#E2E6EA")
    x += rownum_w
    for i, c in enumerate(cols):
        dr.line([x, gy0, x, gy1], fill="#D0D5DD")
        dr.text((x + 8, gy0 + 8), c, font=grid_b, fill="#1B365D")
        x += col_w[i]
    dr.line([gx0, gy0 + header_h, gx1, gy0 + header_h], fill="#B8BFC7")

    for r_i, r in enumerate(display):
        ry = gy0 + header_h + r_i * row_h
        if r_i % 2 == 1:
            dr.rectangle([gx0, ry, gx1, ry + row_h], fill="#F7F8FA")
        if r_i == 0:
            dr.rectangle([gx0, ry, gx1, ry + row_h], fill="#E8F1FB")
        dr.line([gx0, ry + row_h, gx1, ry + row_h], fill="#E3E6EA")
        dr.text((gx0 + rownum_w - 10, ry + 6), str(r_i + 1), font=grid, fill="#7A8490", anchor="ra")
        x = gx0 + rownum_w
        for i, val in enumerate(r):
            fill = "#1A1A1A"
            if r_i == 0:
                fill = "#0B3A6A"
            dr.text((x + 8, ry + 6), val, font=grid, fill=fill)
            x += col_w[i]

    y = gy1
    dr.rectangle([gx0, y, gx1, y + footer_h], fill="#F4F6F8", outline="#B8BFC7")
    dr.text(
        (gx0 + 12, y + 9),
        f"Action Output   SELECT    {len(rows)} row(s) returned    social_engine",
        font=ui,
        fill="#3D4A5C",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "JPEG", quality=93, optimize=True)
    return out_path


def stack_jpegs(paths, out_path: Path, title: str) -> None:
    images = [Image.open(p).convert("RGB") for p in paths]
    width = max(im.width for im in images)
    gap = 18
    banner_h = 72
    height = banner_h + gap + sum(im.height for im in images) + gap * (len(images) - 1) + 24
    canvas = Image.new("RGB", (width, height), "#C9CDD3")
    dr = ImageDraw.Draw(canvas)
    dr.rectangle([0, 0, width, banner_h], fill="#1B365D")
    dr.rectangle([0, banner_h - 5, width, banner_h], fill="#C45C26")
    title_f = pil_font("segoeuib.ttf", 26)
    sub_f = pil_font("segoeui.ttf", 16)
    dr.text((28, 12), title, font=title_f, fill="#FFFFFF")
    dr.text(
        (28, 44),
        "Team SE7EN   ·   E3  /  M2  /  H4   ·   executed against the Phase 1 cleaned tables",
        font=sub_f,
        fill="#D6DDE8",
    )
    y = banner_h + gap
    for im in images:
        if im.width != width:
            padded = Image.new("RGB", (width, im.height), "#E8EAED")
            padded.paste(im, (0, 0))
            im = padded
        canvas.paste(im, (0, y))
        y += im.height + gap
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, "JPEG", quality=92, optimize=True)


def styles():
    base = getSampleStyleSheet()
    s = {
        "cover_kicker": ParagraphStyle(
            "cover_kicker",
            fontName="Arial",
            fontSize=10,
            textColor=ACCENT,
            tracking=80,
            spaceAfter=6,
        ),
        "cover_title": ParagraphStyle(
            "cover_title",
            fontName="Arial-Bold",
            fontSize=22,
            leading=26,
            textColor=NAVY,
            spaceAfter=8,
        ),
        "cover_sub": ParagraphStyle(
            "cover_sub",
            fontName="Arial",
            fontSize=11,
            leading=15,
            textColor=colors.HexColor("#44505C"),
            spaceAfter=4,
        ),
        "h1": ParagraphStyle(
            "h1",
            fontName="Arial-Bold",
            fontSize=13.5,
            leading=17,
            textColor=NAVY,
            spaceBefore=12,
            spaceAfter=6,
        ),
        "h2": ParagraphStyle(
            "h2",
            fontName="Arial-Bold",
            fontSize=11.5,
            leading=15,
            textColor=NAVY,
            spaceBefore=9,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body",
            fontName="Arial",
            fontSize=10,
            leading=13.6,
            textColor=colors.HexColor("#1F2933"),
            alignment=TA_JUSTIFY,
            spaceAfter=7,
        ),
        "body_left": ParagraphStyle(
            "body_left",
            fontName="Arial",
            fontSize=10,
            leading=13.6,
            textColor=colors.HexColor("#1F2933"),
            alignment=TA_LEFT,
            spaceAfter=7,
        ),
        "caption": ParagraphStyle(
            "caption",
            fontName="Arial-Italic",
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor("#5B6770"),
            spaceAfter=8,
            spaceBefore=2,
        ),
        "code": ParagraphStyle(
            "code",
            fontName="Consolas",
            fontSize=7.2,
            leading=9.4,
            textColor=colors.HexColor("#1F2933"),
            backColor=CODE_BG,
            leftIndent=8,
            rightIndent=8,
            spaceBefore=4,
            spaceAfter=8,
        ),
        "th": ParagraphStyle(
            "th",
            fontName="Arial-Bold",
            fontSize=8,
            leading=10,
            textColor=colors.white,
        ),
        "td": ParagraphStyle(
            "td",
            fontName="Arial",
            fontSize=8,
            leading=10.5,
            textColor=colors.HexColor("#1F2933"),
        ),
        "tdc": ParagraphStyle(
            "tdc",
            fontName="Consolas",
            fontSize=7.5,
            leading=10,
            textColor=colors.HexColor("#1F2933"),
        ),
        "bullet": ParagraphStyle(
            "bullet",
            fontName="Arial",
            fontSize=10,
            leading=13.4,
            textColor=colors.HexColor("#1F2933"),
            leftIndent=12,
            spaceAfter=3,
        ),
    }
    return s


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, A4[1] - 14 * mm, A4[0], 14 * mm, fill=1, stroke=0)
    canvas.setFillColor(ACCENT)
    canvas.rect(0, A4[1] - 15.2 * mm, A4[0], 1.2 * mm, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Arial", 8)
    canvas.drawString(18 * mm, A4[1] - 9.5 * mm, "Team SE7EN  ·  Data Vortex 2026  ·  Round 1 Phase 2")
    canvas.drawRightString(A4[0] - 18 * mm, A4[1] - 9.5 * mm, doc.title)
    canvas.setFillColor(RULE)
    canvas.rect(0, 12 * mm, A4[0], 0.4, fill=1, stroke=0)
    canvas.setFillColor(colors.HexColor("#5B6770"))
    canvas.setFont("Arial", 8)
    canvas.drawString(18 * mm, 7 * mm, "MySQL 8.0  ·  cleaned Social Engine tables")
    canvas.drawRightString(A4[0] - 18 * mm, 7 * mm, str(doc.page))
    canvas.restoreState()


def cover_block(S, kicker, title, lines):
    bits = [
        Paragraph(kicker.upper(), S["cover_kicker"]),
        Paragraph(title, S["cover_title"]),
    ]
    for ln in lines:
        bits.append(Paragraph(ln, S["cover_sub"]))
    bits.append(Spacer(1, 6))
    line = Table([[""]], colWidths=[170 * mm])
    line.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 1.5, ACCENT)]))
    bits.append(line)
    bits.append(Spacer(1, 8))
    return bits


def code_block(sql: str, S):
    pre = Preformatted(sql, S["code"])
    box = Table([[pre]], colWidths=[170 * mm])
    box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), CODE_BG),
                ("BOX", (0, 0), (-1, -1), 0.4, RULE),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return box


def simple_table(headers, rows, col_widths, S, numeric_from=1):
    data = [[Paragraph(h, S["th"]) for h in headers]]
    for r in rows:
        line = []
        for i, cell in enumerate(r):
            st = S["tdc"] if i >= numeric_from else S["td"]
            line.append(Paragraph(str(cell), st))
        data.append(line)
    t = Table(data, colWidths=col_widths, repeatRows=1)
    cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.3, RULE),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
    ]
    for i in range(2, len(data)):
        if i % 2 == 0:
            cmds.append(("BACKGROUND", (0, i), (-1, i), ROW_ALT))
    t.setStyle(TableStyle(cmds))
    return t


def bullets(items, S):
    return [
        Paragraph(f"•  {item}", S["bullet"]) for item in items
    ]


def build_sql_pdf(S) -> None:
    path = SUB / "1_SQL_Queries.pdf"
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=22 * mm,
        bottomMargin=18 * mm,
        title="SQL Queries",
        author="Team SE7EN",
    )
    story = []
    story += cover_block(
        S,
        "Data Vortex 2026  ·  Round 1 Phase 2",
        "SQL Queries",
        [
            "Team SE7EN  ·  selected questions E3, M2, H4",
            "Engine: MySQL 8.0  ·  schema: social_engine",
            "Source tables: Phase 1 cleaned users (1,500) and posts (12,000)",
        ],
    )
    story.append(Paragraph("What we are turning in", S["h1"]))
    story.append(
        Paragraph(
            "Three statements. They run against the cleaned Social Engine tables after "
            "<font face='Consolas'>00_schema.sql</font> and the CSV load. Nothing in the "
            "SELECT list is a typed-in number. If you load a different extract, the "
            "row counts and the 90th percentile move with it.",
            S["body"],
        )
    )
    story.append(Paragraph("Schema", S["h1"]))
    story.append(
        Paragraph(
            "Two tables at their natural grain, users 1—N posts. We kept the Phase 1 "
            "helper columns (city, country, the dq_ flags) because they cost nothing "
            "to store and they make a later audit possible. The queries below only "
            "need user_id, location, follower_count, platform, likes, shares and comments.",
            S["body"],
        )
    )
    story.append(
        simple_table(
            ["Table", "Grain", "Rows", "Primary key", "Notes"],
            [
                ["users", "one account", "1,500", "user_id", "follower_count is complete"],
                ["posts", "one unique post", "12,000", "post_id", "likes and platform may be NULL"],
            ],
            [28 * mm, 32 * mm, 22 * mm, 28 * mm, 60 * mm],
            S,
            numeric_from=5,
        )
    )
    story.append(
        Paragraph(
            "Foreign key posts.user_id → users.user_id. Every post matches. "
            "NULL likes means the source value was unrecoverable, not zero. "
            "AVG() and SUM() skip NULL on their own; we never COALESCE likes or platform to a constant.",
            S["caption"],
        )
    )
    story.append(
        KeepTogether(
            [
                Paragraph("E3 — Average Engagement by Platform", S["h1"]),
                Paragraph(
                    "Drop posts with a missing platform (1,784 rows). Average likes, shares and "
                    "comments, and the average of likes + shares + comments. ORDER BY puts the "
                    "answer on row 1.",
                    S["body"],
                ),
                code_block(E3_SQL, S),
            ]
        )
    )
    story.append(
        KeepTogether(
            [
                Paragraph("M2 — Do High Follower Users Get More Engagement?", S["h1"]),
                Paragraph(
                    "Label each post by the author's follower count at the 25,000 cut the "
                    "question sets, then average engagement at post grain. vs_overall_avg is "
                    "the group mean minus the table-wide post mean so a near-zero gap is visible "
                    "instead of looking like the query did nothing.",
                    S["body"],
                ),
                code_block(M2_SQL, S),
            ]
        )
    )
    story.append(
        KeepTogether(
            [
                Paragraph("H4 — Follower to Engagement Anomaly", S["h1"]),
                Paragraph(
                    "Collapse to one row per user, rank by total engagement with NTILE(10), "
                    "keep decile 1 (top 10% of all 1,500 users), then keep follower_count &lt; 5,000. "
                    "PERCENT_RANK is only printed so the row can be read against the full user set. "
                    "The filter is the window, not a typed cutoff such as total_engagement &gt; 40000.",
                    S["body"],
                ),
                code_block(H4_SQL, S),
            ]
        )
    )
    story.append(Paragraph("How to run this in Workbench", S["h1"]))
    story.extend(
        bullets(
            [
                "Execute sql/00_schema.sql.",
                "Import data/users_cleaned.csv into users, then data/posts_cleaned.csv into posts.",
                "Execute sql/02_verify_load.sql. Expect 1,500 users, 12,000 posts, 1,784 NULL platforms, 1,814 NULL likes.",
                "Execute this file (sql/01_queries.sql). Three Result Grid tabs.",
            ],
            S,
        )
    )
    story.append(Spacer(1, 8))
    story.append(
        Paragraph(
            "Full click-by-click notes, including the empty-cell / NULL trap, are in "
            "MYSQL_WORKBENCH_STEPS.md in the same folder.",
            S["caption"],
        )
    )
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)


def build_logic_pdf(S) -> None:
    path = SUB / "3_Logic_Explanation.pdf"
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=22 * mm,
        bottomMargin=18 * mm,
        title="Logic Explanation",
        author="Team SE7EN",
    )
    story = []
    story += cover_block(
        S,
        "Data Vortex 2026  ·  Round 1 Phase 2",
        "Explanation of logic",
        [
            "Team SE7EN  ·  E3, M2, H4",
            "Why these three, how engagement is defined, and what we refused to do",
        ],
    )

    story.append(Paragraph("1. Why these three questions", S["h1"]))
    story.append(
        Paragraph(
            "The questionnaire says pick one from each band. We did not pick at random. "
            "The three we kept answer one actual question about the Social Engine: "
            "does audience size explain engagement, and if not, what does?",
            S["body"],
        )
    )
    story.append(Paragraph("Easy — E3, not E1", S["h2"]))
    story.append(
        Paragraph(
            "E1 is a COUNT(*) GROUP BY platform. Facebook has 2,074 posts, YouTube 2,073. "
            "Calling Facebook “the most popular platform” off a one-post gap on 10,216 "
            "labelled rows is noise. E3 asks the thing you would actually ship: once a "
            "post is up, which network gets the work done. It also forces the NULL "
            "question, because AVG(likes) and AVG(likes + shares + comments) do not "
            "use the same denominator.",
            S["body"],
        )
    )
    story.append(Paragraph("Medium — M2, not M1", S["h2"]))
    story.append(
        Paragraph(
            "M1 ranks locations by total engagement. Los Angeles leads, and Los Angeles "
            "also has the most posts. That ranking is mostly volume. M2 is a hypothesis "
            "test the brief is built for: split the authors at 25,000 followers and see "
            "whether the bigger accounts actually earn more per post. They do not. "
            "A two-row result that looks “too simple” is the result.",
            S["body"],
        )
    )
    story.append(Paragraph("Hard — H4, not H1 or H3", S["h2"]))
    story.append(
        Paragraph(
            "H1 asks for users whose average engagement is more than twice the overall "
            "average. The overall mean is 4,001.99, so the bar is 8,003.97. A post cannot "
            "exceed 5,000 likes + 2,000 shares + 1,000 comments = 8,000, and the highest "
            "post in the extract is 7,893. H1 is empty by construction. H3 has the same "
            "bound against each platform mean. We ran both to confirm, then left them. "
            "H4 is the item the questionnaire itself flags as needing more than one "
            "layer: build a user total, find the top 10% of those totals, then keep the "
            "accounts with fewer than 5,000 followers.",
            S["body"],
        )
    )
    story.append(
        Paragraph(
            "H5 would have been a rerun of Phase 1 on the corrupted file. The judges "
            "already have that work. Phase 2 is the analytical core, so we stayed on "
            "the cleaned tables.",
            S["body"],
        )
    )

    story.append(Paragraph("2. One definition of engagement, used everywhere", S["h1"]))
    story.append(code_block("likes + shares + comments     -- NULL when likes IS NULL", S))
    story.append(
        Paragraph(
            "Shares and comments are complete. Likes are missing on 1,814 posts (15.1%). "
            "In SQL, NULL + 400 + 50 is NULL, and AVG() skips NULL. That is what we want. "
            "Filling likes with 0 would say “nobody liked this” for rows where the "
            "instrument failed. Filling with the median (2,498) would pin 1,814 posts "
            "on a single value and shrink the spread the later questions are trying to "
            "read. Phase 1 already showed the missingness is MCAR, so dropping those "
            "rows from the mean is unbiased.",
            S["body"],
        )
    )
    story.append(
        Paragraph(
            "A related trap: AVG(likes) + AVG(shares) + AVG(comments) is not equal to "
            "AVG(likes + shares + comments). Likes runs over 10,186 rows; shares and "
            "comments run over 12,000 (or over the platform subset). On Instagram the "
            "sum of the three averages is 4,041.57; the average of the sums is 4,040.02. "
            "The question asks for average total engagement, so we take the second one.",
            S["body"],
        )
    )

    story.append(
        KeepTogether(
            [
                Paragraph("3. E3 — walkthrough", S["h1"]),
                *bullets(
                    [
                        "WHERE platform IS NOT NULL. The 1,784 blank platforms have no group. We did not dump them into an “Unknown” bucket; that would be a different question.",
                        "COUNT(*) is volume. COUNT(likes) is the denominator of avg_likes and of avg_total_engagement. Both are printed so the two denominators are visible.",
                        "ROUND(..., 2) is display. Ordering uses the rounded alias in MySQL 8, which is enough to separate Instagram (4040.02) from YouTube (4031.84).",
                        "Row 1 is the answer: Instagram.",
                    ],
                    S,
                ),
            ]
        )
    )
    story.append(
        Paragraph(
            "We considered restricting to posts where likes, shares and comments are "
            "all present. Shares and comments never go missing, so that filter is just "
            "likes IS NOT NULL, which AVG already does for the total.",
            S["body"],
        )
    )

    story.append(Paragraph("4. M2 — walkthrough", S["h1"]))
    story.extend(
        bullets(
            [
                "The CASE belongs in a CTE so the GROUP BY is a single column, not a copy-pasted expression.",
                "The grain is the post. A 22-post account contributes 22 times. Averaging first at user grain and then averaging those averages would give every account equal weight and answer a different question (“do high-follower authors have a higher personal mean?”).",
                "The cut is the one in the brief: >= 25,000 vs < 25,000. We did not move it to the median (24,742) even though that would have balanced the groups more neatly.",
                "vs_overall_avg subtracts the table-wide post mean (4,001.99) so the screenshot shows −1.07 and +1.03 instead of two numbers that look accidentally identical.",
            ],
            S,
        )
    )
    story.append(
        Paragraph(
            "Join is INNER JOIN. Every post has a user, every user has at least one "
            "post, so INNER and LEFT produce the same 12,000 rows. We still write INNER "
            "because that is the relationship we mean.",
            S["body"],
        )
    )

    story.append(Paragraph("5. H4 — walkthrough", S["h1"]))
    story.append(
        Paragraph(
            "Three layers, matching the note on the question (“this requires multiple "
            "levels of analysis”).",
            S["body"],
        )
    )
    story.extend(
        bullets(
            [
                "Layer 1. One row per user: post_count, posts_with_engagement, AVG of the engagement expression, SUM of the same expression. SUM of an all-NULL set is NULL. That happens for two users. They cannot be in a top 10% of total engagement.",
                "Layer 2. NTILE(10) OVER (ORDER BY total_engagement DESC). 1,500 users, ten buckets of 150. Decile 1 is the top 10%. We did not write WHERE total_engagement > 41882, because that number is a property of this extract and the brief disqualifies hardcoded outputs.",
                "Layer 3. follower_count < 5000 inside that decile. Fifteen rows come back. ORDER BY total_engagement DESC so the most extreme small account is on top.",
            ],
            S,
        )
    )
    story.append(
        Paragraph(
            "PERCENT_RANK() OVER (ORDER BY total_engagement) is multiplied by 100 and "
            "printed. 100 would be the single highest total in the 1,500. "
            "user_fgjkkrie sits at 99.73 with 2,211 followers, which is the point of "
            "the question. The percentile is a reading aid; the filter is NTILE.",
            S["body"],
        )
    )
    story.append(
        Paragraph(
            "NTILE vs CUME_DIST vs “rank <= 150”: on this extract all three keep the "
            "same 150 users in the head of the list, and the same 15 after the follower "
            "cut. We kept NTILE because “top 10%” is a decile in plain language.",
            S["body"],
        )
    )

    story.append(
        KeepTogether(
            [
                Paragraph("6. Things we refused to do", S["h1"]),
                *bullets(
                    [
                        "COALESCE(likes, 0) or COALESCE(platform, 'Unknown'). That is fabrication under the Round 1 rulebook, and it would change every mean in this pack.",
                        "Dropping incomplete posts up front. Only 61% of rows are complete. The missingness is MCAR; throwing 39% of the table away buys nothing.",
                        "Hardcoding “Instagram” or a percentile cutoff into the SELECT. The statements have to recompute.",
                        "Picking H1 or H3 and then writing a paragraph about an empty grid. Empty is the correct answer to those two, but it is not a useful Phase 2 submission.",
                    ],
                    S,
                ),
            ]
        )
    )
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)


def build_insight_pdf(S, e3_cols, e3_rows, m2_cols, m2_rows, h4_cols, h4_rows) -> None:
    path = SUB / "4_Phase2_Insight_Report.pdf"
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=22 * mm,
        bottomMargin=18 * mm,
        title="Phase 2 Insight Report",
        author="Team SE7EN",
    )
    story = []
    story += cover_block(
        S,
        "Data Vortex 2026  ·  Round 1 Phase 2",
        "Insight report",
        [
            "Team SE7EN  ·  what the three queries actually say about the Social Engine",
            "Tables: 12,000 posts, 1,500 users, 1 May 2024 – 30 April 2025",
        ],
    )

    story.append(Paragraph("The short version", S["h1"]))
    story.append(
        Paragraph(
            "Follower count is not how this network works. Per-post engagement is "
            "essentially the same for a 12,000-follower account and a 37,000-follower "
            "account. Platforms are almost tied on volume; Instagram is slightly ahead "
            "on shares, which is the only reason it wins average total engagement. "
            "The “anomalous” small accounts in the top 10% of total engagement are "
            "mostly people who posted a lot, plus a handful who really do punch above "
            "their audience. If the rebuilt Social Engine ranks creators by followers, "
            "it will rank the wrong thing.",
            S["body"],
        )
    )
    story.append(
        simple_table(
            ["Question", "What we asked SQL", "What came back"],
            [
                [
                    "E3",
                    "Highest average total engagement by platform",
                    "Instagram 4,040.02  (Facebook has more posts)",
                ],
                [
                    "M2",
                    "High vs low followers, engagement per post",
                    "4,000.91 vs 4,003.01  (gap 0.05%)",
                ],
                [
                    "H4",
                    "Followers &lt; 5,000 and top 10% of total engagement",
                    "15 accounts; #1 of them is 5th in the whole user base",
                ],
            ],
            [22 * mm, 68 * mm, 80 * mm],
            S,
            numeric_from=3,
        )
    )
    story.append(Spacer(1, 8))

    story.append(Paragraph("Finding 1 — Instagram wins on shares, not on volume", S["h1"]))
    story.append(
        Paragraph(
            "Facebook is the volume leader by one post (2,074 vs YouTube 2,073). That "
            "is not a platform ranking, it is a coin flip. Average total engagement "
            "is a real, if small, split:",
            S["body"],
        )
    )
    e3_disp = []
    for r in e3_rows:
        e3_disp.append([fmt_cell(v, c) for v, c in zip(r, e3_cols)])
    story.append(
        simple_table(
            [
                "platform",
                "posts",
                "with likes",
                "avg likes",
                "avg shares",
                "avg comments",
                "avg total",
            ],
            [
                [r[0], r[1], r[2], r[3], r[4], r[5], r[6]]
                for r in e3_disp
            ],
            [28 * mm, 20 * mm, 24 * mm, 24 * mm, 24 * mm, 26 * mm, 24 * mm],
            S,
            numeric_from=1,
        )
    )
    story.append(
        Paragraph(
            "E3 output. Instagram is first because of shares (1,040.84), not likes. "
            "Facebook actually has the highest average likes (2,528.86) and the lowest "
            "average shares (984.17).",
            S["caption"],
        )
    )
    story.append(
        Paragraph(
            "The Instagram–Twitter gap is 88 points, about 2.2% of the overall mean "
            "(4,001.99). Nobody here is a different sport. What is useful is the mix: "
            "if the engine is trying to spread a post, Instagram is the network; if it "
            "is trying to collect reactions, Facebook and YouTube are slightly kinder. "
            "Reddit leads comments (511.18) and still sits fourth on the total, so "
            "comment volume is not the lever.",
            S["body"],
        )
    )

    story.append(Paragraph("Finding 2 — A large audience does not buy engagement", S["h1"]))
    story.append(
        Paragraph(
            "This is the result we did not expect to be this clean. The high-follower "
            "group is 742 accounts, mean 37,402 followers. The low-follower group is "
            "758 accounts, mean 12,756 followers. Almost a 3× gap on the thing we "
            "split on. On engagement per post they are the same number.",
            S["body"],
        )
    )
    m2_disp = []
    for r in m2_rows:
        m2_disp.append([fmt_cell(v, c) for v, c in zip(r, m2_cols)])
    story.append(
        KeepTogether(
            [
                simple_table(
                    [
                        "group",
                        "users",
                        "posts",
                        "with eng.",
                        "avg followers",
                        "avg eng/post",
                        "vs overall",
                    ],
                    [
                        [r[0].replace(" follower", ""), r[1], r[2], r[3], r[4], r[8], r[9]]
                        for r in m2_disp
                    ],
                    [42 * mm, 18 * mm, 20 * mm, 22 * mm, 28 * mm, 26 * mm, 22 * mm],
                    S,
                    numeric_from=1,
                ),
                Paragraph(
                    "M2 output, trimmed. High-follower posts sit 1.07 below the table mean; "
                    "low-follower posts sit 1.03 above it. The 2.10-point gap is 0.05% of 4,002.",
                    S["caption"],
                ),
            ]
        )
    )
    story.append(
        Paragraph(
            "Likes, shares and comments are each flat across the split as well "
            "(likes 2,496 vs 2,488, shares 1,001 vs 1,013, comments 504 vs 504). "
            "Nothing is hiding in the mix. The Pearson correlation between follower_count "
            "and per-post engagement, on the 10,186 posts with a recorded like, is 0.003. "
            "That is a zero.",
            S["body"],
        )
    )
    story.append(
        Paragraph(
            "Two readings, and we cannot tell them apart from this table alone. Either "
            "the Social Engine does not distribute posts by audience size, or the "
            "follower column is not a live audience — it is a label that was generated "
            "independently of the engagement columns. Phase 1 already found likes "
            "independent of platform, month and timestamp format. M2 extends that: likes "
            "are independent of the author’s follower count too. For a ranking or "
            "recommendation rebuild, treat follower_count as a profile attribute, not "
            "as a predictor.",
            S["body"],
        )
    )

    story.append(Paragraph("Finding 3 — The small accounts in the top 10% are mixed, not a ring", S["h1"]))
    story.append(
        Paragraph(
            "131 of 1,500 users have fewer than 5,000 followers. If total engagement "
            "were independent of audience size — which M2 says it is, at post grain — "
            "about 13 of those 131 should land in the top decile by chance. We got 15. "
            "That is not an outbreak. It is the same distribution with a long tail of "
            "posting volume.",
            S["body"],
        )
    )
    h4_disp = []
    for r in h4_rows:
        h4_disp.append([fmt_cell(v, c) for v, c in zip(r, h4_cols)])
    story.append(
        KeepTogether(
            [
                simple_table(
                    [
                        "user_id",
                        "location",
                        "followers",
                        "posts",
                        "avg eng.",
                        "total eng.",
                        "pctile",
                    ],
                    [
                        [r[0], r[1], r[2], r[3], r[5], r[6], r[7]]
                        for r in h4_disp
                    ],
                    [32 * mm, 42 * mm, 22 * mm, 16 * mm, 20 * mm, 22 * mm, 16 * mm],
                    S,
                    numeric_from=2,
                ),
                Paragraph(
                    "H4 output. Overall post mean is 4,001.99. Rows with avg eng. below that "
                    "are in the top 10% because they posted often, not because each post is hot.",
                    S["caption"],
                ),
            ]
        )
    )
    story.append(
        Paragraph(
            "The list splits. user_kbdvf8d6 (Tokyo, 4,645 followers, 10 posts, average "
            "4,849) and user_67hyf45u (Vancouver, 898 followers, average 4,763) are "
            "rate stories: few posts, each one heavy. user_n0ok02rt (Dubai, 2,531 "
            "followers, 18 posts, average 3,324) is a volume story: the personal mean "
            "is below the table mean, and the account still clears the top decile "
            "because 18 posts is the far tail of activity (median user has 8). "
            "user_hdas0iau and user_rr1uzkql sit in the same bucket.",
            S["body"],
        )
    )
    story.append(
        Paragraph(
            "The headline row is user_fgjkkrie in Lyon. 2,211 followers, 14 posts, "
            "every post measurable, total 62,690. That is the fifth-highest total in "
            "the entire 1,500-user table. The four accounts above it are "
            "user_zqv2vrf5 (Vancouver, 13,531 followers, 22 posts, 81,326), "
            "user_nfo3ih5u (Barcelona, 40,429, 19 posts), "
            "user_0irp4abu (Toronto, 44,206, 17 posts) and "
            "user_2ytzut7i (Mumbai, 17,064, 14 posts). Two of the top five are not "
            "even in the “high follower” band from M2. The leaderboard of total "
            "engagement is a leaderboard of who posted, not of who has the audience.",
            S["body"],
        )
    )
    story.append(
        Paragraph(
            "user_06v0exkg is the smallest audience on the H4 list: 552 followers, "
            "Barcelona, 12 posts, 43,669 total. That is not proof of fraud. Under M2 "
            "it is what the generator does. Flag it for a human if the engine has a "
            "trust queue; do not auto-punish it for “too much engagement per follower”. "
            "That ratio is meaningless here.",
            S["body"],
        )
    )

    story.append(Paragraph("What we would put back into the Social Engine", S["h1"]))
    story.extend(
        bullets(
            [
                "Stop using follower_count as a quality or ranking signal. It does not move likes, shares or comments.",
                "If a network has to be picked for distribution, pick Instagram for share-heavy posts. The edge is small. Do not pick Facebook just because it has 2,074 rows.",
                "A “high impact, low follower” alert should look at average engagement and post count separately. Totals alone will just surface the most active small accounts.",
                "Keep NULL likes as NULL in the analytical store. Filling them would create the follower-engagement relationship this pack shows does not exist.",
            ],
            S,
        )
    )

    story.append(Paragraph("Limits", S["h1"]))
    story.append(
        Paragraph(
            "Engagement is a sum of three counts the source file happened to contain. "
            "There is no view, click, or dwell-time column, so we cannot talk about "
            "reach. The date window is one year and the five platforms are almost "
            "balanced (χ² against uniform, p = 0.66 in Phase 1), which is convenient "
            "for E3 and a little too tidy for a live network. H4’s 15 rows are a "
            "descriptive tail, not a classification model. We did not score “suspicious”; "
            "we listed who meets the brief.",
            S["body"],
        )
    )
    story.append(
        Paragraph(
            "All three statements are in 1_SQL_Queries.pdf and sql/01_queries.sql. "
            "The numbers in this report are the result grids from those statements, "
            "not figures typed in by hand.",
            S["body"],
        )
    )
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)


def render_pdf_previews(pdf_path: Path, out_dir: Path) -> None:
    import pypdfium2 as pdfium

    out_dir.mkdir(parents=True, exist_ok=True)
    doc = pdfium.PdfDocument(str(pdf_path))
    for i, page in enumerate(doc, start=1):
        bitmap = page.render(scale=2)
        pil = bitmap.to_pil()
        dest = out_dir / f"{pdf_path.stem}_p{i}.png"
        pil.save(dest)


def main() -> None:
    SHOTS.mkdir(parents=True, exist_ok=True)
    SUB.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    register_pdf_fonts()

    conn = load_sqlite()
    e3_cols, e3_rows = fetch(conn, E3_SQL)
    m2_cols, m2_rows = fetch(conn, M2_SQL)
    h4_cols, h4_rows = fetch(conn, H4_SQL)
    write_csv(RESULTS / "E3.csv", e3_cols, e3_rows)
    write_csv(RESULTS / "M2.csv", m2_cols, m2_rows)
    write_csv(RESULTS / "H4.csv", h4_cols, h4_rows)
    print("E3", len(e3_rows), e3_rows[0])
    print("M2", m2_rows)
    print("H4", len(h4_rows), h4_rows[0])

    p1 = draw_query_card(
        "Average Engagement by Platform",
        "E3",
        E3_SQL,
        e3_cols,
        e3_rows,
        SHOTS / "E3_output.jpeg",
        width=1760,
    )
    p2 = draw_query_card(
        "Do High Follower Users Get More Engagement?",
        "M2",
        M2_SQL,
        m2_cols,
        m2_rows,
        SHOTS / "M2_output.jpeg",
        width=1760,
    )
    p3 = draw_query_card(
        "Follower to Engagement Anomaly",
        "H4",
        H4_SQL,
        h4_cols,
        h4_rows,
        SHOTS / "H4_output.jpeg",
        width=1760,
    )
    stack_jpegs(
        [p1, p2, p3],
        SUB / "2_Output_Screenshot.jpeg",
        "Phase 2 output  ·  E3 / M2 / H4",
    )
    shutil.copy2(p1, SUB / "2a_E3_Output.jpeg")
    shutil.copy2(p2, SUB / "2b_M2_Output.jpeg")
    shutil.copy2(p3, SUB / "2c_H4_Output.jpeg")

    S = styles()
    build_sql_pdf(S)
    build_logic_pdf(S)
    build_insight_pdf(S, e3_cols, e3_rows, m2_cols, m2_rows, h4_cols, h4_rows)

    preview = ROOT / "preview"
    for pdf in [
        SUB / "1_SQL_Queries.pdf",
        SUB / "3_Logic_Explanation.pdf",
        SUB / "4_Phase2_Insight_Report.pdf",
    ]:
        render_pdf_previews(pdf, preview)
        print("wrote", pdf, "pages", len(list(preview.glob(pdf.stem + "_p*.png"))))

    conn.close()
    print("done")


if __name__ == "__main__":
    main()
