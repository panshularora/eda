"""Render the analytical report PDF from its LaTeX source, without a TeX install.

The report's single source is ``reports/Round3_Analytical_Report_Team_SE7EN.tex``.
When no TeX engine is available, this converts that file - only the subset of
LaTeX it actually uses - into HTML with the figures embedded, and prints it to
PDF with a local headless Chrome or Edge. The content therefore cannot drift
from the .tex: there is no second copy of the prose to keep in step.

    python round3/src/build_report_pdf.py
"""
from __future__ import annotations

import html
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPORTS = Path(__file__).resolve().parents[1] / "reports"
TEX = REPORTS / "Round3_Analytical_Report_Team_SE7EN.tex"
HTML_OUT = REPORTS / "Round3_Analytical_Report_Team_SE7EN.html"
PDF_OUT = REPORTS / "Round3_Analytical_Report_Team_SE7EN.pdf"
BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "google-chrome", "chromium", "msedge",
]

BS = "\\"
PH_OPEN, PH_CLOSE = "\ue000", "\ue001"


# ---------------------------------------------------------------------------
# small parsing helpers
# ---------------------------------------------------------------------------
def strip_comments(s: str) -> str:
    out = []
    for line in s.splitlines():
        buf, i = [], 0
        while i < len(line):
            if line[i] == "%" and (i == 0 or line[i - 1] != BS):
                break
            buf.append(line[i])
            i += 1
        out.append("".join(buf))
    return "\n".join(out)


def group(s: str, i: int) -> tuple[str, int]:
    """Content of the brace group starting at s[i] == '{', and the index after it."""
    assert s[i] == "{", s[i:i + 30]
    depth, j = 0, i
    while j < len(s):
        c = s[j]
        if c == BS:
            j += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    raise ValueError("unbalanced group")


def skip_ws_opt(s: str, i: int) -> int:
    """Skip whitespace and one optional [...] argument."""
    while i < len(s) and s[i] in " \t\n":
        i += 1
    if i < len(s) and s[i] == "[":
        i = s.index("]", i) + 1
    return i


def args(s: str, i: int, n: int) -> tuple[list[str], int]:
    out = []
    for _ in range(n):
        while s[i] in " \t\n":
            i += 1
        a, i = group(s, i)
        out.append(a)
    return out, i


# ---------------------------------------------------------------------------
# inline conversion
# ---------------------------------------------------------------------------
MATH_SYMBOLS = {
    r"\times": "\u00d7", r"\to": "\u2192", r"\geq": "\u2265", r"\leq": "\u2264",
    r"\approx": "\u2248", r"\kappa": "\u03ba", r"\rho": "\u03c1", r"\tau": "\u03c4",
    r"\cdot": "\u00b7", r"\downarrow": "\u2193", r"\qquad": "\u2003\u2003",
    r"\,": "\u2009", r"\;": "\u2009",
}


def math(m: str) -> str:
    m = re.sub(r"\\(?:text|mathrm)\{([^}]*)\}", lambda k: "\ue010" + k.group(1) + "\ue011", m)
    m = re.sub(r"\\hat\{?p\}?", "p\u0302", m)
    m = m.replace(BS + "%", "%")
    for k, v in MATH_SYMBOLS.items():
        m = m.replace(k, v)
    m = m.replace("-", "\u2212")
    m = re.sub(r"\^\{([^}]*)\}", r"<sup>\1</sup>", m)
    m = re.sub(r"\^(\S)", r"<sup>\1</sup>", m)
    m = re.sub(r"_\{([^}]*)\}", r"<sub>\1</sub>", m)
    m = re.sub(r"_(\w)", r"<sub>\1</sub>", m)
    # italic single-letter variables, upright everything else
    m = re.sub(r"(?<![\w\ue010&#;])([A-Za-z])(?![\w\ue011;])", r"<i>\1</i>", m)
    m = m.replace("\ue010", "").replace("\ue011", "")
    return f'<span class="m">{m}</span>'


def conv(s: str, labels: dict) -> str:
    store: list[str] = []

    def keep(frag: str) -> str:
        store.append(frag)
        return f"{PH_OPEN}{len(store) - 1}{PH_CLOSE}"

    s = re.sub(r"(?<!\\)\$(.+?)(?<!\\)\$", lambda m: keep(math(m.group(1))), s, flags=re.S)

    def code_cmd(text: str, name: str) -> str:
        key = BS + name + "{"
        while key in text:
            i = text.index(key)
            inner, j = group(text, i + len(key) - 1)
            inner = re.sub(r"\\detokenize\{(.*)\}", r"\1", inner)
            text = text[:i] + keep(f"<code>{html.escape(inner)}</code>") + text[j:]
        return text

    s = code_cmd(s, "code")
    s = code_cmd(s, "texttt")
    s = s.replace(BS + "&", keep("&amp;")).replace(BS + "%", "%")
    s = html.escape(s, quote=False)

    s = re.sub(r"\\(?:S)\\ref\{([^}]+)\}", lambda m: "\u00a7" + labels.get(m.group(1), "?"), s)
    s = re.sub(r"\\ref\{([^}]+)\}", lambda m: labels.get(m.group(1), "?"), s)
    s = re.sub(r"\\label\{[^}]*\}", "", s)
    s = re.sub(r"\\vspace\*?\{[^}]*\}", "", s)
    # size and layout switches carry no text; strip them before \color, because
    # "\small\color{muted}DATA" would otherwise become the unbreakable "\smallDATA"
    s = re.sub(r"\\(smallskip|medskip|bigskip|footnotesize|normalsize|small|large|Large|"
               r"LARGE|bfseries|centering|noindent|hfill|relax|par)(?![a-z])\s*", "", s)
    s = re.sub(r"\\color\{[^}]*\}", "", s)

    for name, (o, c) in {"textbf": ("<b>", "</b>"), "emph": ("<i>", "</i>"),
                         "textit": ("<i>", "</i>"), "underline": ("<u>", "</u>")}.items():
        key = BS + name + "{"
        while key in s:
            i = s.index(key)
            inner, j = group(s, i + len(key) - 1)
            s = s[:i] + o + inner + c + s[j:]

    s = re.sub(r"\\\\(\[[^\]]*\])?", "<br>", s)
    for a, b in (("``", "\u201c"), ("''", "\u201d"), ("`", "\u2018"),
                 ("---", "\u2014"), ("--", "\u2013"), ("~", "\u00a0"),
                 (BS + "ldots", "\u2026"), (BS + "S", "\u00a7"),
                 (BS + "quad", "\u2003"), (BS + ",", "\u2009"), (BS + ";", " ")):
        s = s.replace(a, b)
    s = re.sub(r"\\(small|footnotesize|normalsize|large|Large|LARGE|bfseries|centering|"
               r"noindent|smallskip|medskip|bigskip|par|hfill|relax)\b\s*", "", s)
    s = s.replace("{", "").replace("}", "")
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(PH_OPEN + r"(\d+)" + PH_CLOSE, lambda m: store[int(m.group(1))], s)
    return s


# ---------------------------------------------------------------------------
# numbering pass: sections, tables, figures, and what each label points at
# ---------------------------------------------------------------------------
def number(body: str) -> tuple[dict, list[str]]:
    labels: dict[str, str] = {}
    heads: list[str] = []
    sec = sub = tab = fig = 0
    appendix = False
    current = ""
    pat = re.compile(r"\\section\*|\\section\{|\\subsection\{|\\appendix|"
                     r"\\begin\{table\}|\\reportfig\{|\\label\{")
    for m in pat.finditer(body):
        t = m.group(0)
        if t == BS + "appendix":
            appendix, sec = True, 0
        elif t == BS + "section{":
            sec += 1
            sub = 0
            current = chr(64 + sec) if appendix else str(sec)
            heads.append(current)
        elif t == BS + "subsection{":
            sub += 1
            current = f"{chr(64 + sec) if appendix else sec}.{sub}"
            heads.append(current)
        elif t == BS + "begin{table}":
            tab += 1
            current = str(tab)
        elif t == BS + "reportfig{":
            fig += 1
            a4, _ = args(body, m.end() - 1, 4)
            labels[a4[3]] = str(fig)
        elif t == BS + "label{":
            name, _ = group(body, m.end() - 1)
            labels.setdefault(name, current)
    return labels, heads


# ---------------------------------------------------------------------------
# block conversion
# ---------------------------------------------------------------------------
def table_html(env: str, labels: dict, n: int) -> str:
    cap = ""
    if BS + "caption{" in env:
        i = env.index(BS + "caption{")
        cap, _ = group(env, i + len(BS + "caption"))
    blocks = []
    for m in re.finditer(r"\\begin\{(tabularx|tabular)\}", env):
        kind = m.group(1)
        i = m.end()
        if kind == "tabularx":
            _, i = args(env, i, 1)
        (spec,), i = args(env, i, 1)
        end = env.index(BS + "end{" + kind + "}", i)
        content = env[i:end]
        cols = re.findall(r"[lrcX]", re.sub(r"@\{[^}]*\}", "", spec))
        pieces = re.split(r"\\\\(?:\[[^\]]*\])?", content)
        head, bodyrows, mids = [], [], 0
        for piece in pieces:
            has_mid = BS + "midrule" in piece
            mids += has_mid
            piece = re.sub(r"\\(toprule|bottomrule|midrule)", "", piece).strip()
            if not piece:
                continue
            row = []
            for k, c in enumerate(re.split(r"(?<!\\)&", piece)):
                align = cols[k] if k < len(cols) else "l"
                row.append((conv(c, labels), ' class="r"' if align == "r" else ""))
            # rows before the first \midrule are the header; a later \midrule
            # draws a rule above the row it precedes
            if mids == 0:
                head.append((row, False))
            else:
                bodyrows.append((row, has_mid and mids >= 2))
        if mids == 0:                       # a table with no \midrule has no header
            head, bodyrows = [], head
        rows_html = []
        if head:
            for row, _ in head:
                rows_html.append("<tr>" + "".join(f"<th{c}>{t}</th>" for t, c in row) + "</tr>")
        for row, b in bodyrows:
            tr = '<tr class="rule">' if b else "<tr>"
            rows_html.append(tr + "".join(f"<td{c}>{t}</td>" for t, c in row) + "</tr>")
        blocks.append("<table>" + "".join(rows_html) + "</table>")
    inner = blocks[0] if len(blocks) == 1 else \
        '<div class="side">' + "".join(f"<div>{b}</div>" for b in blocks) + "</div>"
    return (f'<figure class="tab"><figcaption><b>Table {n}.</b> '
            f"{conv(cap, labels)}</figcaption>{inner}</figure>")


def list_html(content: str, ordered: bool, labels: dict) -> str:
    content = content[skip_ws_opt(content, 0):]
    items = [x for x in re.split(r"\\item\b", content) if x.strip()]
    tag = "ol" if ordered else "ul"
    return f"<{tag}>" + "".join(f"<li>{conv(x, labels)}</li>" for x in items) + f"</{tag}>"


def paragraphs(text: str, labels: dict) -> str:
    out = []
    for chunk in re.split(r"\n\s*\n", text):
        c = chunk.strip()
        if not c or not re.sub(r"\\(smallskip|medskip|par|noindent)", "", c).strip():
            continue
        cls = ' class="small"' if c.startswith("{" + BS + "small") else ""
        body = conv(c, labels)
        if body:
            out.append(f"<p{cls}>{body}</p>")
    return "\n".join(out)


DISPLAY_EQ = (
    '<div class="eq"><span class="m"><i>D\u0302</i><sub>bt</sub> = <i>N</i><sub>bt</sub>'
    ' <i>p\u0302</i><sub>bt</sub>,\u2003\u2003SE(<i>D\u0302</i><sub>bt</sub>) = '
    '<i>N</i><sub>bt</sub> \u00b7 \u221a[ <i>p\u0302</i><sub>bt</sub>(1 \u2212 '
    '<i>p\u0302</i><sub>bt</sub>) / <i>n</i><sub>bt</sub> ] \u00b7 \u221a[ '
    '(<i>N</i><sub>bt</sub> \u2212 <i>n</i><sub>bt</sub>) / (<i>N</i><sub>bt</sub> '
    '\u2212 1) ]</span></div>')


def blocks(s: str, labels: dict, heads: list, figdir: Path, counters: dict) -> str:
    starts = [BS + "section*{", BS + "section{", BS + "subsection{", BS + "appendix",
              BS + "keybox{", BS + "begin{itemize}", BS + "begin{enumerate}",
              BS + "begin{table}", BS + "reportfig{", BS + "begin{verbatim}",
              BS + "[", BS + "begin{center}"]
    out, i = [], 0
    while True:
        hits = [(s.find(k, i), k) for k in starts]
        hits = [(p, k) for p, k in hits if p != -1]
        if not hits:
            out.append(paragraphs(s[i:], labels))
            break
        p, k = min(hits)
        out.append(paragraphs(s[i:p], labels))
        if k == BS + "appendix":
            i = p + len(k)
        elif k in (BS + "section*{", BS + "section{", BS + "subsection{"):
            title, j = group(s, p + len(k) - 1)
            if k == BS + "section*{":
                out.append(f"<h2>{conv(title, labels)}</h2>")
            else:
                num = heads[counters["head"]]
                counters["head"] += 1
                tag = "h2" if k == BS + "section{" else "h3"
                out.append(f'<{tag}><span class="num">{num}</span>{conv(title, labels)}</{tag}>')
            i = j
        elif k == BS + "keybox{":
            inner, i = group(s, p + len(k) - 1)
            out.append(f'<div class="box">{blocks(inner, labels, heads, figdir, counters)}</div>')
        elif k in (BS + "begin{itemize}", BS + "begin{enumerate}"):
            env = "itemize" if "itemize" in k else "enumerate"
            end = s.index(BS + "end{" + env + "}", p)
            out.append(list_html(s[p + len(k):end], env == "enumerate", labels))
            i = end + len(BS + "end{" + env + "}")
        elif k == BS + "begin{table}":
            end = s.index(BS + "end{table}", p)
            counters["table"] += 1
            out.append(table_html(s[p:end], labels, counters["table"]))
            i = end + len(BS + "end{table}")
        elif k == BS + "reportfig{":
            (f, w, cap, lab), i = args(s, p + len(k) - 1, 4)
            counters["figure"] += 1
            src = (figdir / f).resolve().as_uri()
            out.append(f'<figure class="fig"><img src="{src}" style="width:{float(w)*100:.0f}%">'
                       f'<figcaption><b>Figure {counters["figure"]}.</b> '
                       f"{conv(cap, labels)}</figcaption></figure>")
        elif k == BS + "begin{verbatim}":
            end = s.index(BS + "end{verbatim}", p)
            out.append("<pre>" + html.escape(s[p + len(k):end].strip("\n")) + "</pre>")
            i = end + len(BS + "end{verbatim}")
        elif k == BS + "[":
            end = s.index(BS + "]", p)
            out.append(DISPLAY_EQ)
            i = end + 2
        elif k == BS + "begin{center}":
            end = s.index(BS + "end{center}", p)
            lines = re.split(r"\\\\(?:\[[^\]]*\])?", s[p + len(k):end])
            classes = ["kicker", "title", "subtitle", "authors", "meta"]
            out.append('<header>' + "".join(
                f'<div class="{classes[n] if n < len(classes) else "meta"}">'
                f"{conv(line, labels)}</div>" for n, line in enumerate(lines) if line.strip())
                + "</header>")
            i = end + len(BS + "end{center}")
    return "\n".join(x for x in out if x)


CSS = """
@page { size: A4; margin: 19mm 20mm 20mm 20mm;
  @bottom-left { content: "Team SE7EN \\00b7  Round 3 Analytical Report"; font: 8pt 'Segoe UI', sans-serif; color: #5B6470; }
  @bottom-right { content: counter(page); font: 8pt 'Segoe UI', sans-serif; color: #5B6470; } }
body { font-family: Cambria, 'Times New Roman', serif; font-size: 10.6pt; line-height: 1.42; color: #1b1f24; }
header { text-align: center; margin-bottom: 14px; }
header .kicker { font: 8.5pt 'Segoe UI', sans-serif; letter-spacing: .06em; color: #5B6470; }
header .title { font-size: 21pt; font-weight: bold; margin: 8px 0 3px; }
header .subtitle { font-size: 13pt; }
header .authors { margin-top: 9px; }
header .meta { font-size: 9pt; color: #5B6470; margin-top: 3px; }
h2 { font-family: 'Segoe UI', sans-serif; color: #1F4E79; font-size: 14pt; margin: 20px 0 6px; break-after: avoid; }
h3 { font-family: 'Segoe UI', sans-serif; font-size: 11pt; margin: 13px 0 4px; break-after: avoid; }
.num { margin-right: .55em; }
p { margin: 0 0 7px; text-align: justify; hyphens: auto; }
p.small { font-size: 9.3pt; }
ul, ol { margin: 3px 0 8px; padding-left: 1.4em; }
li { margin-bottom: 3px; text-align: justify; }
.box { background: #EEF3F8; border: 1px solid #1F4E79; padding: 7px 12px 3px; margin: 4px 0 8px; }
figure { margin: 10px 0 12px; break-inside: avoid; }
figure.fig { text-align: center; }
figure.fig img { max-width: 100%; }
figcaption { font-size: 9pt; text-align: left; margin: 4px 0; }
figure.tab figcaption { margin-bottom: 5px; }
table { border-collapse: collapse; font-size: 8.9pt; width: 100%; border-top: 1.4px solid #1b1f24; border-bottom: 1.4px solid #1b1f24; }
th { text-align: left; font-weight: bold; border-bottom: 0.8px solid #1b1f24; padding: 3px 6px; vertical-align: bottom; }
td { padding: 2.5px 6px; vertical-align: top; }
tr.rule td { border-top: 0.8px solid #1b1f24; }
.r { text-align: right; white-space: nowrap; }
.side { display: flex; gap: 18px; align-items: flex-start; }
.side > div { flex: 1; }
.side > div:first-child { flex: 1.6; }
code { font-family: Consolas, monospace; font-size: 8.8pt; }
pre { font-family: Consolas, monospace; font-size: 8.8pt; background: #f5f6f8; padding: 8px 10px; }
.m { font-family: Cambria, 'Cambria Math', serif; white-space: nowrap; }
.eq { text-align: center; margin: 8px 0 10px; }
"""


def build_html() -> str:
    src = strip_comments(TEX.read_text(encoding="utf-8"))
    body = src[src.index(BS + "begin{document}") + len(BS + "begin{document}"):
               src.index(BS + "end{document}")]
    labels, heads = number(body)
    counters = {"head": 0, "table": 0, "figure": 0}
    content = blocks(body, labels, heads, REPORTS / "figures", counters)
    doc = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
           f"<title>Round 3 Analytical Report - Team SE7EN</title><style>{CSS}</style>"
           f"</head><body>{content}</body></html>")
    HTML_OUT.write_text(doc, encoding="utf-8")
    return doc


def main() -> Path:
    build_html()
    browser = next((b for b in BROWSERS if Path(b).exists() or shutil.which(b)), None)
    if not browser:
        raise SystemExit("no Chrome or Edge found to print the PDF")
    subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    "--allow-file-access-from-files", f"--print-to-pdf={PDF_OUT}",
                    HTML_OUT.resolve().as_uri()], check=True, timeout=180,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"  wrote {PDF_OUT.name} ({PDF_OUT.stat().st_size / 1e6:.2f} MB)")
    return PDF_OUT


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
