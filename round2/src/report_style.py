"""Shared ReportLab furniture for the two round-2 PDFs.

Both reports are built from the same primitives so they read as one pack:
same cover, same running header, same table and callout treatment.
"""
from __future__ import annotations

from datetime import date

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Flowable, Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
    Spacer, Table, TableStyle,
)

from config import EVENT, FIGURES, MEMBERS, TEAM

INK = colors.HexColor("#1b1f24")
MUTED = colors.HexColor("#5b6472")
RULE = colors.HexColor("#d7dbe0")
ACCENT = colors.HexColor("#2563eb")
SOFT = colors.HexColor("#eef3fb")
WARN = colors.HexColor("#fdf3e7")
WARN_EDGE = colors.HexColor("#ea7317")
BAND = colors.HexColor("#f5f7fa")

PAGE_W, PAGE_H = A4
MARGIN = 19 * mm
CONTENT_W = PAGE_W - 2 * MARGIN

_base = getSampleStyleSheet()

S = {
    "title": ParagraphStyle("title", parent=_base["Title"], fontName="Helvetica-Bold",
                            fontSize=25, leading=30, textColor=INK, alignment=TA_CENTER,
                            spaceAfter=4),
    "subtitle": ParagraphStyle("subtitle", parent=_base["Normal"], fontName="Helvetica",
                               fontSize=12.5, leading=17, textColor=MUTED,
                               alignment=TA_CENTER),
    "cover_meta": ParagraphStyle("cover_meta", parent=_base["Normal"],
                                 fontName="Helvetica", fontSize=9.5, leading=15,
                                 textColor=MUTED, alignment=TA_CENTER),
    "h1": ParagraphStyle("h1", parent=_base["Heading1"], fontName="Helvetica-Bold",
                         fontSize=15, leading=19, textColor=INK,
                         spaceBefore=15, spaceAfter=3),
    "h2": ParagraphStyle("h2", parent=_base["Heading2"], fontName="Helvetica-Bold",
                         fontSize=11, leading=14.5, textColor=INK,
                         spaceBefore=11, spaceAfter=3),
    "h3": ParagraphStyle("h3", parent=_base["Heading3"], fontName="Helvetica-Bold",
                         fontSize=9.5, leading=13, textColor=ACCENT,
                         spaceBefore=8, spaceAfter=2),
    "body": ParagraphStyle("body", parent=_base["BodyText"], fontName="Helvetica",
                           fontSize=9.2, leading=13.6, textColor=INK,
                           alignment=TA_JUSTIFY, spaceAfter=5),
    "bullet": ParagraphStyle("bullet", parent=_base["BodyText"], fontName="Helvetica",
                             fontSize=9.2, leading=13.4, textColor=INK,
                             leftIndent=11, bulletIndent=2, spaceAfter=2.5),
    "caption": ParagraphStyle("caption", parent=_base["Normal"], fontName="Helvetica-Oblique",
                              fontSize=7.8, leading=10.5, textColor=MUTED,
                              alignment=TA_CENTER, spaceBefore=2, spaceAfter=7),
    "mono": ParagraphStyle("mono", parent=_base["Normal"], fontName="Courier",
                           fontSize=7.8, leading=10.6, textColor=INK),
    "callout": ParagraphStyle("callout", parent=_base["BodyText"], fontName="Helvetica",
                              fontSize=9.0, leading=13.2, textColor=INK,
                              alignment=TA_JUSTIFY),
    "callout_head": ParagraphStyle("callout_head", parent=_base["Normal"],
                                   fontName="Helvetica-Bold", fontSize=9.0,
                                   leading=12.5, textColor=ACCENT),
    "cell": ParagraphStyle("cell", parent=_base["Normal"], fontName="Helvetica",
                           fontSize=7.6, leading=10, textColor=INK),
    "cell_head": ParagraphStyle("cell_head", parent=_base["Normal"],
                                fontName="Helvetica-Bold", fontSize=7.6,
                                leading=10, textColor=colors.white),
}


class HRule(Flowable):
    """Thin horizontal rule used under section headings."""

    def __init__(self, width=CONTENT_W, thickness=0.7, colour=RULE, space=3):
        super().__init__()
        self.width, self.thickness, self.colour, self.space = width, thickness, colour, space
        self.height = thickness + space

    def draw(self):
        self.canv.setStrokeColor(self.colour)
        self.canv.setLineWidth(self.thickness)
        self.canv.line(0, self.space, self.width, self.space)


def h1(text, number=None):
    label = f"{number}. {text}" if number else text
    return [Paragraph(label, S["h1"]), HRule(), Spacer(1, 3)]


def h2(text):
    return Paragraph(text, S["h2"])


def h3(text):
    return Paragraph(text, S["h3"])


def p(text):
    return Paragraph(text, S["body"])


def bullets(items, marker="•"):
    return [Paragraph(f"{marker}&nbsp;&nbsp;{i}", S["bullet"]) for i in items]


_FIGURE_COUNT = {"n": 0}


def reset_figures():
    """Restart figure numbering; each PDF calls this before it starts building."""
    _FIGURE_COUNT["n"] = 0


def figure(name, caption, width=CONTENT_W, max_height=None):
    """Place a figure from reports/figures, scaled to the text column.

    Captions are numbered here rather than written by hand, so inserting or
    moving a figure cannot leave the numbering wrong.
    """
    _FIGURE_COUNT["n"] += 1
    caption = f"Figure {_FIGURE_COUNT['n']}. {caption}"
    path = FIGURES / name
    img = Image(str(path))
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth = width
    img.drawHeight = width * ratio
    if max_height and img.drawHeight > max_height:
        img.drawHeight = max_height
        img.drawWidth = max_height / ratio
    img.hAlign = "CENTER"
    return KeepTogether([Spacer(1, 3), img, Paragraph(caption, S["caption"])])


def table(rows, widths=None, align=None, highlight_first_row=True, font_size=7.6,
          zebra=True, bold_rows=()):
    """Header-banded table with optional zebra striping and bolded body rows."""
    data = [[Paragraph(str(c), S["cell_head"] if (r == 0 and highlight_first_row)
                       else S["cell"]) for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if highlight_first_row else 0,
              hAlign="LEFT")
    style = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, -2), 0.3, RULE),
        ("BOX", (0, 0), (-1, -1), 0.5, RULE),
    ]
    if highlight_first_row:
        style += [("BACKGROUND", (0, 0), (-1, 0), INK),
                  ("LINEBELOW", (0, 0), (-1, 0), 0.6, INK)]
    if zebra:
        start = 1 if highlight_first_row else 0
        for r in range(start, len(rows)):
            if (r - start) % 2 == 1:
                style.append(("BACKGROUND", (0, r), (-1, r), BAND))
    for r in bold_rows:
        style.append(("BACKGROUND", (0, r), (-1, r), SOFT))
    if align:
        for col, a in align.items():
            style.append(("ALIGN", (col, 0), (col, -1), a))
    t.setStyle(TableStyle(style))
    return t


def callout(title, body, tone="info"):
    """Boxed finding. ``tone='warn'`` for the data-quality findings."""
    bg, edge = (SOFT, ACCENT) if tone == "info" else (WARN, WARN_EDGE)
    head = ParagraphStyle("ch", parent=S["callout_head"],
                          textColor=ACCENT if tone == "info" else WARN_EDGE)
    inner = [[Paragraph(title, head)], [Paragraph(body, S["callout"])]]
    t = Table(inner, colWidths=[CONTENT_W])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("LINEBEFORE", (0, 0), (0, -1), 2.2, edge),
        ("BOX", (0, 0), (-1, -1), 0.4, edge),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (0, 0), 6),
        ("BOTTOMPADDING", (0, -1), (-1, -1), 7),
        ("TOPPADDING", (0, 1), (-1, -1), 1),
    ]))
    return KeepTogether([Spacer(1, 4), t, Spacer(1, 6)])


def metric_cards(pairs):
    """A row of big-number cards: [(value, label), ...]."""
    value_style = ParagraphStyle("mv", fontName="Helvetica-Bold", fontSize=15,
                                 leading=18, textColor=ACCENT, alignment=TA_CENTER)
    label_style = ParagraphStyle("ml", fontName="Helvetica", fontSize=7.2,
                                 leading=9.4, textColor=MUTED, alignment=TA_CENTER)
    cells = [[Paragraph(v, value_style) for v, _ in pairs],
             [Paragraph(l, label_style) for _, l in pairs]]
    w = CONTENT_W / len(pairs)
    t = Table(cells, colWidths=[w] * len(pairs))
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), SOFT),
        ("BOX", (0, 0), (-1, -1), 0.4, RULE),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.white),
        ("TOPPADDING", (0, 0), (-1, 0), 8),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 0),
        ("TOPPADDING", (0, 1), (-1, 1), 0),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return KeepTogether([Spacer(1, 2), t, Spacer(1, 8)])


def cover(title, subtitle, deliverable, extra_rows):
    """Full cover page; ``extra_rows`` is a list of (label, value) strings."""
    flow = [Spacer(1, 46 * mm)]
    tag = ParagraphStyle("tag", fontName="Helvetica-Bold", fontSize=8.5, leading=11,
                         textColor=ACCENT, alignment=TA_CENTER)
    flow.append(Paragraph(deliverable.upper(), tag))
    flow.append(Spacer(1, 7))
    flow.append(Paragraph(title, S["title"]))
    flow.append(Spacer(1, 2))
    flow.append(Paragraph(subtitle, S["subtitle"]))
    flow.append(Spacer(1, 9))
    flow.append(HRule(width=58 * mm))
    flow.append(Spacer(1, 11))
    flow.append(Paragraph(f"<b>{TEAM}</b>", S["cover_meta"]))
    flow.append(Paragraph(MEMBERS, S["cover_meta"]))
    flow.append(Spacer(1, 6))
    flow.append(Paragraph(EVENT, S["cover_meta"]))
    flow.append(Paragraph(date.today().strftime("%d %B %Y"), S["cover_meta"]))
    flow.append(Spacer(1, 16))
    if extra_rows:
        rows = [[Paragraph(f"<b>{k}</b>", S["cell"]), Paragraph(v, S["cell"])]
                for k, v in extra_rows]
        t = Table(rows, colWidths=[52 * mm, 78 * mm], hAlign="CENTER")
        t.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 2.6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.6),
            ("LINEBELOW", (0, 0), (-1, -2), 0.3, RULE),
            ("BOX", (0, 0), (-1, -1), 0.4, RULE),
            ("BACKGROUND", (0, 0), (0, -1), BAND),
        ]))
        flow.append(t)
    flow.append(PageBreak())
    return flow


def _decorations(title):
    def draw(canv, doc):
        canv.saveState()
        if doc.page > 1:
            canv.setFont("Helvetica", 7)
            canv.setFillColor(MUTED)
            canv.drawString(MARGIN, PAGE_H - MARGIN + 6 * mm, title)
            canv.drawRightString(PAGE_W - MARGIN, PAGE_H - MARGIN + 6 * mm, TEAM)
            canv.setStrokeColor(RULE)
            canv.setLineWidth(0.5)
            canv.line(MARGIN, PAGE_H - MARGIN + 4.6 * mm,
                      PAGE_W - MARGIN, PAGE_H - MARGIN + 4.6 * mm)
            canv.line(MARGIN, MARGIN - 4.5 * mm, PAGE_W - MARGIN, MARGIN - 4.5 * mm)
            canv.drawCentredString(PAGE_W / 2, MARGIN - 8.5 * mm, str(doc.page))
            canv.drawRightString(PAGE_W - MARGIN, MARGIN - 8.5 * mm,
                                 "Data Vortex A'26 · Round 2")
        else:
            canv.setFillColor(ACCENT)
            canv.rect(0, PAGE_H - 9 * mm, PAGE_W, 9 * mm, stroke=0, fill=1)
            canv.setFillColor(INK)
            canv.rect(0, 0, PAGE_W, 5 * mm, stroke=0, fill=1)
        canv.restoreState()
    return draw


def build(path, title, flow):
    doc = SimpleDocTemplate(
        str(path), pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN + 4 * mm, bottomMargin=MARGIN + 3 * mm,
        title=title, author=f"{TEAM} - {MEMBERS}", subject=EVENT,
    )
    deco = _decorations(title)
    doc.build(flow, onFirstPage=deco, onLaterPages=deco)
    return path
