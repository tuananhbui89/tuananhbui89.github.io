from pathlib import Path
import re
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path("/Users/tuananhbui/Personal/Gitpages/al-folio")
SOURCE = ROOT / "output/pdf/report-source.md"
OUTPUT = ROOT / "output/pdf/online-learning-reprogrammable-hardware-deep-research.pdf"

PAGE = landscape(A4)
PAGE_W, PAGE_H = PAGE
MARGIN_X = 13 * mm
MARGIN_TOP = 15 * mm
MARGIN_BOTTOM = 14 * mm
CONTENT_W = PAGE_W - 2 * MARGIN_X

NAVY = colors.HexColor("#12334A")
NAVY_2 = colors.HexColor("#204E68")
TEAL = colors.HexColor("#138A8A")
PALE_TEAL = colors.HexColor("#E8F5F4")
PALE_BLUE = colors.HexColor("#EDF4F8")
PALE_GOLD = colors.HexColor("#FFF6DF")
PALE_RED = colors.HexColor("#FBEDEC")
INK = colors.HexColor("#17232D")
MUTED = colors.HexColor("#536774")
GRID = colors.HexColor("#C8D4DB")
WHITE = colors.white


def rich(text: str) -> str:
    """Escape text and support the small Markdown subset used in the report."""
    text = escape(text.strip())
    text = re.sub(
        r"\[([^\]]+)\]\((https?://[^)]+)\)",
        lambda m: f'<link href="{m.group(2)}" color="#087F8C"><u>{m.group(1)}</u></link>',
        text,
    )
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)
    return text


styles = getSampleStyleSheet()
title_style = ParagraphStyle(
    "ReportTitle",
    parent=styles["Title"],
    fontName="Helvetica-Bold",
    fontSize=24,
    leading=28,
    textColor=NAVY,
    alignment=TA_LEFT,
    spaceAfter=10,
)
subtitle_style = ParagraphStyle(
    "Subtitle",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=12,
    leading=15,
    textColor=TEAL,
    spaceAfter=8,
)
h2_style = ParagraphStyle(
    "H2",
    parent=styles["Heading2"],
    fontName="Helvetica-Bold",
    fontSize=15,
    leading=18,
    textColor=NAVY,
    spaceBefore=10,
    spaceAfter=7,
    keepWithNext=True,
)
h3_style = ParagraphStyle(
    "H3",
    parent=styles["Heading3"],
    fontName="Helvetica-Bold",
    fontSize=11.5,
    leading=14,
    textColor=NAVY_2,
    spaceBefore=8,
    spaceAfter=5,
    keepWithNext=True,
)
body_style = ParagraphStyle(
    "Body",
    parent=styles["BodyText"],
    fontName="Helvetica",
    fontSize=9.2,
    leading=12.2,
    textColor=INK,
    spaceAfter=5,
    allowWidows=0,
    allowOrphans=0,
)
meta_style = ParagraphStyle(
    "Meta",
    parent=body_style,
    fontSize=8.8,
    leading=11.5,
    textColor=MUTED,
    spaceAfter=3,
)
bullet_style = ParagraphStyle(
    "Bullet",
    parent=body_style,
    leftIndent=0,
    firstLineIndent=0,
    spaceAfter=2,
)
callout_style = ParagraphStyle(
    "Callout",
    parent=body_style,
    fontName="Helvetica-Bold",
    textColor=NAVY,
    leading=12.5,
)
ref_style = ParagraphStyle(
    "Reference",
    parent=body_style,
    fontSize=7.7,
    leading=9.6,
    textColor=MUTED,
    spaceAfter=4,
)
table_header_style = ParagraphStyle(
    "TableHeader",
    parent=body_style,
    fontName="Helvetica-Bold",
    fontSize=7.2,
    leading=8.5,
    textColor=WHITE,
    alignment=TA_LEFT,
    spaceAfter=0,
)
table_cell_style = ParagraphStyle(
    "TableCell",
    parent=body_style,
    fontSize=6.8,
    leading=8.2,
    textColor=INK,
    spaceAfter=0,
    splitLongWords=True,
)
table_cell_small = ParagraphStyle(
    "TableCellSmall",
    parent=table_cell_style,
    fontSize=6.35,
    leading=7.65,
)


def column_widths(headers, count):
    header_text = " ".join(headers).lower()
    if count == 6 and "rank" in header_text:
        fractions = [0.045, 0.17, 0.075, 0.215, 0.18, 0.315]
    elif count == 6:
        fractions = [0.17, 0.047, 0.075, 0.205, 0.205, 0.298]
    elif count == 5:
        fractions = [0.16, 0.22, 0.21, 0.19, 0.22]
    elif count == 4:
        fractions = [0.16, 0.29, 0.22, 0.33]
    elif count == 3:
        fractions = [0.17, 0.35, 0.48]
    elif count == 2:
        fractions = [0.28, 0.72]
    else:
        fractions = [1.0 / count] * count
    return [CONTENT_W * f for f in fractions]


def make_table(raw_rows):
    headers = raw_rows[0]
    count = len(headers)
    cell_style = table_cell_small if count >= 6 else table_cell_style
    data = []
    for ridx, row in enumerate(raw_rows):
        row = (row + [""] * count)[:count]
        style = table_header_style if ridx == 0 else cell_style
        data.append([Paragraph(rich(cell), style) for cell in row])

    tbl = Table(
        data,
        colWidths=column_widths(headers, count),
        repeatRows=1,
        hAlign="LEFT",
        splitByRow=1,
        spaceBefore=3,
        spaceAfter=7,
    )
    ts = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("GRID", (0, 0), (-1, -1), 0.3, GRID),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
    ]
    for ridx in range(1, len(data)):
        ts.append(("BACKGROUND", (0, ridx), (-1, ridx), WHITE if ridx % 2 else PALE_BLUE))

    # Visually flag benefit and safety terms without relying on icons.
    for ridx, row in enumerate(raw_rows[1:], start=1):
        for cidx, cell in enumerate(row):
            low = cell.lower().strip()
            if cidx <= 2 and low.startswith("high"):
                ts.append(("BACKGROUND", (cidx, ridx), (cidx, ridx), PALE_TEAL))
            if "avoid autonomous" in low or low == "avoid":
                ts.append(("BACKGROUND", (cidx, ridx), (cidx, ridx), PALE_RED))
            if "conditional" in low and cidx <= 2:
                ts.append(("BACKGROUND", (cidx, ridx), (cidx, ridx), PALE_GOLD))

    tbl.setStyle(TableStyle(ts))
    return tbl


def parse_table(lines, start):
    rows = []
    i = start
    while i < len(lines) and lines[i].lstrip().startswith("|"):
        cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
        if not all(re.fullmatch(r":?-{3,}:?", c.replace(" ", "")) for c in cells):
            rows.append(cells)
        i += 1
    return rows, i


def make_bullets(items, ordered=False):
    if ordered:
        number_style = ParagraphStyle(
            "ListNumber",
            parent=bullet_style,
            fontName="Helvetica-Bold",
            textColor=TEAL,
            alignment=TA_LEFT,
        )
        rows = [
            [Paragraph(f"{idx}.", number_style), Paragraph(rich(item), bullet_style)]
            for idx, item in enumerate(items, start=1)
        ]
        return Table(
            rows,
            colWidths=[23, CONTENT_W - 23],
            hAlign="LEFT",
            spaceAfter=6,
            style=TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (0, -1), 5),
                    ("RIGHTPADDING", (1, 0), (1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 1),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ]
            ),
        )
    flow_items = []
    for item in items:
        flow_items.append(ListItem(Paragraph(rich(item), bullet_style), leftIndent=10))
    return ListFlowable(
        flow_items,
        bulletType="bullet",
        leftIndent=18,
        bulletFontName="Helvetica-Bold",
        bulletFontSize=7.5,
        bulletColor=TEAL,
        spaceAfter=6,
    )


def parse_markdown(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    story = []
    i = 0
    first_title = True
    in_sources = False
    inserted_section_break = False

    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if stripped.startswith("# "):
            if first_title:
                story.extend(
                    [
                        Spacer(1, 8 * mm),
                        Paragraph(rich(stripped[2:]), title_style),
                        Table([[""]], colWidths=[55 * mm], rowHeights=[2.2 * mm],
                              style=TableStyle([("BACKGROUND", (0, 0), (-1, -1), TEAL)])),
                        Spacer(1, 6 * mm),
                    ]
                )
                first_title = False
            else:
                story.append(Paragraph(rich(stripped[2:]), title_style))
            i += 1
            continue

        if stripped.startswith("## "):
            heading = stripped[3:]
            if heading == "Executive findings":
                story.append(PageBreak())
            if heading.startswith("1. ") and not inserted_section_break:
                story.append(PageBreak())
                inserted_section_break = True
            if heading.startswith("3. ") or heading.startswith("9. "):
                story.append(PageBreak())
            in_sources = heading.startswith("10. Sources")
            story.append(Paragraph(rich(heading), h2_style))
            i += 1
            continue

        if stripped.startswith("### "):
            story.append(Paragraph(rich(stripped[4:]), h3_style))
            i += 1
            continue

        if stripped.startswith("|"):
            rows, i = parse_table(lines, i)
            if rows:
                story.append(make_table(rows))
            continue

        if re.match(r"^[-*] ", stripped):
            items = []
            while i < len(lines) and re.match(r"^[-*] ", lines[i].strip()):
                items.append(re.sub(r"^[-*] ", "", lines[i].strip()))
                i += 1
            story.append(make_bullets(items, ordered=False))
            continue

        if re.match(r"^\d+\. ", stripped):
            items = []
            while i < len(lines) and re.match(r"^\d+\. ", lines[i].strip()):
                items.append(re.sub(r"^\d+\. ", "", lines[i].strip()))
                i += 1
            story.append(make_bullets(items, ordered=True))
            continue

        # Join ordinary Markdown lines into one paragraph.
        para_lines = [stripped]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if (
                not nxt
                or nxt.startswith("#")
                or nxt.startswith("|")
                or re.match(r"^[-*] ", nxt)
                or re.match(r"^\d+\. ", nxt)
            ):
                break
            para_lines.append(nxt)
            i += 1
        text = " ".join(para_lines)

        if text.startswith("**Deep-research"):
            story.append(Paragraph(rich(text), subtitle_style))
        elif text.startswith("**Audience:") or text.startswith("**Research"):
            story.append(Paragraph(rich(text), meta_style))
        elif text.startswith("Yes—") or text.startswith("Yes-"):
            callout = Table(
                [[Paragraph(rich(text), callout_style)]],
                colWidths=[CONTENT_W],
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), PALE_TEAL),
                        ("BOX", (0, 0), (-1, -1), 0.8, TEAL),
                        ("LEFTPADDING", (0, 0), (-1, -1), 10),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                        ("TOPPADDING", (0, 0), (-1, -1), 8),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ]
                ),
            )
            story.append(callout)
            story.append(Spacer(1, 5))
        else:
            story.append(Paragraph(rich(text), ref_style if in_sources else body_style))

    return story


class ReportDocTemplate(BaseDocTemplate):
    def __init__(self, filename):
        super().__init__(
            filename,
            pagesize=PAGE,
            leftMargin=MARGIN_X,
            rightMargin=MARGIN_X,
            topMargin=MARGIN_TOP,
            bottomMargin=MARGIN_BOTTOM,
            title="Online Learning and Self-Adaptation in Reprogrammable-Hardware Applications",
            author="OpenAI Codex",
            subject="Deep research assessment of R/S programmable-hardware application classes",
        )
        frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        )
        self.addPageTemplates(PageTemplate(id="main", frames=[frame], onPage=self.on_page))

    @staticmethod
    def on_page(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(GRID)
        canvas.setLineWidth(0.45)
        canvas.line(MARGIN_X, PAGE_H - 10 * mm, PAGE_W - MARGIN_X, PAGE_H - 10 * mm)
        canvas.setFont("Helvetica", 7.2)
        canvas.setFillColor(MUTED)
        canvas.drawString(
            MARGIN_X,
            PAGE_H - 7.2 * mm,
            "ONLINE LEARNING + REPROGRAMMABLE HARDWARE  |  DEEP RESEARCH  |  8 SEP 2026",
        )
        canvas.drawRightString(
            PAGE_W - MARGIN_X,
            7.2 * mm,
            f"Page {doc.page}",
        )
        canvas.setFillColor(TEAL)
        canvas.rect(MARGIN_X, 6.2 * mm, 18 * mm, 1.1 * mm, fill=1, stroke=0)
        canvas.restoreState()


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    story = parse_markdown(SOURCE)
    doc = ReportDocTemplate(str(OUTPUT))
    doc.build(story)
    print(OUTPUT)


if __name__ == "__main__":
    main()
