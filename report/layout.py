"""Publication layout primitives for the measured Assignment 2 report."""

import html
import math
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Flowable, Image, Paragraph, Spacer, Table, TableStyle

W, H = A4
MARGIN = 46
WIDTH = W - 2 * MARGIN
NAVY = colors.HexColor("#173B4B")
AMBER = colors.HexColor("#95631C")
INK = colors.HexColor("#202A31")
PALE = colors.HexColor("#F3F5F5")
LINE = colors.HexColor("#D5DEE3")

for alias, file, fallback in [("Body", "Georgia.ttf", "Times-Roman"), ("BodyB", "Georgia Bold.ttf", "Times-Bold"),
                             ("Sans", "Arial.ttf", "Helvetica"), ("SansB", "Arial Bold.ttf", "Helvetica-Bold")]:
    font = Path("/System/Library/Fonts/Supplemental") / file
    pdfmetrics.registerFont(TTFont(alias, str(font)) if font.exists() else pdfmetrics.Font(alias, fallback, "WinAnsiEncoding"))

pdfmetrics.registerFontFamily("Body", normal="Body", bold="BodyB", italic="Body", boldItalic="BodyB")
pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="SansB", italic="Sans", boldItalic="SansB")


def styles(scale=1):
    return {
        "body": ParagraphStyle("body", fontName="Body", fontSize=10 * scale, leading=14 * scale, textColor=INK, spaceAfter=8 * scale),
        "h": ParagraphStyle("heading", fontName="SansB", fontSize=10.6 * scale, leading=14 * scale, textColor=NAVY, spaceBefore=5 * scale, spaceAfter=6 * scale),
        "title": ParagraphStyle("title", fontName="BodyB", fontSize=20 * scale, leading=25 * scale, spaceAfter=13 * scale),
        "section": ParagraphStyle("section", fontName="SansB", fontSize=8.4 * scale, leading=11 * scale, textColor=AMBER, spaceAfter=6 * scale),
        "cell": ParagraphStyle("cell", fontName="Sans", fontSize=8.6 * scale, leading=11.2 * scale, textColor=INK),
        "thead": ParagraphStyle("thead", fontName="SansB", fontSize=8.6 * scale, leading=11.2 * scale, textColor=colors.white),
        "caption": ParagraphStyle("caption", fontName="Sans", fontSize=8.5 * scale, leading=11 * scale, textColor=colors.HexColor("#596670"), spaceAfter=9 * scale),
        "code": ParagraphStyle("code", fontName="Courier", fontSize=7.8 * scale, leading=10.3 * scale, textColor=NAVY),
        "ref": ParagraphStyle("ref", fontName="Body", fontSize=9 * scale, leading=12.5 * scale, spaceAfter=8 * scale),
    }


def esc(text):
    return html.escape(str(text)).replace("\n", "<br/>")


class Diagram(Flowable):
    """Two readable lanes with explicit dependency arrows and feedback."""

    def __init__(self, kind):
        super().__init__()
        self.kind, self.width, self.height = kind, WIDTH, 256

    def draw(self):
        c = self.canv
        gap, boxw = 15, (WIDTH - 45) / 4

        def arrow(a, b):
            c.setStrokeColor(AMBER)
            c.setLineWidth(.9)
            c.line(*a, *b)
            angle = math.atan2(b[1] - a[1], b[0] - a[0])
            for offset in [-.55, .55]:
                c.line(*b, b[0] - 4 * math.cos(angle + offset), b[1] - 4 * math.sin(angle + offset))

        def box(i, y, title, sub):
            x = i * (boxw + gap)
            c.setFillColor(PALE)
            c.setStrokeColor(LINE)
            c.roundRect(x, y, boxw, 63, 4, stroke=1, fill=1)
            for text, font, size, top in [(title, "SansB", 9, y + 54), (sub, "Sans", 7.8, y + 22)]:
                style = ParagraphStyle("diagram", fontName=font, fontSize=size, leading=size + 2, alignment=1, textColor=NAVY)
                p = Paragraph(esc(text), style)
                _, height = p.wrap(boxw - 12, 100)
                p.drawOn(c, x + 6, top - height)

        if self.kind == "pipeline":
            top = [("Ingest + validate", "Raw CSV / SHA-256"), ("Preprocess + features", "Time policy / past lags"),
                   ("Train + evaluate", "10 runs x 3 folds"), ("Freeze + register", "October / MLflow")]
            bottom = [("Frozen model", "Version + policy"), ("Locked evaluation", "November-December"),
                      ("Replay API", "Validated history"), ("Monitor + review", "Metrics / audit")]
            label1, label2 = "A  DEVELOPMENT AND MODEL FREEZE", "B  EVALUATION AND DEMONSTRATION DEPLOYMENT"
        else:
            top = [("Prepare", "Validate / features"), ("Develop", "10 configurations"),
                   ("Sensitivity", "Literal-time study"), ("Calibrate", "Freeze + register")]
            bottom = [("Confirm", "Explicit test access"), ("Final evaluation", "Frozen parameters"),
                      ("Export", "Tables + figures"), ("Verify", "Registry + serving")]
            label1, label2 = "RESEARCH DAG  /  MANUAL TRIGGER", "DEFAULT CONFIRMATION = FALSE  /  FINAL TASKS SKIP"
        c.setFont("SansB", 8.4)
        c.setFillColor(AMBER)
        c.drawString(0, 244, label1)
        c.drawString(0, 117, label2)
        for i, (title, sub) in enumerate(top):
            box(i, 166, title, sub)
        for i, (title, sub) in enumerate(bottom):
            box(i, 35, title, sub)
        for y in [197, 66]:
            for i in range(3):
                arrow((i * (boxw + gap) + boxw, y), ((i + 1) * (boxw + gap), y))
        x = 3 * (boxw + gap) + boxw / 2
        c.setStrokeColor(AMBER)
        c.line(x, 166, x, 140)
        # Route the row transition in the left gutter, clear of the second heading.
        c.line(x, 140, -10, 140)
        c.line(-10, 140, -10, 66)
        arrow((-10, 66), (0, 66))
        c.setFillColor(NAVY)
        c.setFont("Sans", 8)
        c.drawString(0, 10, "Artifacts and compact manifests cross stages; no DataFrames are passed through Airflow XCom.")


def blocks(page, scale=1):
    st = styles(scale)
    flows = [Paragraph(esc(page["section"].upper()), st["section"]), Paragraph(esc(page["title"]), st["title"])]
    for block in page["blocks"]:
        kind = block[0]
        if kind in {"p", "h", "caption", "ref"}:
            flows.append(Paragraph(esc(block[1]), st["body" if kind == "p" else kind]))
        elif kind == "link":
            url = html.escape(block[1], quote=True)
            flows.append(Paragraph(f'<a href="{url}" color="#173B4B">{url}</a>', st["caption"]))
        elif kind == "table":
            headers, rows, ratios = block[1:]
            data = [[Paragraph(esc(v), st["thead"]) for v in headers]]
            data += [[Paragraph(esc(v), st["cell"]) for v in row] for row in rows]
            table = Table(data, colWidths=[WIDTH * x for x in ratios], repeatRows=1)
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), NAVY), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 6 * scale),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6 * scale), ("TOPPADDING", (0, 0), (-1, -1), 5.5 * scale),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5.5 * scale), ("LINEBELOW", (0, 0), (-1, -1), .35, LINE)]))
            flows += [table, Spacer(1, 9 * scale)]
        elif kind in {"code", "note"}:
            text = esc(block[1]).replace(" ", "&#160;") if kind == "code" else esc(block[1])
            table = Table([[Paragraph(text, st["code" if kind == "code" else "caption"])]], colWidths=[WIDTH])
            table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), PALE), ("BOX", (0, 0), (-1, -1), .4, LINE),
                ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 9), ("BOTTOMPADDING", (0, 0), (-1, -1), 9)]))
            flows += [table, Spacer(1, 9 * scale)]
        elif kind == "diagram":
            flows.append(Diagram(block[1]))
        elif kind == "figure":
            from PIL import Image as PILImage
            with PILImage.open(block[1]) as im:
                width, height = im.size
            flows += [Image(str(block[1]), width=WIDTH, height=WIDTH * height / width), Spacer(1, 8)]
        else:
            raise ValueError(kind)
    return flows
