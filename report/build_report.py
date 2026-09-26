"""Build the English report from completed, measured pipeline outputs."""

import argparse
import json
import os
from datetime import date
from pathlib import Path

from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import BaseDocTemplate, Frame, PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle

from content import AUTHOR, STUDENT, TITLE, pages
from layout import H, INK, LINE, MARGIN, NAVY, W, WIDTH, blocks, esc, styles

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, default=Path(os.getenv("OUTPUT_DIR", ROOT / "artifacts/run")))
    parser.add_argument("--output", type=Path, default=ROOT / "report/DDM501_Assignment2_25MS13293_DoQuangHiep.pdf")
    args = parser.parse_args()
    abstract, content = pages(args.evidence)
    st = styles()
    title = ParagraphStyle("cover", fontName="BodyB", fontSize=27, leading=34, textColor=INK, spaceAfter=15)
    label = ParagraphStyle("cover-label", fontName="Sans", fontSize=10, leading=15, textColor=NAVY)
    course = ParagraphStyle("course", fontName="SansB", fontSize=10, leading=15, textColor=NAVY, spaceAfter=23)
    story = [Spacer(1, 16), Paragraph("DDM501<br/>AI in DevOps, DataOps, MLOps", course),
             Paragraph(esc(TITLE), title), Paragraph("ML Pipeline Design &amp; MLOps Analysis", st["title"]), Spacer(1, 10)]
    metadata = [("Assignment", "Individual Assignment 2"), ("Author", AUTHOR), ("Student ID", STUDENT),
                ("Document date", date.today().strftime("%d %B %Y")),
                ("Evidence status", "Measured historical experiments / Docker project")]
    table = Table([[Paragraph(a, label), Paragraph(b, label)] for a, b in metadata], colWidths=[110, WIDTH - 110])
    table.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 6), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story += [table, Spacer(1, 21), Paragraph("Executive summary", st["h"]), Paragraph(esc(abstract), st["body"]),
              Spacer(1, 10), Paragraph("Reproducible source, licensed data, recorded results and assessor run instructions accompany this report.", st["caption"]), PageBreak()]
    layout = []
    for index, page in enumerate(content, 2):
        scale = 1.0
        while True:
            flows = blocks(page, scale)
            height = sum(f.wrap(WIDTH, 2000)[1] + f.getSpaceBefore() + f.getSpaceAfter() for f in flows)
            if height <= H - 114:
                break
            scale -= .01
            if scale < .89:
                raise ValueError(f"Page {index} too tall: {height:.1f}; split its content")
        layout.append({"page": index, "title": page["title"], "scale": round(scale, 2), "height": round(height, 1)})
        story.extend(flows)
        if index < len(content) + 1:
            story.append(PageBreak())

    def decorate(canvas, doc):
        canvas.setStrokeColor(LINE)
        canvas.line(MARGIN, 35, W - MARGIN, 35)
        canvas.setFont("Sans", 7.7)
        canvas.setFillColor(NAVY)
        canvas.drawString(MARGIN, 22, "Steel energy forecasting | Measured pipeline and MLOps analysis")
        canvas.drawRightString(W - MARGIN, 22, f"{doc.page:02d}")
        if doc.page > 1:
            canvas.drawString(MARGIN, H - 28, "DDM501 | INDIVIDUAL ASSIGNMENT 2")
            canvas.drawRightString(W - MARGIN, H - 28, f"{AUTHOR} / {STUDENT}")
            canvas.line(MARGIN, H - 36, W - MARGIN, H - 36)
        bookmark = f"page-{doc.page}"
        canvas.bookmarkPage(bookmark)
        canvas.addOutlineEntry("Executive summary" if doc.page == 1 else content[doc.page - 2]["title"], bookmark, level=0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(str(args.output), pagesize=(W, H), title=TITLE, author=AUTHOR,
                          subject="DDM501 Individual Assignment 2 - ML Pipeline Design and MLOps Analysis",
                          leftMargin=MARGIN, rightMargin=MARGIN, topMargin=54, bottomMargin=48)
    frame = Frame(MARGIN, 48, WIDTH, H - 102, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates(PageTemplate(id="report", frames=[frame], onPage=decorate))
    doc.build(story)
    (ROOT / "report/layout.json").write_text(json.dumps(layout, indent=2))
    text = [f"# {TITLE}", "ML Pipeline Design & MLOps Analysis", f"{AUTHOR} | {STUDENT}", "## Executive summary", abstract]
    for page in content:
        text += [f'## {page["section"]}', f'### {page["title"]}']
        for block in page["blocks"]:
            kind = block[0]
            if kind in {"p", "caption", "ref", "note"}:
                text.append(str(block[1]))
            elif kind == "link":
                text.append(f"[Public source repository]({block[1]})")
            elif kind == "h":
                text.append("#### " + block[1])
            elif kind == "code":
                text.append("```text\n" + block[1] + "\n```")
            elif kind == "table":
                rows = [block[1], ["---"] * len(block[1])] + block[2]
                text.append("\n".join("| " + " | ".join(str(c).replace("\n", "<br>").replace("|", "\\|") for c in row) + " |" for row in rows))
            elif kind == "figure":
                text.append(f"Figure: {Path(block[1]).name}")
            elif kind == "diagram":
                text.append("Diagram: " + block[1] + "; see the fully rendered PDF.")
    (ROOT / "report/report.md").write_text("\n\n".join(text) + "\n")
    print(json.dumps(layout, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
