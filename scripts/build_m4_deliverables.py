"""Build TerraWatch Milestone 4 Submission PDF Report & Presentation Slides
from reports/Milestone4_Report.md and reports/figures/m4_*.png.
"""
from __future__ import annotations

import argparse
import html as html_lib
import json
import textwrap
from io import BytesIO
from pathlib import Path

import markdown
import pymupdf
from bs4 import BeautifulSoup, NavigableString, Tag
from pypdf import PdfWriter
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

REPO = Path(__file__).resolve().parents[1]

FONTS_DIR = Path(r"C:\Windows\Fonts")
FONT_REG = "Helvetica"
FONT_BOLD = "Helvetica-Bold"
FONT_ITAL = "Helvetica-Oblique"
FONT_BOLD_ITAL = "Helvetica-BoldOblique"
FONT_MONO = "Courier"
FONT_MONO_BOLD = "Courier-Bold"


def _register_fonts() -> None:
    global FONT_REG, FONT_BOLD, FONT_ITAL, FONT_BOLD_ITAL, FONT_MONO, FONT_MONO_BOLD
    mapping = {
        "SegoeUI": FONTS_DIR / "segoeui.ttf",
        "SegoeUI-Bold": FONTS_DIR / "segoeuib.ttf",
        "SegoeUI-Italic": FONTS_DIR / "segoeuii.ttf",
        "SegoeUI-BoldItalic": FONTS_DIR / "segoeuiz.ttf",
        "Consolas": FONTS_DIR / "consola.ttf",
        "Consolas-Bold": FONTS_DIR / "consolab.ttf",
    }
    if all(p.is_file() for p in mapping.values()):
        for name, path in mapping.items():
            pdfmetrics.registerFont(TTFont(name, str(path)))
        pdfmetrics.registerFontFamily(
            "SegoeUI",
            normal="SegoeUI",
            bold="SegoeUI-Bold",
            italic="SegoeUI-Italic",
            boldItalic="SegoeUI-BoldItalic",
        )
        FONT_REG = "SegoeUI"
        FONT_BOLD = "SegoeUI-Bold"
        FONT_ITAL = "SegoeUI-Italic"
        FONT_BOLD_ITAL = "SegoeUI-BoldItalic"
        FONT_MONO = "Consolas"
        FONT_MONO_BOLD = "Consolas-Bold"


_register_fonts()

NAVY = colors.HexColor("#102630")
TEAL = colors.HexColor("#1B6B74")
BLUE = colors.HexColor("#1F5FAE")
INK = colors.HexColor("#1E293B")
MUTED = colors.HexColor("#64748B")
CODE_FG = "#184E5E"
CODE_BG = "#EDF4F6"
RULE_COLOR = colors.HexColor("#D2DCE1")
ZEBRA_BG = colors.HexColor("#F7FAFB")
HIGHLIGHT_BG = colors.HexColor("#E8F4F3")

STYLE_SECTION = ParagraphStyle(
    "M4Section", fontName=FONT_BOLD, fontSize=13.2, leading=17.5,
    textColor=NAVY, spaceBefore=14, spaceAfter=3, keepWithNext=True,
)
STYLE_SUBSECTION = ParagraphStyle(
    "M4Subsection", fontName=FONT_BOLD, fontSize=10.6, leading=14.5,
    textColor=TEAL, spaceBefore=10, spaceAfter=4, keepWithNext=True,
)
STYLE_BODY = ParagraphStyle(
    "M4Body", fontName=FONT_REG, fontSize=9.6, leading=14.4,
    textColor=INK, alignment=TA_JUSTIFY, spaceAfter=6.5,
)
STYLE_BULLET = ParagraphStyle(
    "M4Bullet", fontName=FONT_REG, fontSize=9.5, leading=14.0,
    textColor=INK, leftIndent=16, firstLineIndent=-10, spaceAfter=4.5,
)
STYLE_CAPTION = ParagraphStyle(
    "M4Caption", fontName=FONT_ITAL, fontSize=8.5, leading=12.0,
    textColor=MUTED, alignment=TA_LEFT, spaceBefore=2, spaceAfter=8,
)
STYLE_TH = ParagraphStyle(
    "M4TH", fontName=FONT_BOLD, fontSize=7.7, leading=10.2,
    textColor=colors.white, alignment=TA_LEFT,
)
STYLE_TH_R = ParagraphStyle("M4THR", parent=STYLE_TH, alignment=TA_RIGHT)
STYLE_TD = ParagraphStyle(
    "M4TD", fontName=FONT_REG, fontSize=7.8, leading=10.6,
    textColor=INK, alignment=TA_LEFT,
)
STYLE_TD_R = ParagraphStyle("M4TDR", parent=STYLE_TD, alignment=TA_RIGHT)


def _inline_markup(node, in_table: bool = False) -> str:
    code_sz = "7.0" if in_table else "8.3"
    parts = []
    for child in node.children:
        if isinstance(child, NavigableString):
            parts.append(html_lib.escape(str(child)))
        elif isinstance(child, Tag):
            inner = _inline_markup(child, in_table=in_table)
            if child.name in ("strong", "b"):
                parts.append(f"<b>{inner}</b>")
            elif child.name in ("em", "i"):
                parts.append(f"<i>{inner}</i>")
            elif child.name == "code":
                if in_table:
                    parts.append(f'<font name="{FONT_MONO}" size="{code_sz}" color="{CODE_FG}">{inner}</font>')
                else:
                    parts.append(
                        f'<font name="{FONT_MONO}" size="{code_sz}" color="{CODE_FG}" backColor="{CODE_BG}">{inner}</font>'
                    )
            else:
                parts.append(inner)
    return "".join(parts)


def _build_table(node: Tag, avail_width: float) -> Table:
    rows = node.find_all("tr")
    raw_rows = [r.find_all(["th", "td"], recursive=False) for r in rows]
    ncols = len(raw_rows[0])
    headers = [c.get_text(" ", strip=True) for c in raw_rows[0]]
    h0 = headers[0] if headers else ""

    if ncols == 10 and "Chg mIoU" in headers:
        ratios = [0.192, 0.128, 0.194, 0.036, 0.074, 0.074, 0.074, 0.072, 0.074, 0.082]
    elif ncols == 10 and "BLEU-4" in headers:
        ratios = [0.180, 0.240, 0.052, 0.036, 0.074, 0.076, 0.074, 0.064, 0.064, 0.140]
    elif ncols == 8 and "Precision" in headers:
        ratios = [0.275, 0.115, 0.095, 0.095, 0.090, 0.105, 0.120, 0.105]
    elif ncols == 7 and "Chg mIoU" in headers:
        ratios = [0.290, 0.190, 0.104, 0.104, 0.104, 0.104, 0.104]
    elif ncols == 5 and "Pair ID" in headers:
        ratios = [0.175, 0.105, 0.240, 0.110, 0.370]
    elif ncols == 5 and h0 == "Partition":
        ratios = [0.155, 0.185, 0.165, 0.125, 0.370]
    elif ncols == 5 and "Seeds Evaluated" in headers:
        ratios = [0.240, 0.120, 0.220, 0.215, 0.205]
    elif ncols == 4 and "Advantage (BIT − Siam)" in headers:
        ratios = [0.400, 0.200, 0.200, 0.200]
    elif ncols == 4 and h0 == "Compute Resource":
        ratios = [0.175, 0.230, 0.245, 0.350]
    else:
        lens = [max(len(c.get_text(" ", strip=True)) for c in col) for col in zip(*raw_rows)]
        weights = [max(4.0, min(32.0, float(l))) for l in lens]
        tot = sum(weights)
        ratios = [w / tot for w in weights]

    right_cols = {
        ci for ci, th in enumerate(raw_rows[0])
        if "right" in (th.get("style") or "").lower()
    }

    cells = []
    highlight_rows = []
    for ri, row in enumerate(raw_rows):
        row_text = " ".join(c.get_text(" ", strip=True) for c in row)
        if ri > 0 and (
            "(Selected)" in row_text or "(Sel.)" in row_text
            or "Mask-Only Checker" in row_text or "Gaussian Blur" in row_text or "SECOND-CC" in row_text
        ):
            highlight_rows.append(ri)
        rendered_row = []
        for ci, cell in enumerate(row):
            is_r = ci in right_cols
            st = (STYLE_TH_R if is_r else STYLE_TH) if ri == 0 else (STYLE_TD_R if is_r else STYLE_TD)
            rendered_row.append(Paragraph(_inline_markup(cell, in_table=True), st))
        cells.append(rendered_row)

    tbl = Table(cells, colWidths=[r * avail_width for r in ratios], hAlign="LEFT", repeatRows=1)
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4.5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4.5),
        ("TOPPADDING", (0, 0), (-1, 0), 5.5),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 5.5),
        ("TOPPADDING", (0, 1), (-1, -1), 4.0),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 4.0),
        ("LINEBELOW", (0, 0), (-1, 0), 1.0, TEAL),
        ("LINEBELOW", (0, 1), (-1, -2), 0.35, RULE_COLOR),
        ("LINEBELOW", (0, -1), (-1, -1), 0.9, NAVY),
    ]
    for ri in range(1, len(raw_rows)):
        if ri in highlight_rows:
            style_cmds.append(("BACKGROUND", (0, ri), (-1, ri), HIGHLIGHT_BG))
        elif ri % 2 == 0:
            style_cmds.append(("BACKGROUND", (0, ri), (-1, ri), ZEBRA_BG))
    tbl.setStyle(TableStyle(style_cmds))
    return tbl


class ReportCanvas(canvas.Canvas):
    def __init__(self, *args, cover_offset: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []
        self._cover_offset = cover_offset

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._saved_page_states) + self._cover_offset
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._draw_chrome(total)
            super().showPage()
        super().save()

    def _draw_chrome(self, total: int):
        page_num = self._pageNumber + self._cover_offset
        w, h = A4
        self.saveState()
        self.setFont(FONT_BOLD, 8.0)
        self.setFillColor(NAVY)
        self.drawString(46, h - 30, "TerraWatch  |  Group 7 (BSDA4001)")
        self.setFont(FONT_REG, 8.0)
        self.setFillColor(MUTED)
        self.drawRightString(w - 46, h - 30, "Milestone 4: Model Training, Optimization & Evidence Review")
        self.setStrokeColor(RULE_COLOR)
        self.setLineWidth(0.55)
        self.line(46, h - 36, w - 46, h - 36)
        self.line(46, 36, w - 46, 36)
        self.setFont(FONT_REG, 8.0)
        self.setFillColor(MUTED)
        self.drawString(46, 24, "IIT Madras BS Data Science & AI Lab")
        self.drawRightString(w - 46, 24, f"Page {page_num} of {total}")
        self.restoreState()


def build_cover(cover_template: Path | None) -> BytesIO:
    if cover_template and cover_template.is_file():
        doc = pymupdf.open(cover_template)
        page = doc[0]
        page.add_redact_annot(pymupdf.Rect(60, 270, 535, 293), fill=(1, 1, 1))
        for rect in page.search_for("October 2026"):
            page.add_redact_annot(rect, fill=(1, 1, 1))
        page.apply_redactions(images=0, graphics=0)
        overlay = BytesIO()
        c = canvas.Canvas(overlay, pagesize=A4)
        c.setFont(FONT_BOLD, 9.2)
        c.setFillColor(BLUE)
        c.drawCentredString(A4[0] / 2, A4[1] - 283, "MILESTONE 4: MODEL TRAINING, OPTIMIZATION & EVIDENCE REVIEW")
        c.setFont(FONT_REG, 8.8)
        c.setFillColor(MUTED)
        c.drawCentredString(A4[0] / 2, A4[1] - 746, "November 2026")
        c.save()
        with pymupdf.open(stream=overlay.getvalue(), filetype="pdf") as layer:
            page.show_pdf_page(page.rect, layer, 0)
        out = BytesIO(doc.tobytes(garbage=4, deflate=True))
        doc.close()
        return out

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    c.setFillColor(NAVY)
    c.setFont(FONT_BOLD, 24)
    c.drawCentredString(A4[0] / 2, A4[1] - 220, "TERRAWATCH")
    c.setFillColor(BLUE)
    c.setFont(FONT_BOLD, 11)
    c.drawCentredString(A4[0] / 2, A4[1] - 260, "MILESTONE 4: MODEL TRAINING, OPTIMIZATION & EVIDENCE REVIEW")
    c.setFillColor(INK)
    c.setFont(FONT_REG, 10.5)
    c.drawCentredString(A4[0] / 2, A4[1] - 320, "Group 7  |  IIT Madras BS Data Science & AI Lab (BSDA4001)")
    c.drawCentredString(A4[0] / 2, A4[1] - 345, "Priyanshu Agarwal  |  Sarthak Saini  |  Sujal Raj Pradhan  |  Shoaib")
    c.setFillColor(MUTED)
    c.setFont(FONT_REG, 9.5)
    c.drawCentredString(A4[0] / 2, 100, "November 2026")
    c.save()
    buf.seek(0)
    return buf


def build_report_pdf(md_path: Path, out_pdf: Path, cover_template: Path | None = None) -> Path:
    source = md_path.read_text(encoding="utf-8-sig")
    marker = "## 1. Introduction"
    body = marker + source.split(marker, 1)[1] if marker in source else source
    html = markdown.markdown(body.replace("<!-- PAGEBREAK -->", '<div class="pagebreak"></div>'), extensions=["tables"])
    soup = BeautifulSoup(html, "html.parser")

    stream = BytesIO()
    doc = SimpleDocTemplate(
        stream, pagesize=A4, leftMargin=46, rightMargin=46, topMargin=52, bottomMargin=48,
        title="TerraWatch - Milestone 4: Model Training, Optimization & Evidence Review",
        author="Group 7 - IIT Madras BS DSAI Lab",
    )

    story = []
    fig_idx = 0
    for el in soup.children:
        if isinstance(el, NavigableString):
            continue
        if el.name == "div":
            story.append(PageBreak())
        elif el.name == "hr":
            story.append(Spacer(1, 2))
        elif el.name == "h2":
            story.append(KeepTogether([
                Paragraph(_inline_markup(el), STYLE_SECTION),
                HRFlowable(width="100%", thickness=0.85, color=TEAL, spaceBefore=0, spaceAfter=6),
            ]))
        elif el.name == "h3":
            story.append(Paragraph(_inline_markup(el), STYLE_SUBSECTION))
        elif el.name == "p":
            img = el.find("img")
            if img:
                img_path = (md_path.parent / img["src"]).resolve()
                if img_path.is_file():
                    fig_idx += 1
                    pic = Image(str(img_path))
                    max_w, max_h = doc.width, 255
                    ratio = pic.imageHeight / max(pic.imageWidth, 1)
                    w = max_w
                    h = w * ratio
                    if h > max_h:
                        h = max_h
                        w = h / ratio
                    pic.drawWidth, pic.drawHeight = w, h
                    alt_text = img.get("alt") or f"Figure {fig_idx}"
                    story.append(KeepTogether([
                        Spacer(1, 4),
                        pic,
                        Paragraph(f"Figure {fig_idx}: {html_lib.escape(alt_text)}", STYLE_CAPTION),
                    ]))
            else:
                story.append(Paragraph(_inline_markup(el), STYLE_BODY))
        elif el.name == "table":
            story.extend([Spacer(1, 4), _build_table(el, doc.width), Spacer(1, 8)])
        elif el.name in ("ol", "ul"):
            for i, li in enumerate(el.find_all("li", recursive=False), 1):
                prefix = f"<b>{i}.</b> " if el.name == "ol" else '<font color="#1B6B74"><b>•</b></font> '
                story.append(Paragraph(prefix + _inline_markup(li), STYLE_BULLET))
            story.append(Spacer(1, 3))

    doc.build(story, canvasmaker=lambda *a, **kw: ReportCanvas(*a, cover_offset=1, **kw))

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    writer = PdfWriter()
    writer.append(build_cover(cover_template))
    stream.seek(0)
    writer.append(stream)
    writer.add_metadata({
        "/Title": "TerraWatch - Milestone 4: Model Training, Optimization & Evidence Review",
        "/Author": "IIT Madras Group 7",
    })
    with out_pdf.open("wb") as f:
        writer.write(f)
    print("BUILT_REPORT_PDF:", out_pdf)
    return out_pdf


def build_slides_pdf(figs_dir: Path, out_pdf: Path) -> Path:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(out_pdf), pagesize=(960, 540))
    c.setTitle("TerraWatch Milestone 4 Review")
    navy, cream, lime, muted = "#0C1F29", "#F7F5EF", "#D9ED85", "#607780"

    def txt(x, y, s, size, color, font=FONT_REG):
        c.setFillColor(colors.HexColor(color))
        c.setFont(font, size)
        c.drawString(x, y, s)

    def frame(title, dark, num):
        bg, fg = (navy, cream) if dark else (cream, navy)
        c.setFillColor(colors.HexColor(bg))
        c.rect(0, 0, 960, 540, fill=1, stroke=0)
        txt(44, 502, "TERRAWATCH   /   MILESTONE 04", 10.5, lime if dark else muted, FONT_BOLD)
        txt(44, 446, title, 26, fg, FONT_BOLD)
        txt(44, 24, "IIT MADRAS  /  BSDA4001  /  GROUP 7", 9.5, lime if dark else muted)
        txt(892, 24, f"{num:02d}", 9.5, lime if dark else muted)
        return fg

    def bullets(title, items, dark, num):
        fg = frame(title, dark, num)
        y = 382
        for head, body in items:
            txt(44, y, head, 16.5, lime if dark else navy, FONT_BOLD)
            y -= 24
            for line in textwrap.wrap(body, 90):
                txt(44, y, line, 14.0, fg)
                y -= 20
            y -= 15
        c.showPage()

    def fig_slide(title, img_path: Path, caption: str, dark: bool, num: int):
        fg = frame(title, dark, num)
        if img_path.is_file():
            c.drawImage(str(img_path), 44, 100, width=872, height=322, mask="auto", preserveAspectRatio=True, anchor="c")
        y = 76
        for line in textwrap.wrap(caption, 98):
            txt(44, y, line, 13.0, fg)
            y -= 18
        c.showPage()

    fg = frame("Model Training, Optimization & Evidence Review", True, 1)
    txt(44, 344, "TERRAWATCH", 58, lime, FONT_BOLD)
    txt(46, 260, "Evidence-Grounded Satellite Change Detection, Captioning & Verification", 20.5, fg)
    txt(46, 152, "Milestone 4 Review  |  November 2026", 15, fg)
    txt(46, 112, "Priyanshu Agarwal  /  Sarthak Saini  /  Sujal Raj Pradhan  /  Shoaib", 14, fg)
    c.showPage()

    bullets("Full-Data Contract & System Pipeline", [
        ("Governed LEVIR-MCI Cohort (9,544 pairs)", "Every experiment trains on 100% of the strict split: 6,444 train (32,220 captions, 388-word train-only vocab), 1,242 validation (selection only), and 1,858 held-out test pairs with SHA-256 verification."),
        ("Four-Stage Modular Architecture", "BIT-ResNet18 3-class detector -> RSCaMa paired captioner (10x faster via exact Triton selective-scan kernel) -> OpenCV structural witness (32px contextual crops) -> Frozen Qwen3-VL-4B-Instruct referee + safety gate."),
        ("Key Engineering Upgrades", "Contextual 32px crop padding guarantees witness compatibility, explicit count gating separates articles ('a road') from numeric counts ('two houses'), and full provenance flows into the Evidence Review UI."),
    ], False, 2)

    det_desc = (
        "Across 24 full-data runs, BIT-ResNet18 (D03, LR 3e-4) reached 74.54% val Change mIoU "
        "(3-seed: 74.23% +/- 0.28%, Bin F1 86.25%, FA@50px 6.80%) and 75.93% on held-out Test (1,858 pairs)."
    )
    cap_desc = (
        "Winner C06 (RSCaMa Frozen CLIP, AdamW cosine) achieved 0.6358 Val BLEU-4 (3-seed: 0.6297 +/- 0.0183), "
        "3.32% val NC FA, and 0.6128 BLEU-4 / 0.7326 ROUGE-L / 3.57% NC FA on held-out Test (1,858 pairs)."
    )

    fig_slide("Detector Convergence: Full-Data Learning & Validation Curves", figs_dir / "m4_detector_learning_curves.png", det_desc, True, 3)
    fig_slide("Detector Controlled Matrix & 5-Architecture Comparison", figs_dir / "m4_detector_comparison_bar.png", "All 24 arms trained on 6,444 strict LEVIR-MCI pairs and ranked on 1,242 val pairs across LR, WD, batch size, schedule, loss, freeze policy, augmentation, and CNN/Transformer architectures.", False, 4)
    fig_slide("Corruption Robustness, False-Alarm Curve & Transfer", figs_dir / "m4_robustness_and_false_alarms.png", "Why D03 (BIT-ResNet18) is selected over A03 (Siam-R18-UNet): BIT beats Siam-UNet by +12.4% at Blur r=1.0 (70.9% vs 58.6%), +30.4% at Blur r=2.0 (38.2% vs 7.9%), and +5.5% on SECOND-CC Test (23.2% vs 17.7%).", True, 5)
    fig_slide("Full-Data RSCaMa Captioner Training & Ablations", figs_dir / "m4_captioner_learning_curves.png", cap_desc, False, 6)
    fig_slide("Factual Verification & Disagreement-Triggered Reinspection", figs_dir / "m4_verification_comparison.png", "120 stratified val pairs (147 claims): Full-data BIT Mask-Only achieves 91.38% precision and 63.0% contradiction catch vs Direct Qwen3-VL (82.43%) and 4-View Reinspection (72.41%), guiding our M5 mask-first hybrid design.", True, 7)
    fig_slide("Authentic End-to-End Pipeline Traces (Full-Data Models)", figs_dir / "m4_e2e_traces.png", "Real satellite pairs demonstrating genuine model-accepted (building & road), model-rejected (hallucination), and model-deferred (explicit numeric count) decisions with SHA-256 checkpoint provenance.", False, 8)

    bullets("Interactive Evidence Review UI & Milestone 5 Readiness", [
        ("Complete Visual Provenance", "Displays T1, T2, predicted 3-class BIT mask, overlay + candidate box, zoomed 32px+ T1/T2 crops, Qwen3-VL probability bars, disagreement triggers, witness diagnostics, and checkpoint SHA-256 digests."),
        ("Immutable Model Outputs + Append-Only Audit Trail", "Reviewers filter claims, flag errors, record corrected wording/counts, and persist human overrides into SQLite without mutating original model decisions, with one-click JSON audit export."),
        ("Milestone 5 Roadmap", "Scale dual-rater expert audit (147 claims, Cohen's kappa), deploy the calibrated mask-first hybrid verifier, and package the complete inference and review pipeline for streamlined deployment."),
    ], True, 9)

    c.save()
    print("BUILT_SLIDES_PDF:", out_pdf)
    return out_pdf


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Milestone 4 PDF Report & Slide Deck.")
    ap.add_argument("--report-md", type=Path, default=REPO / "reports/Milestone4_Report.md")
    ap.add_argument("--figures-dir", type=Path, default=REPO / "reports/figures")
    ap.add_argument("--report-pdf", type=Path, default=REPO / "reports/DSAI Lab_Group 7_Milestone 4.pdf")
    ap.add_argument("--slides-pdf", type=Path, default=REPO / "presentation/DSAI Lab_Group 7_Milestone 4_Slides.pdf")
    ap.add_argument("--cover-template", type=Path, default=REPO / "POC/milestone1_v2/assets/original_cover.pdf")
    args = ap.parse_args()

    build_report_pdf(args.report_md, args.report_pdf, args.cover_template if args.cover_template.is_file() else None)
    build_slides_pdf(args.figures_dir, args.slides_pdf)


if __name__ == "__main__":
    main()
