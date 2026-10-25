"""Build the report and internal guide from their Markdown sources.

Run from any directory: python POC/milestone1_v2/build_report.py
No network requests or project-model dependencies are involved.
"""
from pathlib import Path
import html
import re

import markdown
from bs4 import BeautifulSoup, NavigableString
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output' / 'pdf'
OUT.mkdir(parents=True, exist_ok=True)
NAVY = colors.HexColor('#17354A')
TEAL = colors.HexColor('#197D85')
GRAY = colors.HexColor('#506273')


def fonts():
    base = Path('C:/Windows/Fonts')
    names = [('Studio', 'calibri.ttf'), ('Studio-Bold', 'calibrib.ttf'),
             ('Studio-Italic', 'calibrii.ttf'), ('Studio-BoldItalic', 'calibriz.ttf')]
    if all((base / filename).exists() for _, filename in names):
        for name, filename in names:
            pdfmetrics.registerFont(TTFont(name, str(base / filename)))
        pdfmetrics.registerFontFamily('Studio', normal='Studio', bold='Studio-Bold',
                                     italic='Studio-Italic', boldItalic='Studio-BoldItalic')
        return 'Studio', 'Studio-Bold'
    return 'Helvetica', 'Helvetica-Bold'


REG, BOLD = fonts()
STYLES = getSampleStyleSheet()
for name, size, leading, color, before, after, bold in [
    ('Text', 10.3, 13.5, '#263543', 0, 7, False),
    ('Title2', 21, 24, '#17354A', 0, 12, True),
    ('Section', 14, 17, '#17354A', 10, 8, True),
    ('Subsection', 11.3, 14, '#197D85', 7, 5, True),
    ('Cell', 8.1, 10.4, '#263543', 0, 0, False),
    ('CellHead', 8.1, 10.4, '#FFFFFF', 0, 0, True),
    ('Reference', 8.3, 10.7, '#263543', 0, 5, False),
]:
    STYLES.add(ParagraphStyle(name=name, fontName=BOLD if bold else REG,
        fontSize=size, leading=leading, textColor=colors.HexColor(color),
        spaceBefore=before, spaceAfter=after, alignment=TA_LEFT,
        splitLongWords=True, allowWidows=0, allowOrphans=0,
        keepWithNext=name in ('Title2', 'Section', 'Subsection')))


def clean(s):
    return s.replace('\u2011', '-').replace('\u2013', '-').replace('\u2014', ' - ')


def inline(node):
    if isinstance(node, NavigableString):
        text = html.escape(clean(str(node)))
        # Auto-link plain URLs without changing visible source text.
        return re.sub(r'https?://[^\s<>]+', lambda m: '<a href="' + m[0] + '" color="#197D85">' + m[0] + '</a>', text)
    body = ''.join(inline(c) for c in node.children)
    tags = {'strong': 'b', 'b': 'b', 'em': 'i', 'i': 'i', 'sup': 'super', 'sub': 'sub'}
    if node.name in tags:
        tag = tags[node.name]
        return f'<{tag}>{body}</{tag}>'
    if node.name == 'code':
        return f'<font name="Courier">{body}</font>'
    if node.name == 'br':
        return '<br/>'
    return body


def para(node, style='Text', prefix=''):
    return Paragraph(prefix + inline(node), STYLES[style])


def table(node, width):
    raw = node.find_all('tr')
    n = len(raw[0].find_all(['th', 'td'], recursive=False))
    headings = [c.get_text(' ', strip=True) for c in raw[0].find_all(['th', 'td'], recursive=False)]
    if n == 6:
        ratios = [0.13, 0.12, 0.12, 0.21, 0.18, 0.24]
    elif n == 4:
        ratios = [0.17, 0.26, 0.30, 0.27]
    elif n == 3 and headings[0] == 'Team member':
        ratios = [0.27, 0.46, 0.27]
    elif n == 3:
        ratios = [0.20, 0.44, 0.36]
    else:
        ratios = [0.32, 0.68] if n == 2 else [1/n]*n
    data = []
    for ri, row in enumerate(raw):
        cells = row.find_all(['th', 'td'], recursive=False)
        data.append([para(c, 'CellHead' if ri == 0 else 'Cell') for c in cells])
    result = Table(data, colWidths=[r*width for r in ratios], repeatRows=1, hAlign='LEFT')
    padding = 4 if headings[0] in ('Team member', 'Stakeholder') else 6
    result.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#F1F5F7'), colors.white]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), padding),
        ('BOTTOMPADDING', (0, 0), (-1, -1), padding),
        ('LINEBELOW', (0, 0), (-1, 0), 0.6, TEAL),
        ('LINEBELOW', (0, 1), (-1, -1), 0.25, colors.HexColor('#DAE3E8')),
    ]))
    return [Spacer(1, 3), result, Spacer(1, 8)]


def draw_page(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setFont(BOLD, 8)
    canvas.setFillColor(NAVY)
    canvas.drawString(doc.leftMargin, h-28, 'TERRAWATCH  /  GROUP 7')
    canvas.setFont(REG, 8)
    canvas.setFillColor(GRAY)
    canvas.drawRightString(w-doc.rightMargin, h-28, doc.short_label)
    canvas.setStrokeColor(TEAL)
    canvas.setLineWidth(0.7)
    canvas.line(doc.leftMargin, h-35, w-doc.rightMargin, h-35)
    canvas.setStrokeColor(colors.HexColor('#D9E2E7'))
    canvas.line(doc.leftMargin, 34, w-doc.rightMargin, 34)
    canvas.setFont(REG, 8)
    canvas.drawString(doc.leftMargin, 22, 'IIT Madras  |  DS & AI Lab  |  October 2026')
    canvas.drawRightString(w-doc.rightMargin, 22, str(doc.page))
    canvas.restoreState()


def build(source, destination, label):
    text = source.read_text(encoding='utf-8')
    text = text.replace('<!-- PAGEBREAK -->', '<div class="pagebreak"></div>')
    soup = BeautifulSoup(markdown.markdown(text, extensions=['tables', 'fenced_code']), 'html.parser')
    doc = SimpleDocTemplate(str(destination), pagesize=A4, leftMargin=42, rightMargin=42,
        topMargin=49, bottomMargin=45, title=label, author='TerraWatch - Group 7',
        subject='DS & AI Lab - Milestone 1', pageCompression=1)
    doc.short_label = label
    story = []
    in_refs = False
    for el in soup.children:
        if isinstance(el, NavigableString):
            continue
        if el.name == 'div' and 'pagebreak' in el.get('class', []):
            story.append(PageBreak())
        elif el.name in ('h1', 'h2', 'h3'):
            in_refs = el.get_text(strip=True) == 'References'
            story.append(para(el, {'h1': 'Title2', 'h2': 'Section', 'h3': 'Subsection'}[el.name]))
        elif el.name == 'p':
            story.append(para(el, 'Reference' if in_refs else 'Text'))
        elif el.name == 'table':
            story.extend(table(el, doc.width))
        elif el.name in ('ol', 'ul'):
            for i, li in enumerate(el.find_all('li', recursive=False), 1):
                prefix = f'{i}. ' if el.name == 'ol' else '- '
                story.append(para(li, 'Reference' if in_refs else 'Text', prefix))
        elif el.name == 'hr':
            story.append(HRFlowable(width='100%', color=TEAL))
    doc.build(story, onFirstPage=draw_page, onLaterPages=draw_page)
    return destination


if __name__ == '__main__':
    outputs = [
        build(ROOT/'Milestone1_Report.md', OUT/'TerraWatch_Milestone1_Revised.pdf', 'Milestone 1 - Revised Report'),
        build(ROOT/'Team_Defence_Guide.md', OUT/'TerraWatch_M1_Team_Defence_Guide.pdf', 'Team Defence Guide - Internal'),
    ]
    for path in outputs:
        print(path)
