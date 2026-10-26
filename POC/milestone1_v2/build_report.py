"""Build the report and internal guide from their Markdown sources.

Run from any directory: python POC/milestone1_v2/build_report.py
No network requests or project-model dependencies are involved.
"""
from pathlib import Path
import html
import re
import argparse
import io
import shutil

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
    PageBreak, HRFlowable, Preformatted,
)
from reportlab.pdfgen import canvas as canvas_module
from pypdf import PdfReader, PdfWriter
from pipeline_figure import pipeline, export_assets

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output' / 'pdf'
OUT.mkdir(parents=True, exist_ok=True)
NAVY = colors.HexColor('#12202B')
TEAL = colors.HexColor('#1F5FAE')
GRAY = colors.HexColor('#6B7780')


def fonts():
    base = Path('C:/Windows/Fonts')
    names = [('Studio', 'arial.ttf'), ('Studio-Bold', 'arialbd.ttf'),
             ('Studio-Italic', 'ariali.ttf'), ('Studio-BoldItalic', 'arialbi.ttf')]
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
    ('Text', 10.3, 13.8, '#1B2228', 0, 7, False),
    ('Title2', 21, 25, '#12202B', 0, 12, True),
    ('Section', 15, 18, '#12202B', 8, 6, True),
    ('Subsection', 11.5, 14, '#1F5FAE', 6, 4, True),
    ('Cell', 8.0, 10.0, '#1B2228', 0, 0, False),
    ('CellHead', 8.0, 10.0, '#1B2228', 0, 0, True),
    ('Reference', 8.3, 10.7, '#1B2228', 0, 5, False),
]:
    STYLES.add(ParagraphStyle(name=name, fontName=BOLD if bold else REG,
        fontSize=size, leading=leading, textColor=colors.HexColor(color),
        spaceBefore=before, spaceAfter=after, alignment=TA_LEFT,
        splitLongWords=True, allowWidows=0, allowOrphans=0,
        keepWithNext=name in ('Title2', 'Section', 'Subsection')))
STYLES.add(ParagraphStyle(name='Callout', parent=STYLES['Text'], spaceAfter=0))
STYLES.add(ParagraphStyle(name='Caption', parent=STYLES['Reference'],
    textColor=GRAY, fontSize=8.5, leading=11, spaceAfter=9))
STYLES.add(ParagraphStyle(name='BodyBullet', parent=STYLES['Text'],
    leftIndent=11, firstLineIndent=0, bulletIndent=0, spaceAfter=5))
STYLES.add(ParagraphStyle(name='MatrixCell', parent=STYLES['Cell'], splitLongWords=False))
STYLES.add(ParagraphStyle(name='MatrixHead', parent=STYLES['CellHead'], splitLongWords=False))
STYLES.add(ParagraphStyle(name='CodeBlock', fontName='Courier', fontSize=8,
    leading=10.5, leftIndent=8, rightIndent=8, spaceBefore=4, spaceAfter=9))


def clean(s):
    return s.replace('\u2011', '-').replace('\u2013', '-').replace('\u2014', ' - ')


def inline(node):
    if isinstance(node, NavigableString):
        text = html.escape(clean(str(node)))
        # Auto-link plain URLs without changing visible source text.
        return re.sub(r'https?://[^\s<>]+', lambda m: '<a href="' + m[0] + '" color="#1F5FAE">' + m[0] + '</a>', text)
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
        ratios = [0.14, 0.13, 0.14, 0.20, 0.17, 0.22]
    elif n == 4:
        ratios = [0.17, 0.26, 0.30, 0.27]
    elif n == 3 and headings[0] == 'Team member':
        ratios = [0.27, 0.46, 0.27]
    elif n == 3 and headings[0] == 'Objective':
        ratios = [0.18, 0.32, 0.50]
    elif n == 3:
        ratios = [0.20, 0.44, 0.36]
    else:
        ratios = [0.32, 0.68] if n == 2 else [1/n]*n
    data = []
    for ri, row in enumerate(raw):
        cells = row.find_all(['th', 'td'], recursive=False)
        style = ('MatrixHead' if ri == 0 else 'MatrixCell') if n == 6 else ('CellHead' if ri == 0 else 'Cell')
        data.append([para(c, style) for c in cells])
    result = Table(data, colWidths=[r*width for r in ratios], repeatRows=1, hAlign='LEFT')
    padding = 3.5
    result.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E6EBEE')),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), padding),
        ('BOTTOMPADDING', (0, 0), (-1, -1), padding),
        ('GRID', (0, 0), (-1, -1), 0.45, colors.HexColor('#B9C3C9')),
    ]))
    return [Spacer(1, 3), result, Spacer(1, 8)]


def draw_page(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setFont(REG, 8.5)
    canvas.setFillColor(GRAY)
    canvas.drawString(doc.leftMargin, h-34, 'TerraWatch - Group 7')
    header = 'Milestone 1: Problem Definition & Literature Review' if doc.is_report else 'Team Defence Guide - Internal'
    canvas.drawRightString(w-doc.rightMargin, h-34, header)
    canvas.restoreState()


class NumberedCanvas(canvas_module.Canvas):
    """Count body pages before writing original-style Page n / total footers."""
    def __init__(self, *args, cover_offset=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.saved_pages = []
        self.cover_offset = cover_offset

    def showPage(self):
        self.saved_pages.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self.saved_pages) + self.cover_offset
        for state in self.saved_pages:
            self.__dict__.update(state)
            self.setFont(REG, 8.5)
            self.setFillColor(GRAY)
            self.drawRightString(A4[0]-50, 28,
                f'Page {self._pageNumber+self.cover_offset} / {total}')
            super().showPage()
        super().save()


def build(source, destination, label):
    text = source.read_text(encoding='utf-8')
    is_report = source.name == 'Milestone1_Report.md'
    # Keep the separate internal guide compact without changing report typography.
    STYLES['Text'].fontSize = 10.3 if is_report else 9.8
    STYLES['Text'].leading = 13.8 if is_report else 12.5
    STYLES['BodyBullet'].fontSize = STYLES['Text'].fontSize
    STYLES['BodyBullet'].leading = STYLES['Text'].leading
    for name in ('Cell', 'CellHead'):
        STYLES[name].fontSize = 8.0 if is_report else 7.8
        STYLES[name].leading = 10.0 if is_report else 9.5
    if is_report:
        # Team approvals live in Markdown and the PR, outside the submission PDF.
        text = text.split('## Team Section Sign-off', 1)[0].rstrip()
        # The original searchable cover carries the same team and title metadata.
        cover_text = PdfReader(ROOT/'assets'/'original_cover.pdf').pages[0].extract_text()
        for member in ('Priyanshu Agarwal', 'Sarthak Saini', 'Sujal Raj Pradhan', 'Shoaib SSM'):
            if member not in text or member not in cover_text:
                raise ValueError(f'Cover/source team mismatch: {member}')
        text = '## 1. Introduction and Problem Definition' + text.split('## 1. Introduction and Problem Definition', 1)[1]
    text = text.replace('<!-- PAGEBREAK -->', '<div class="pagebreak"></div>')
    soup = BeautifulSoup(markdown.markdown(text, extensions=['tables', 'fenced_code']), 'html.parser')
    body_path = destination.with_name('_body.pdf') if is_report else destination
    doc = SimpleDocTemplate(str(body_path), pagesize=A4, leftMargin=50, rightMargin=50,
        topMargin=60, bottomMargin=45, title=label, author='TerraWatch - Group 7',
        subject='DS & AI Lab - Milestone 1', pageCompression=1)
    doc.short_label = label
    doc.is_report = is_report
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
            if el.name == 'h2':
                rule = HRFlowable(width='100%', color=colors.HexColor('#B9C3C9'),
                    thickness=0.45, spaceBefore=0, spaceAfter=4)
                rule.keepWithNext = True
                story.append(rule)
        elif el.name == 'p':
            if el.find('img'):
                image_node = el.find('img')
                if image_node.get('src') == 'assets/pipeline.png':
                    story.append(pipeline(doc.width, REG, BOLD))
                else:
                    raise ValueError('Unsupported report image: ' + str(image_node.get('src')))
            elif el.get_text().startswith('Problem statement.'):
                # Table padding reserves actual layout space; paragraph backColor padding did not.
                box = Table([[para(el, 'Callout')]], colWidths=[doc.width], hAlign='LEFT')
                box.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#EBF1F7')),
                    ('LINEBEFORE', (0, 0), (0, 0), 2, TEAL),
                    ('LEFTPADDING', (0, 0), (-1, -1), 11),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 11),
                    ('TOPPADDING', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                ]))
                story.extend([Spacer(1, 3), box, Spacer(1, 10)])
            else:
                style = 'Caption' if el.get_text().startswith('Figure 1.') else ('Reference' if in_refs else 'Text')
                story.append(para(el, style))
        elif el.name == 'table':
            story.extend(table(el, doc.width))
        elif el.name in ('ol', 'ul'):
            for i, li in enumerate(el.find_all('li', recursive=False), 1):
                if in_refs:
                    story.append(para(li, 'Reference', f'{i}. '))
                else:
                    bullet = f'{i}.' if el.name == 'ol' else '-'
                    story.append(Paragraph(inline(li), STYLES['BodyBullet'], bulletText=bullet))
        elif el.name == 'hr':
            story.append(HRFlowable(width='100%', color=TEAL))
        elif el.name == 'pre':
            story.append(Preformatted(clean(el.get_text()), STYLES['CodeBlock']))
    doc.build(story, onFirstPage=draw_page, onLaterPages=draw_page,
        canvasmaker=lambda *a, **kw: NumberedCanvas(*a, cover_offset=int(is_report), **kw))
    if is_report:
        writer = PdfWriter()
        writer.append(ROOT/'assets'/'original_cover.pdf')
        writer.append(body_path)
        writer.add_metadata({'/Title': label, '/Author': 'TerraWatch - Group 7'})
        with destination.open('wb') as stream:
            writer.write(stream)
        body_path.unlink()
    else:
        # Bookmarks make the longer internal handbook easy to navigate.
        reader = PdfReader(io.BytesIO(destination.read_bytes()))
        writer = PdfWriter()
        writer.append(reader)
        pages_text = [' '.join((p.extract_text() or '').split()) for p in reader.pages]
        for heading in soup.find_all('h2'):
            title = clean(heading.get_text(' ', strip=True))
            needle = ' '.join(title.split())
            page_index = next((i for i, content in enumerate(pages_text) if needle in content), None)
            if page_index is not None:
                writer.add_outline_item(title, page_index)
        writer.add_metadata({'/Title': label, '/Author': 'TerraWatch - Group 7'})
        buffer = io.BytesIO()
        writer.write(buffer)
        destination.write_bytes(buffer.getvalue())
    return destination


def publish_report(destination):
    """Stage the submission artifacts in reports/ for branch review."""
    reports = ROOT.parents[1] / 'reports'
    figures = reports / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    source = (ROOT / 'Milestone1_Report.md').read_text(encoding='utf-8')
    source = source.replace('assets/iitm_logo.png', 'figures/iitm_logo.png')
    source = source.replace('assets/pipeline.png', 'figures/m1_pipeline.png')
    (reports / 'Milestone1_Report.md').write_text(source, encoding='utf-8', newline='\n')
    shutil.copy2(destination, reports / 'DSAI Lab_Group 7_Milestone 1.pdf')
    shutil.copy2(ROOT / 'assets' / 'pipeline.png', figures / 'm1_pipeline.png')
    (figures / 'm1_pipeline.svg').write_text(
        (ROOT / 'assets' / 'pipeline.svg').read_text(encoding='utf-8'),
        encoding='utf-8', newline='\n')
    shutil.copy2(ROOT / 'assets' / 'iitm_logo.png', figures / 'iitm_logo.png')
    return reports


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--report-only', action='store_true', help='Leave the internal guide PDF unchanged')
    group.add_argument('--guide-only', action='store_true', help='Leave the submission PDF unchanged')
    group.add_argument('--publish-report', action='store_true', help='Build only the report and copy the PDF, Markdown and figure assets into reports/ for branch review')
    args = parser.parse_args()
    export_assets(ROOT/'assets')
    outputs = []
    if not args.guide_only:
        outputs.append(build(ROOT/'Milestone1_Report.md', OUT/'TerraWatch_Milestone1_Revised.pdf', 'Milestone 1: Problem Definition & Literature Review'))
    if not args.report_only and not args.publish_report:
        outputs.append(build(ROOT/'Team_Defence_Guide.md', OUT/'TerraWatch_M1_Team_Defence_Guide.pdf', 'Team Defence Guide - Internal'))
    if args.publish_report:
        print(publish_report(outputs[0]))
    for path in outputs:
        print(path)
