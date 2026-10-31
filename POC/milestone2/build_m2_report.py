"""Build the M2 PDF with the approved M1 cover and typography.

Run: python POC/milestone2/build_m2_report.py
The report is built entirely from the local Markdown and declared figure data.
"""
from pathlib import Path
import io
import json
import sys

import markdown
import pymupdf
from bs4 import BeautifulSoup, NavigableString
from pypdf import PdfReader,PdfWriter
from reportlab.graphics import renderPDF,renderSVG
from reportlab.graphics.shapes import Drawing,Rect,String,Line,Polygon
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate,Paragraph,PageBreak,Spacer,Table,TableStyle,HRFlowable,Image

ROOT=Path(__file__).resolve().parent
M1=ROOT.parent/'milestone1_v2'
sys.path.insert(0,str(M1))
from build_report import STYLES,REG,BOLD,inline,para,NumberedCanvas

ASSETS=ROOT/'assets'
OUT=ROOT/'output/pdf'
BLUE=colors.HexColor('#1F5FAE')
GRAY=colors.HexColor('#6B7780')
NAVY=colors.HexColor('#12202B')


def flow():
    d=Drawing(495,153)
    labels=[('LOAD RELEASE','Ordered image pairs','Masks + five captions'),
            ('CHECK QUALITY','Decode + verify hashes','Audit labels + repeats'),
            ('FIX SPLITS','Canonical pair records','Retain + exclude ledger'),
            ('READY INPUTS','RGB pairs + targets','Verified batch loader')]
    for i,(heading,line1,line2) in enumerate(labels):
        x=i*128
        d.add(Rect(x,65,111,77,rx=4,ry=4,fillColor=colors.white,strokeColor=colors.HexColor('#B9C3C9'),strokeWidth=.7))
        d.add(Rect(x+.4,119,110.2,22.6,fillColor=colors.HexColor('#E6EBEE'),strokeColor=None))
        for y,text,font,size in [(127,heading,BOLD,7.8),(101,line1,REG,8),(85,line2,REG,8)]:
            d.add(String(x+55.5,y,text,fontName=font,fontSize=size,textAnchor='middle',fillColor=NAVY))
        if i<3:
            d.add(Line(x+111,101,x+125,101,strokeColor=BLUE,strokeWidth=1))
            d.add(Polygon([x+125,101,x+121,104,x+121,98],fillColor=BLUE,strokeColor=None))
    d.add(Line(55.5,65,55.5,30,strokeColor=GRAY,strokeWidth=.8))
    d.add(Line(55.5,30,252,30,strokeColor=GRAY,strokeWidth=.8))
    d.add(Polygon([252,30,248,33,248,27],fillColor=GRAY,strokeColor=None))
    d.add(Rect(256,9,239,42,rx=4,ry=4,fillColor=colors.HexColor('#EBF1F7'),strokeColor=colors.HexColor('#B9C3C9'),strokeWidth=.7))
    d.add(String(375.5,35,'SEPARATE REFERENCE STORE',fontName=BOLD,fontSize=8,textAnchor='middle',fillColor=BLUE))
    d.add(String(375.5,19,'Source masks and five reference captions',fontName=REG,fontSize=8,textAnchor='middle',fillColor=NAVY))
    d.add(String(55.5,12,'Evaluation references',fontName=REG,fontSize=7.6,textAnchor='middle',fillColor=GRAY))
    return d


def prepare_assets():
    ASSETS.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    (ASSETS/'iitm_logo.png').write_bytes((M1/'assets/iitm_logo.png').read_bytes())
    doc=pymupdf.open(M1/'assets/original_cover.pdf')
    page=doc[0]
    page.add_redact_annot(pymupdf.Rect(60,270,535,293),fill=(1,1,1))
    page.apply_redactions(images=0,graphics=0)
    from reportlab.pdfgen import canvas
    overlay=io.BytesIO();c=canvas.Canvas(overlay,pagesize=A4)
    text='MILESTONE 2: DATASET PREPARATION';size=8.7;spacing=1.75
    from reportlab.pdfbase import pdfmetrics
    width=pdfmetrics.stringWidth(text,REG,size)+spacing*(len(text)-1)
    t=c.beginText((A4[0]-width)/2,A4[1]-283)
    t.setFont(REG,size);t.setCharSpace(spacing);t.setFillColor(BLUE);t.textOut(text);c.drawText(t);c.save()
    over=pymupdf.open(stream=overlay.getvalue(),filetype='pdf')
    page.show_pdf_page(page.rect,over,0)
    doc.save(ASSETS/'cover.pdf',garbage=4,deflate=True)
    d=flow()
    renderSVG.drawToFile(d,str(ASSETS/'data_flow.svg'))
    tmp=pymupdf.open(stream=renderPDF.drawToString(d),filetype='pdf')
    tmp[0].get_pixmap(matrix=pymupdf.Matrix(3,3),alpha=False).save(ASSETS/'data_flow.png')
    # EDA figures come from executed calculations, never hard-coded aggregates.
    import subprocess
    subprocess.run([sys.executable,str(ROOT/'scripts/eda_figures.py')],check=True)
    # Added figures are versioned assets; regenerate them from the full audit when available.
    if not all((ASSETS/name).exists() for name in ['eda_source_comparison.png','eda_image_mask_captions.png']):
        subprocess.run([sys.executable,str(ROOT/'scripts/report_eda_examples.py')],check=True)


def table(node,width):
    rows=node.find_all('tr');n=len(rows[0].find_all(['th','td'],recursive=False))
    headings=[x.get_text(' ',strip=True) for x in rows[0].find_all(['th','td'],recursive=False)]
    ratios=([.22,.34,.44] if n==3 else [.28,.20,.23,.29] if n==4 else [.30,.70] if n==2 else [1/n]*n)
    if n==3 and headings[0] in ('Partition','Split'):ratios=[.36,.32,.32]
    if n==4 and headings[0]=='Team member':ratios=[.32,.14,.27,.27]
    if n==3 and headings[0]=='Revision':ratios=[.20,.40,.40]
    cells=[[para(c,'CellHead' if i==0 else 'Cell') for c in row.find_all(['th','td'],recursive=False)] for i,row in enumerate(rows)]
    t=Table(cells,colWidths=[r*width for r in ratios],hAlign='LEFT',repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E6EBEE')),('GRID',(0,0),(-1,-1),.45,colors.HexColor('#B9C3C9')),
        ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),
        ('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
    return t


def draw_page(c,doc):
    c.saveState();c.setFillColor(GRAY);c.setFont(REG,8.5)
    c.drawString(50,A4[1]-34,'TerraWatch - Group 7')
    c.drawRightString(A4[0]-50,A4[1]-34,'Milestone 2: Dataset Preparation');c.restoreState()


def build():
    prepare_assets()
    source=(ROOT/'Milestone2_Report.md').read_text(encoding='utf-8')
    # Document control and dated sign-off fields belong in both submitted formats.
    source='## 1. Introduction'+source.split('## 1. Introduction',1)[1]
    source=source.replace('<!-- PAGEBREAK -->','<div class="pagebreak"></div>')
    soup=BeautifulSoup(markdown.markdown(source,extensions=['tables']),'html.parser')
    doc=SimpleDocTemplate(str(OUT/'_body.pdf'),pagesize=A4,leftMargin=50,rightMargin=50,topMargin=60,bottomMargin=45,
        title='TerraWatch - Milestone 2: Dataset Preparation',author='TerraWatch - Group 7')
    story=[];refs=False
    for el in soup.children:
        if isinstance(el,NavigableString):continue
        if el.name=='div':story.append(PageBreak())
        elif el.name in ['h2','h3']:
            refs=el.get_text(strip=True)=='References' if el.name=='h2' else refs
            story.append(para(el,'Section' if el.name=='h2' else 'Subsection'))
            if el.name=='h2':
                line=HRFlowable(width='100%',color=colors.HexColor('#B9C3C9'),thickness=.45,spaceAfter=5);line.keepWithNext=True;story.append(line)
        elif el.name=='p':
            img=el.find('img')
            if img:
                if img['src'].endswith('data_flow.png'):story.append(flow())
                else:
                    picture=Image(str(ROOT/img['src']));picture.drawHeight=doc.width*picture.imageHeight/picture.imageWidth;picture.drawWidth=doc.width;story.append(picture)
            else:story.append(para(el,'Caption' if el.get_text().startswith('Figure ') else 'Reference' if refs else 'Text'))
        elif el.name=='table':story.extend([Spacer(1,3),table(el,doc.width),Spacer(1,9)])
        elif el.name in ['ul','ol']:
            for i,li in enumerate(el.find_all('li',recursive=False),1):
                if refs:story.append(para(li,'Reference',f'{i}. '))
                else:story.append(Paragraph(inline(li),STYLES['BodyBullet'],bulletText='-'))
    doc.build(story,onFirstPage=draw_page,onLaterPages=draw_page,
        canvasmaker=lambda *a,**kw:NumberedCanvas(*a,cover_offset=1,**kw))
    writer=PdfWriter();writer.append(ASSETS/'cover.pdf');writer.append(OUT/'_body.pdf')
    writer.add_metadata({'/Title':'TerraWatch - Milestone 2: Dataset Preparation','/Author':'TerraWatch - Group 7'})
    target=OUT/'TerraWatch_Milestone2_Report.pdf'
    with target.open('wb') as f:writer.write(f)
    (OUT/'_body.pdf').unlink()
    print(target)
    print('Pages:',len(PdfReader(target).pages),'Markdown words:',len((ROOT/'Milestone2_Report.md').read_text(encoding='utf-8').split()))


if __name__=='__main__':build()
