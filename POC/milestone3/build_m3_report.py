"""Build the M3 PDF with the approved M1 cover and typography.

Run: python POC/milestone3/build_m3_report.py
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


def box(d,x,y,w,h,title,line1='',line2=''):
    d.add(Rect(x,y,w,h,rx=4,ry=4,fillColor=colors.HexColor('#F4F7FA'),strokeColor=colors.HexColor('#B9C3C9'),strokeWidth=.6))
    d.add(String(x+w/2,y+h-14,title,fontName=BOLD,fontSize=7.7,textAnchor='middle',fillColor=BLUE))
    for yy,text in [(y+h-29,line1),(y+h-41,line2)]:
        if text:d.add(String(x+w/2,yy,text,fontName=REG,fontSize=7.5,textAnchor='middle',fillColor=NAVY))


def arrow(d,x1,y1,x2,y2):
    import math
    d.add(Line(x1,y1,x2,y2,strokeColor=BLUE,strokeWidth=.9))
    a=math.atan2(y2-y1,x2-x1);r=4
    d.add(Polygon([x2,y2,x2-r*math.cos(a-.5),y2-r*math.sin(a-.5),x2-r*math.cos(a+.5),y2-r*math.sin(a+.5)],fillColor=BLUE,strokeColor=None))


def system_architecture():
    d=Drawing(495,235)
    box(d,0,105,96,58,'ORDERED INPUTS','Earlier T1 / later T2','RGB + provenance')
    box(d,119,171,109,58,'A: BIT','Shared visual encoder','Road / building mask')
    box(d,119,84,109,58,'B: RSCaMa','Paired visual features','Change caption')
    box(d,251,84,109,58,'CLAIM PARSER','Object / action / place','Source span + quantity')
    box(d,379,135,116,65,'C: VISUAL EVIDENCE','Original-frame regions','Crops + quality flags')
    box(d,379,51,116,65,'D: REFEREE','Claim + images + evidence','Accept / reject / defer')
    box(d,251,0,109,58,'FINAL REPORT','Accepted description','Review flag + reasons')
    box(d,0,0,228,58,'SEPARATE REFERENCE STORE','Source masks and caption references','Training / evaluation access only')
    arrow(d,96,145,119,196);arrow(d,96,124,119,113)
    arrow(d,228,200,379,183);arrow(d,228,113,251,113);arrow(d,360,114,379,158)
    arrow(d,437,135,437,116);arrow(d,379,70,360,29)
    return d


def workflow():
    d=Drawing(495,93);labels=[('VALIDATE','T1 / T2'),('DETECT','Class mask'),('DESCRIBE','Caption'),('PARSE','Claims'),('INSPECT','Crops'),('REFEREE','Decisions'),('COMPOSE','Report JSON')]
    for i,(title,line) in enumerate(labels):
        x=i*71;box(d,x,34,68,53,title,line)
        if i<6:arrow(d,x+68,60,x+71,60)
    d.add(String(247.5,14,'Persist masks, source spans, transforms, raw responses and decision reasons',fontName=REG,fontSize=8,textAnchor='middle',fillColor=GRAY))
    return d


def model_architectures():
    d=Drawing(495,145)
    for y,labels in [(84,[('BIT: RESNET18','Shared T1/T2 features'),('TOKEN ENCODER','4 tokens per date'),('TEMPORAL DECODER','8 layers + differences'),('CLASSIFIER','Background / road / building')]),(9,[('RSCaMa: CLIP','7 x 7 patch features'),('SSM FUSION','3 spatial/temporal layers'),('TEXT DECODER','1 Transformer layer'),('CAPTION','Greedy, max 42 tokens')])]:
        for i,(title,line) in enumerate(labels):
            x=i*128;box(d,x,y,111,55,title,line)
            if i<3:arrow(d,x+111,y+27,x+125,y+27)
    return d


DIAGRAMS={'system_architecture.png':system_architecture,'workflow.png':workflow,'model_architectures.png':model_architectures}


def prepare_assets():
    ASSETS.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    (ASSETS/'iitm_logo.png').write_bytes((M1/'assets/iitm_logo.png').read_bytes())
    doc=pymupdf.open(M1/'assets/original_cover.pdf');page=doc[0]
    page.add_redact_annot(pymupdf.Rect(60,270,535,293),fill=(1,1,1))
    for rect in page.search_for('October 2026'):page.add_redact_annot(rect,fill=(1,1,1))
    page.apply_redactions(images=0,graphics=0)
    from reportlab.pdfgen import canvas
    overlay=io.BytesIO();c=canvas.Canvas(overlay,pagesize=A4)
    c.setFillColor(BLUE);c.setFont(REG,8.8);c.drawCentredString(A4[0]/2,A4[1]-283,'MILESTONE 3: MODEL SELECTION AND SYSTEM ARCHITECTURE')
    c.setFillColor(GRAY);c.setFont(REG,8.5);c.drawCentredString(A4[0]/2,A4[1]-746,'November 2026');c.save()
    over=pymupdf.open(stream=overlay.getvalue(),filetype='pdf');page.show_pdf_page(page.rect,over,0)
    doc.save(ASSETS/'cover.pdf',garbage=4,deflate=True)
    for name,make in DIAGRAMS.items():
        d=make();renderSVG.drawToFile(d,str(ASSETS/Path(name).with_suffix('.svg')))
        tmp=pymupdf.open(stream=renderPDF.drawToString(d),filetype='pdf')
        tmp[0].get_pixmap(matrix=pymupdf.Matrix(3,3),alpha=False).save(ASSETS/name)


def table(node,width):
    rows=node.find_all('tr');n=len(rows[0].find_all(['th','td'],recursive=False))
    headings=[x.get_text(' ',strip=True) for x in rows[0].find_all(['th','td'],recursive=False)]
    ratios=([.22,.34,.44] if n==3 else [.28,.20,.23,.29] if n==4 else [.30,.70] if n==2 else [1/n]*n)
    if n==4 and headings[0]=='Captioning configuration':ratios=[.36,.22,.23,.19]
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
    c.drawRightString(A4[0]-50,A4[1]-34,'Milestone 3: Models and Architecture');c.restoreState()


def build():
    prepare_assets()
    STYLES["Text"].fontSize=10
    STYLES["Text"].leading=13.2
    STYLES["Text"].spaceAfter=5
    source=(ROOT/'Milestone3_Report.md').read_text(encoding='utf-8')
    # The report body excludes administrative records maintained separately.
    source='## 1. Introduction'+source.split('## 1. Introduction',1)[1]
    source=source.replace('<!-- PAGEBREAK -->','<div class="pagebreak"></div>')
    soup=BeautifulSoup(markdown.markdown(source,extensions=['tables']),'html.parser')
    doc=SimpleDocTemplate(str(OUT/'_body.pdf'),pagesize=A4,leftMargin=50,rightMargin=50,topMargin=60,bottomMargin=45,
        title='TerraWatch - Milestone 3: Models and Architecture',author='TerraWatch - Group 7')
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
                if Path(img['src']).name in DIAGRAMS:story.append(DIAGRAMS[Path(img['src']).name]())
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
    writer.add_metadata({'/Title':'TerraWatch - Milestone 3: Models and Architecture','/Author':'TerraWatch - Group 7'})
    target=OUT/'TerraWatch_Milestone3_Report.pdf'
    with target.open('wb') as f:writer.write(f)
    # Preserve intermediate builds for layout review.
    from datetime import datetime
    history=ROOT/'qa/builds'/datetime.now().strftime('%Y%m%d_%H%M%S_%f');history.mkdir(parents=True)
    (history/'body.pdf').write_bytes((OUT/'_body.pdf').read_bytes())
    (history/'report.pdf').write_bytes(target.read_bytes())
    print(target)
    print('Pages:',len(PdfReader(target).pages),'Markdown words:',len((ROOT/'Milestone3_Report.md').read_text(encoding='utf-8').split()))


if __name__=='__main__':build()
