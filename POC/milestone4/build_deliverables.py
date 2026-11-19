"""Build the M4 PDF and meeting slides from the report and recorded evidence."""
from pathlib import Path
from io import BytesIO
import json
import sys
import textwrap

import markdown
import pymupdf
from bs4 import BeautifulSoup, NavigableString
from pypdf import PdfWriter
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.platypus import HRFlowable, PageBreak, SimpleDocTemplate, Spacer, Image

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'deliverables'
sys.path.insert(0,str(HERE.parent/'milestone3'))
import build_m3_report as theme


def cover():
    source=theme.M1/'assets/original_cover.pdf'
    doc=pymupdf.open(source);page=doc[0]
    page.add_redact_annot(pymupdf.Rect(60,270,535,293),fill=(1,1,1))
    for rect in page.search_for('October 2026'):page.add_redact_annot(rect,fill=(1,1,1))
    page.apply_redactions(images=0,graphics=0)
    overlay=BytesIO();c=canvas.Canvas(overlay,pagesize=A4)
    c.setFont(theme.REG,9);c.setFillColor(theme.BLUE)
    c.drawCentredString(A4[0]/2,A4[1]-283,'MILESTONE 4: MODEL TRAINING AND EVIDENCE REVIEW')
    c.setFillColor(theme.GRAY);c.drawCentredString(A4[0]/2,A4[1]-746,'November 2026');c.save()
    with pymupdf.open(stream=overlay.getvalue(),filetype='pdf') as layer:page.show_pdf_page(page.rect,layer,0)
    receipt=BytesIO(doc.tobytes(garbage=4,deflate=True));doc.close();return receipt


def build_report():
    source=(HERE/'Milestone4_Report.md').read_text(encoding='utf-8-sig')
    body='## 1. Introduction'+source.split('## 1. Introduction',1)[1]
    soup=BeautifulSoup(markdown.markdown(body.replace('<!-- PAGEBREAK -->','<div class="pagebreak"></div>'),extensions=['tables']),'html.parser')
    stream=BytesIO();doc=SimpleDocTemplate(stream,pagesize=A4,leftMargin=50,rightMargin=50,topMargin=60,bottomMargin=48,
        title='TerraWatch - Milestone 4',author='Group 7')
    theme.STYLES['Text'].fontSize=10;theme.STYLES['Text'].leading=13.3;theme.STYLES['Text'].spaceAfter=8
    story=[]
    for element in soup.children:
        if isinstance(element,NavigableString):continue
        if element.name=='div':story.append(PageBreak())
        elif element.name in ('h2','h3'):
            story.append(theme.para(element,'Section' if element.name=='h2' else 'Subsection'))
        elif element.name=='p':
            img=element.find('img')
            if img:
                picture=Image(str(HERE/img['src']));picture.drawHeight=doc.width*picture.imageHeight/picture.imageWidth;picture.drawWidth=doc.width;story.append(picture)
            else:story.append(theme.para(element,'Text'))
        elif element.name=='table':story.extend([Spacer(1,4),theme.table(element,doc.width),Spacer(1,8)])
        elif element.name in ('ol','ul'):
            for i,item in enumerate(element.find_all('li',recursive=False),1):story.append(theme.para(item,'Reference',f'{i}. '))
    def header(c,doc):
        c.setFont(theme.REG,8.5);c.setFillColor(theme.GRAY)
        c.drawString(50,A4[1]-34,'TerraWatch - Group 7');c.drawRightString(A4[0]-50,A4[1]-34,'Milestone 4: Training and evidence review')
    doc.build(story,onFirstPage=header,onLaterPages=header,canvasmaker=lambda *a,**kw:theme.NumberedCanvas(*a,cover_offset=1,**kw))
    writer=PdfWriter();writer.append(cover());stream.seek(0);writer.append(stream)
    writer.add_metadata({'/Title':'TerraWatch - Milestone 4: Training and evidence review','/Author':'IIT Madras Group 7'})
    with (OUT/'TerraWatch_Milestone4_Report.pdf').open('wb') as file:writer.write(file)


def build_slides():
    c=canvas.Canvas(str(OUT/'TerraWatch_Milestone4_Slides.pdf'),pagesize=(960,540))
    c.setTitle('TerraWatch Milestone 4 review');navy='#0C1F29';cream='#F7F5EF';lime='#D9ED85';muted='#607780'
    def text(x,y,s,size,color,font=theme.REG):
        c.setFillColor(colors.HexColor(color));c.setFont(font,size);c.drawString(x,y,s)
    def base(title,dark,number):
        bg,fg=(navy,cream) if dark else (cream,navy)
        c.setFillColor(colors.HexColor(bg));c.rect(0,0,960,540,fill=1,stroke=0)
        text(48,497,'TERRAWATCH    /    MILESTONE 04',11,lime if dark else muted,theme.BOLD)
        text(48,434,title,33,fg,theme.BOLD)
        text(48,28,'IIT MADRAS  /  BSDA4001  /  GROUP 7',10,lime if dark else muted);text(880,28,f'{number:02d}',10,lime if dark else muted)
        return fg
    def block(title,points,dark,number):
        fg=base(title,dark,number);y=367
        for heading,body in points:
            text(48,y,heading,19,lime if dark else navy,theme.BOLD);y-=31
            for line in textwrap.wrap(body,83):text(48,y,line,17,fg);y-=26
            y-=20
        c.showPage()
    fg=base('Training and evidence review',True,1)
    text(48,344,'TERRAWATCH',61,lime,theme.BOLD)
    text(50,258,'Building and road changes in paired satellite imagery',23,fg)
    text(50,147,'November 2026',15,fg)
    text(50,105,'Priyanshu Agarwal  /  Sarthak Saini  /  Sujal Raj Pradhan  /  Shoaib SSM',15,fg);c.showPage()
    block('From image pair to reviewed claim',[('Detect and describe','BIT predicts a three-class mask. RSCaMa reads the ordered images and generates a caption. The parser extracts individual claims.'),('Inspect and decide','The witness matches a region and inspects both crops. Qwen reads the claim and evidence, and the gate records accept, reject or defer.'),('Review and correct','Inspect the original images, mask and crops. Save review decisions and corrections with a traceable history.')],False,2)
    block('Training and validation',[('9,544 prepared LEVIR-MCI pairs','The strict dataset has 6,444 train, 1,242 validation and 1,858 test pairs. The training comparison samples the train and validation manifests.'),('Detector','ImageNet ResNet18, class-weighted focal and Dice losses, AdamW and paired flips. Validation change-class mean IoU selects the comparison checkpoint.'),('Captioner','The RSCaMa subset retrain uses token cross-entropy and a training-only vocabulary. Development BLEU-4 selects the checkpoint.')],True,3)
    block('Measured model evidence',[('BIT pilot','Epoch 1 reached 47.1% validation all-class macro-IoU. Epoch 2 reached 44.6%. On the matched evaluation screen BIT scored 45.8% binary F1 versus 16.7% for Otsu.'),('RSCaMa local comparison','Released RSCaMa reached 78.69% strict automatic claim precision and 0.6223 BLEU-4. The subset retrain scored 76.04% and 0.5793.')],False,4)
    selection_path=HERE/'evidence/colab_comparison/selection.json'
    if not selection_path.exists():selection_path=HERE/'evidence/bit_comparison/selection.json'
    if selection_path.exists():
        selection=json.loads(selection_path.read_text());winner=selection['selected'];curves=selection_path.parent/'learning_curves.png'
        if winner and curves.exists():
            base('BIT learning-rate comparison',True,5)
            c.drawImage(str(curves),48,103,width=840,height=270,mask='auto',preserveAspectRatio=True,anchor='sw')
            text(48,67,f"Selected {winner['arm']} at epoch {winner['epoch']} with {100*winner['change_class_mean_iou']:.2f}% validation change-class mean IoU.",16,cream)
            c.showPage()
    block('Live code walkthrough',[('Two real pairs completed the learned pipeline','Review the predicted masks and generated claims. Both outcomes deferred. The no-change claim had a raw accept score of 0.6357 against a 0.80 gate.'),('Record a review','Flag a claim, save a correction or decision, reload and inspect the review events. Export the audit record.'),('Trace the outputs','Show the run ID, original image hashes, decision reasons and model/control provenance. Retrieve the exact run from the Walkthrough guide.')],True,6 if selection_path.exists() and winner and curves.exists() else 5)
    c.save()


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    build_report();build_slides()
