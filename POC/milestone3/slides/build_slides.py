from pathlib import Path
import sys,io
import pymupdf
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'presentation/DSAI Lab_Group 7_Milestone 3_Slides.pdf'
QA=ROOT/'POC/milestone3/slides';QA.mkdir(parents=True,exist_ok=True)
REF=ROOT/'presentation/TerraWatch_Milestone_1_Redesigned_Pitch_Deck.pdf'
N='#0C1F29';L='#D9ED85';C='#F7F5EF';G='#607780';T='#163640'
for name,file in [('Body','C:/Windows/Fonts/arial.ttf'),('Bold','C:/Windows/Fonts/arialbd.ttf')]:pdfmetrics.registerFont(TTFont(name,file))
W,H=960,540
buf=io.BytesIO();c=canvas.Canvas(buf,pagesize=(W,H));c.setTitle('TerraWatch - Milestone 3 Review');c.setAuthor('TerraWatch Group 7')
def text(x,y,s,size=17,color=N,font='Body'):
 c.setFillColor(HexColor(color));c.setFont(font,size);c.drawString(x,y,s)
def base(num,title,subtitle='',dark=False):
 global fg,bg
 bg=N if dark else C;fg=C if dark else N
 c.setFillColor(HexColor(bg));c.rect(0,0,W,H,fill=1,stroke=0)
 text(48,505,'T E R R A W A T C H  /  M I L E S T O N E  0 3',9,L if dark else G)
 text(48,444,title,29,fg,'Bold')
 if subtitle:text(48,411,subtitle,14,'#B6CCD0' if dark else G)
 c.setStrokeColor(HexColor(T if dark else '#D8DFDB'));c.line(48,40,912,40)
 text(48,23,'IIT MADRAS  /  BSDA4001  /  GROUP 7',8,G);text(875,23,f'{num:02d} / 06',8,G)
def finish():c.showPage()
def lines(x,y,items,size=17,step=29,color=None):
 for s in items:text(x,y,s,size,color or fg);y-=step
# Cover uses the earlier deck's aerial illustration and its explicit caption.
ref=pymupdf.open(REF);pix=ref[0].get_pixmap(matrix=pymupdf.Matrix(1.5,1.5),alpha=False)
from PIL import Image
im=Image.frombytes('RGB',[pix.width,pix.height],pix.samples);im=im.crop((int(im.width*.585),0,im.width,im.height))
c.setFillColor(HexColor(N));c.rect(0,0,W,H,fill=1,stroke=0);c.drawImage(ImageReader(im),560,0,width=400,height=540,mask='auto')
text(48,480,'MODEL SELECTION AND SYSTEM ARCHITECTURE',11,L,'Bold')
text(48,359,'TERRA',58,C,'Bold');text(48,297,'WATCH',58,L,'Bold')
lines(50,226,['From image changes to','evidence-grounded descriptions'],23,32,C)
text(50,107,'Milestone 3 Review',18,C,'Bold');text(50,78,'IIT Madras BS / Data Science & AI Lab / November 2026',11,'#B6CCD0')
text(50,43,'Priyanshu Agarwal   Sarthak Saini   Sujal Raj Pradhan   Shoaib SSM',9,'#B6CCD0');finish()
# Architecture.
base(2,'One image pair. An auditable change description.','Ordered images feed detection and captioning; claims are checked against visual evidence.',True)
labels=[('01','DETECT','BIT-ResNet18','Road / building mask'),('02','DESCRIBE','RSCaMa','Caption and atomic claims'),('03','INSPECT','Structural witness','Matched regions and crops'),('04','DECIDE','Qwen3-VL','Accept / reject / defer')]
for i,(n,title,model,detail) in enumerate(labels):
 x=48+i*221;c.setFillColor(HexColor(T));c.roundRect(x,218,200,144,8,fill=1,stroke=0)
 text(x+16,330,n,19,L,'Bold');text(x+16,300,title,17,C,'Bold');text(x+16,269,model,15,C);text(x+16,242,detail,11,'#B6CCD0')
 if i<3:c.setStrokeColor(HexColor(L));c.line(x+203,289,x+217,289);c.line(x+213,293,x+217,289);c.line(x+213,285,x+217,289)
lines(50,170,['Input: earlier T1 and later T2 RGB images','Output: accepted description, claim decisions, crop evidence and report JSON'],18,33,C)
text(50,75,'Source masks and reference captions stay in the training/evaluation store.',13,'#B6CCD0');finish()
# Model rationale and interfaces.
base(3,'Model choices follow the task.','Domain-specific generation combines with dense localisation and explicit claim verification.')
rows=[('BIT-ResNet18','Shared temporal features','256 x 256 inputs; three-class mask'),('RSCaMa','CLIP + state-space fusion','Paired RGB images; caption up to 42 tokens'),('Visual witness','Parameter-free crop inspection','Original-frame crops; quality and change evidence'),('Qwen3-VL-4B','Image-backed claim judgement','Images + claim + evidence; three decisions')]
for i,(name,why,contract) in enumerate(rows):
 y=345-i*69;c.setStrokeColor(HexColor('#D8DFDB'));c.line(48,y-35,912,y-35)
 text(50,y,name,19,N,'Bold');text(290,y,why,17);text(290,y-24,contract,13,G)
text(50,79,'Serial model workers isolate dependencies and release GPU memory between stages.',16,N);finish()
# Caption evidence native bars.
base(4,'RSCaMa leads the local captioning screen.','100 LEVIR pairs. Automatic claim/reference checks. Same population as the B0 comparison.')
names=['Released RSCaMa','RSCaMa retrain','Qwen3-VL-4B','SmolVLM2','Pixel template','Always no change'];values=[78.69,76.04,55.70,37.76,69,40]
for i,(name,value) in enumerate(zip(names,values)):
 y=355-i*39;text(50,y,name,15);c.setFillColor(HexColor('#DCE4DD'));c.rect(245,y-3,420,21,fill=1,stroke=0);c.setFillColor(HexColor('#799940' if i==0 else '#365966'));c.rect(245,y-3,420*value/100,21,fill=1,stroke=0);text(680,y,f'{value:.2f}%',15,N,'Bold')
text(245,104,'Strict claim precision (%)',12,G)
text(745,340,'0.6223',32,N,'Bold');text(745,312,'BLEU-4',14,G)
text(745,243,'0 / 40',32,N,'Bold');text(745,215,'unchanged-pair',12,G);text(745,197,'false alarms',12,G)
text(50,65,'Released RSCaMa reproduces the stored B0 aggregates at reported precision.',13,G);finish()
# Component evidence.
base(5,'Localisation and visual evidence support the stack.','Component experiments retain their own samples and metric definitions.',True)
text(50,354,'DETECTOR SCREEN',12,L,'Bold');text(50,305,'45.8%',43,C,'Bold');text(260,305,'16.7%',43,'#B6CCD0','Bold')
text(50,275,'BIT binary F1',16,C);text(260,275,'Otsu binary F1',16,C)
lines(50,231,['Two epochs: 100 train / 25 validation / 100 evaluation pairs','Unchanged-pair alarms: BIT 31/45, Otsu 45/45','BIT road/building mean IoU: 26.8%'],14,29,C)
c.setStrokeColor(HexColor('#35515A'));c.line(507,168,507,365)
text(550,354,'REFEREE SCREEN',12,L,'Bold');text(550,305,'56.7%',43,C,'Bold');text(550,275,'Image-backed Qwen3-VL agreement',16,C)
lines(550,231,['Text / common evidence: 41.3%','VL / common evidence: 45.3%','30 pairs and 150 annotation-proxy claims'],15,29,C)
text(50,100,'Visual witness: 50/50 synthetic checks; 2/3 coarse region checks on authored real-image cases.',14,C)
text(50,70,'Caption source: local rerun summary. Referee source: preserved Colab L4 screen.',11,'#B6CCD0');finish()
# Training + traceability.
base(6,'Training controls and evidence remain traceable.','The architecture specifies the complete data flow, model interfaces and checkpoint rules.')
text(50,352,'TRAINING',13,G,'Bold');lines(50,316,['BIT: ImageNet initialisation; focal + Dice loss','AdamW, cosine decay and validation IoU selection','RSCaMa: CLIP initialisation; token cross-entropy','Adam and development BLEU-4 checkpoint selection'],17,33,N)
text(50,151,'VERIFICATION',13,G,'Bold');lines(50,119,['83 software checks in the supplied local run','Raw decisions, coordinates and evidence remain attached to each claim'],16,29,N)
text(600,352,'REPRODUCIBLE SOURCES',13,G,'Bold');lines(600,316,['Pipeline interfaces and training:','POC/pipeline/','Captioning and evaluation:','POC/execution/','Local benchmark proof and scripts:','POC/milestone3/sarthak/'],15,27,N)
text(600,128,'Two independent human raters',15,N,'Bold');text(600,103,'Cohen\'s kappa; third-rater adjudication',13,G)
text(50,64,'References: M3 report [2-9]; sarthak/evaluation_proof.json; consolidation/evidence/checks.json',10,G);finish()
c.save();OUT.parent.mkdir(exist_ok=True);OUT.write_bytes(buf.getvalue())
d=pymupdf.open(OUT);assert len(d)==6
thumbs=[]
for i,page in enumerate(d):
 for b in page.get_text('blocks'):
  assert b[0]>=0 and b[2]<=W+1 and b[1]>=0 and b[3]<=H+1,(i,b)
 pix=page.get_pixmap(matrix=pymupdf.Matrix(.65,.65),alpha=False);im=Image.frombytes('RGB',[pix.width,pix.height],pix.samples);thumbs.append(im)
contact=Image.new('RGB',(1248,1053),'white')
for i,im in enumerate(thumbs):contact.paste(im,((i%2)*624,(i//2)*351))
contact.save(QA/'contact_sheet.png')
print(OUT);print('Six slides generated and text bounds checked.')
