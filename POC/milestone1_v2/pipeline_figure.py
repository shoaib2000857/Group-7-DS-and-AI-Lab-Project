"""Editable vector pipeline used by both the PDF and Markdown report."""
from pathlib import Path
from reportlab.graphics.shapes import Drawing, Rect, Line, Polygon, String
from reportlab.graphics import renderSVG
from reportlab.lib import colors


def pipeline(width=495, font='Helvetica', bold='Helvetica-Bold'):
    height = 190
    drawing = Drawing(width, height)
    navy = colors.HexColor('#12202B')
    blue = colors.HexColor('#1F5FAE')
    muted = colors.HexColor('#506273')
    border = colors.HexColor('#B9C3C9')
    card_width = (width-40)/5
    gap = 10
    y, card_height = 20, 123
    cards = [
        ('01  INPUT', ['Earlier image: T1', 'Later image: T2', '', 'Quality and', 'alignment checks']),
        ('02  DETECTION', ['Predict change', 'mask and regions', '', 'Retain location', 'and uncertainty']),
        ('03  DESCRIPTION', ['Generate a caption', 'from the pair', '', 'Separate claims', 'and link regions']),
        ('04  VERIFICATION', ['Check mask', 'consistency', 'Inspect uncertain', 'claims in T1 / T2', 'Accept / reject / defer']),
        ('05  REPORT', ['Change overlay', 'Supported claims', '', 'Review flags for', 'uncertain changes']),
    ]
    for index, (title, lines) in enumerate(cards):
        x = index*(card_width+gap)
        fill = colors.HexColor('#EBF1F7') if index == 3 else colors.white
        drawing.add(Rect(x, y, card_width, card_height, rx=4, ry=4,
                         fillColor=fill, strokeColor=blue if index == 3 else border, strokeWidth=0.7))
        drawing.add(Rect(x+0.4, y+card_height-27, card_width-0.8, 26.5,
                         fillColor=colors.HexColor('#E6EBEE'), strokeColor=None))
        drawing.add(String(x+card_width/2, y+card_height-17, title, fontName=bold,
                           fontSize=7.7, textAnchor='middle', fillColor=navy))
        for row, line in enumerate(lines):
            drawing.add(String(x+card_width/2, y+card_height-45-row*13,
                line, fontName=font, fontSize=7.5 if index == 3 else 7.8,
                textAnchor='middle', fillColor=navy if row < 2 else muted))
        if index < 4:
            ax = x+card_width
            ay = y+card_height/2
            drawing.add(Line(ax, ay, ax+gap-2, ay, strokeColor=muted, strokeWidth=0.8))
            drawing.add(Polygon([ax+gap-2,ay,ax+gap-6,ay+2.3,ax+gap-6,ay-2.3],
                                 fillColor=muted, strokeColor=None))
    # Original images bypass the detector/captioner to supply verification evidence.
    start = card_width/2
    end = 3*(card_width+gap)+card_width/2
    top = y+card_height
    drawing.add(Line(start, top, start, 165, strokeColor=blue, strokeWidth=0.8))
    drawing.add(Line(start, 165, end, 165, strokeColor=blue, strokeWidth=0.8))
    drawing.add(Line(end, 165, end, top+3, strokeColor=blue, strokeWidth=0.8))
    drawing.add(Polygon([end,top+1,end-2.5,top+6,end+2.5,top+6], fillColor=blue, strokeColor=None))
    drawing.add(String((start+end)/2, 175, 'Original image pair retained for verification',
                       fontName=font, fontSize=8, textAnchor='middle', fillColor=blue))
    return drawing


def export_assets(folder):
    """SVG is editable; PNG supports ordinary Markdown viewers."""
    import pymupdf
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    svg = renderSVG.drawToString(pipeline())
    (folder/'pipeline.svg').write_text(svg, encoding='utf-8')
    document = pymupdf.open(stream=svg.encode('utf-8'), filetype='svg')
    document[0].get_pixmap(matrix=pymupdf.Matrix(3, 3), alpha=False).save(folder/'pipeline.png')
