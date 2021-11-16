from pdfminer.high_level import extract_pages
from pdfminer.layout import LTChar, LTTextContainer
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.pagesizes import A4
from ftfy import fix_encoding

from collections import Counter
from reportlab.lib.enums import TA_JUSTIFY

out = canvas.Canvas("out.pdf")

def fix(text):
    text = text.replace('ﬁ', 'fi')
    text = text.replace('´ı', 'í')
    text = text.replace('´o', 'ó')
    return text

def text_parse(text_object):
    text = fix(text_object.get_text())
    fonts = Counter()
    sizes = Counter()
    for line in text_object:
        for char in line:
            if isinstance(char, LTChar):
                fonts.update({char.fontname: 1})
                sizes.update({char.size: 1})
    style = getSampleStyleSheet()["Normal"]
    style.alignment = TA_JUSTIFY
    style.fontSize = round(sizes.most_common(1)[0][0]) - 1
    #style.fontName = fonts.most_common(1)[0][0]
    return style

for page_layout in extract_pages("/home/pyro/Downloads/AD-tema2-21-22.pdf"):
    for element in page_layout:
        if isinstance(element, LTTextContainer):
            text = fix_encoding(fix(element.get_text()))
            if len(text) > 0:
                style = text_parse(element)
                parag = Paragraph(text, style)
                x, y = parag.wrap(element.width, element.height)
                parag.drawOn(out, element.x0, element.y0)
                #out.drawString(element.x0, element.y0,element.get_text())
                #print(element.get_text())
    out.showPage()

out.save()