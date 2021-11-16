from os import read
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTChar, LTTextContainer, LAParams
import fitz
from fitz import Rect, Point
from fitz import TEXT_ALIGN_JUSTIFY
from ftfy import fix_encoding


from pdf_annotate import PdfAnnotator, Location, Appearance

def read_layout(path):
    pdf = fitz.open(path)
    a = PdfAnnotator(path)
    for i, page in enumerate(pdf):
        page.wrap_contents()
        blocks = page.get_text("blocks")
        for block in blocks:
            area = Rect(block[0], block[1], block[2], block[3])
            page.add_redact_annot(area, fill=False, cross_out=False)

            a.add_annotation(
            'square',
            Location(x1=block[0], y1=block[1], x2=block[2], y2=block[3], page=i),
            Appearance(fill=(1, 0, 0), fill_transparency = 1)
)

        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
    a.write('out.pdf')
    return pdf


path = "/home/pyro/Projects/TFG/Docs/GEP02/Entrega.pdf"
#path = "/home/pyro/Downloads/AD-tema2-21-22.pdf"


def fix(text):
    text = text.replace('ﬁ', 'fi')
    text = text.replace('´ı', 'í')
    text = text.replace('´a', 'á')
    text = text.replace('´e', 'é')
    text = text.replace('´i', 'í')
    text = text.replace('´o', 'ó')
    text = text.replace('´u', 'ú')
    #text = text.replace('\x0c','')
    text = text.replace('-\n', '')
    text = text.replace('\n', ' ')
    return text


pdf_r = extract_pages(path, laparams=LAParams(detect_vertical=True))
pdf_w = read_layout(path)

#pdf_w.save("out.pdf")


from PyPDF2 import PdfFileWriter, PdfFileReader
output = PdfFileWriter()