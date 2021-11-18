from os import read
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTChar, LTTextContainer, LAParams
import pikepdf





#path = "/home/pyro/Projects/TFG/Docs/GEP02/Entrega.pdf"
path = "/home/pyro/Downloads/AD-tema2-21-22.pdf"


def fix(text):
    text = text.replace('ﬁ', 'fi')
    text = text.replace('´ı', 'í')
    text = text.replace('´a', 'á')
    text = text.replace('´e', 'é')
    text = text.replace('´i', 'í')
    text = text.replace('´o', 'ó')
    text = text.replace('´u', 'ú')
    text = text.replace('-\n', '')
    text = text.replace('\n', ' ')
    return text


pdf_r = extract_pages(path, laparams=LAParams(detect_vertical=True))
pdf_w = pikepdf.Pdf.open(path)

for page_r, page_w in zip(pdf_r, pdf_w.pages):
    for element in page_r:
        if isinstance(element, LTTextContainer):
            text = fix(element.get_text())
            if len(text) > 0:
                pass

pdf_w.save("out.pdf")
