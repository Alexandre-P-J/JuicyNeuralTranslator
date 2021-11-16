from canvas_from_pdf import CanvasFromPDF
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTChar, LTTextContainer, LAParams

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

in_path = "/home/pyro/Downloads/AD-tema2-21-22.pdf"
#in_path = "/home/pyro/Projects/TFG/Docs/GEP02/Entrega.pdf"
out_path = "out.pdf"

pdf_r = extract_pages(in_path, laparams=LAParams(detect_vertical=True))
with CanvasFromPDF(in_path, out_path) as pdf_w:
    for page_r, page_w in zip(pdf_r, pdf_w.pages()):
        for element in page_r:
            if isinstance(element, LTTextContainer):
                text = fix(element.get_text())
                if len(text) > 0:
                    print(text)
