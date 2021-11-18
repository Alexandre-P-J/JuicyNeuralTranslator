from reportlab.pdfgen import canvas
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTChar, LTTextContainer, LAParams, LTImage, LTFigure


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

r_pdf = extract_pages(in_path, laparams=LAParams(detect_vertical=True))
w_pdf = canvas.Canvas("out.pdf")
for page in r_pdf:
    w_pdf.setPageSize((page.width, page.height))
    print((page.width, page.height))
    for element in page:
        if isinstance(element, LTTextContainer):
            text = fix(element.get_text())
            if len(text) > 0:
                pass  # print(text)
        elif isinstance(element, LTImage):
            print("img")
        elif isinstance(element, LTFigure):
            pass#print("fig")
    w_pdf.showPage()
w_pdf.save()