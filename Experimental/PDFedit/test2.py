from os import read
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTChar, LTTextContainer, LAParams
import fitz
from fitz import Rect, Point
from fitz import TEXT_ALIGN_JUSTIFY
from ftfy import fix_encoding


def read_layout(path):
    pdf = fitz.open(path)
    for page in pdf:
        page.wrap_contents()
        blocks = page.get_text("blocks")
        for block in blocks:
            area = Rect(block[0], block[1], block[2], block[3])
            page.add_redact_annot(area, fill=False, cross_out=False)
        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
    return pdf


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
    #text = text.replace('\x0c','')
    text = text.replace('-\n', '')
    text = text.replace('\n', ' ')
    return text


pdf_r = extract_pages(path, laparams=LAParams(detect_vertical=True))
pdf_w = read_layout(path)

for page_r, page_w in zip(pdf_r, pdf_w):
    for element in page_r:
        if isinstance(element, LTTextContainer):
            text = fix(element.get_text())
            if len(text) > 0:
                to_pdf_space_mat = page_w.derotation_matrix * page_w.transformation_matrix
                # page_rect = page_w.rect * page_w.derotation_matrix * page_w.transformation_matrix
                # element_pos = Point(element.x0, element.y0) * to_pdf_space_mat
                element_rect = Rect(element.x0, element.y0, element.x1, element.y1)
                print(element_rect, page_r.rect)
                # y = page_rect.y1 - element.y0
                # element_rect = Rect(element.x0, y,
                #                     element.x0 + element.width, y + element.height) * page_w.derotation_matrix
                # element_pos = Point(element_rect.x0, element_rect.y0)

                # r = Rect(element.x0, element.y0, element.x1, element.y1) * page_w.derotation_matrix
                # print(pos, r)

                # area = Rect(pos.x, pos.y,
                #             pos.x + element.width, pos.y + element.height)

                # area = area * page_w.derotation_matrix
                page_w.draw_rect(element_rect)

                # wr = fitz.TextWriter(page_rect=page_rect)
                # #pos = Point(element.x0, page_w.rect.y1 - element.y1)
                # wr.fill_textbox(rect=element_rect, text=text, pos=element_pos, align=TEXT_ALIGN_JUSTIFY)
                # wr.write_text(page_w)

pdf_w.save("out.pdf")
