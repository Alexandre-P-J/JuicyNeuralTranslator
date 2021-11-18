from PyPDF2 import PdfFileReader, PdfFileWriter, PdfFileMerger
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTChar, LTTextContainer, LAParams
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_JUSTIFY
from io import BytesIO
from xml.sax.saxutils import escape
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser

def fix(text):
    text = escape(text)
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

def get_text_overlay(in_path) -> BytesIO:
    #pdf_r = extract_pages(in_path, laparams=LAParams(detect_vertical=True))
    out_stream = BytesIO()
    pdf_w = canvas.Canvas(out_stream)
    output_string = BytesIO()
    with open(in_path, "rb") as pdf_r:
        parser = PDFParser(pdf_r)
        doc = PDFDocument(parser)
        rsrcmgr = PDFResourceManager()
        device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        for page_r in PDFPage.create_pages(doc):
            interpreter.process_page(page_r)
            print(page_r.rotate)
            print(page_r.mediabox)

            pdf_w.setPageSize((page_r.mediabox[2], page_r.mediabox[3])) # inverse watchout!
            for element in page_r:
                if isinstance(element, LTTextContainer):
                    text = fix(element.get_text())
                    if len(text) > 0:
                        style = getSampleStyleSheet()["Normal"]
                        style.alignment = TA_JUSTIFY
                        parag = Paragraph(text, style)
                        x, y = parag.wrap(element.width, element.height)
                        parag.drawOn(pdf_w, element.x0, element.y0)
            pdf_w.showPage()
        pdf_w.save()
        out_stream.seek(0)
        return out_stream


def get_background(in_path) -> BytesIO:
    out_stream = BytesIO()
    with open(in_path, "rb") as in_stream:
        original_pdf = PdfFileReader(in_stream)
        output_pdf = PdfFileWriter()

        [output_pdf.addPage(original_pdf.getPage(i))
         for i in range(original_pdf.getNumPages())]
        output_pdf.removeText(ignoreByteStringObject=True)
        output_pdf.removeLinks()

        output_pdf.write(out_stream)
    out_stream.seek(0)
    return out_stream


def replace_text(in_path, out_path):
    with get_text_overlay(in_path) as a, get_background(in_path) as b:
        pdfReader = PdfFileReader(a)
        pdfWriter = PdfFileWriter()

        for pageNum in range(pdfReader.getNumPages()):
            page = pdfReader.getPage(pageNum).rotateCounterClockwise(90)
            pdfWriter.addPage(page)
            
        pdfOut = BytesIO()
        pdfWriter.write(pdfOut)
        pdfOut.seek(0)

        text_pdf = PdfFileReader(pdfOut)
        back_pdf = PdfFileReader(b)
        output_pdf = PdfFileWriter()

        for i in range(text_pdf.getNumPages()):
            p1 = text_pdf.getPage(i)
            p0 = back_pdf.getPage(i)
            #p0.rotateCounterClockwise(90)
            print(p0.mediaBox, p1.mediaBox)
            #print(p0.get('/Rotate'))
            
            p0.mergePage(p1)
            output_pdf.addPage(p1)

        with open(out_path, "wb") as out:
            output_pdf.write(out)


#path = "/home/pyro/Downloads/Practica5-AD-21-22.pdf"
path = "/home/pyro/Downloads/AD-tema2-21-22.pdf"

replace_text(path, "out.pdf")
