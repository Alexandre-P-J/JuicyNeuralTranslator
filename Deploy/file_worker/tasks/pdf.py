from celery.result import allow_join_result
from PyPDF2 import PdfFileReader, PdfFileWriter
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTChar, LTTextContainer, LAParams
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_JUSTIFY
from io import BytesIO
from xml.sax.saxutils import escape
import ftfy
import math


def translate_callable(app, from_lang, to_lang):
    def process(data):
        result = app.send_task(
            name="translate_multiple",
            queue="translation_low",
            kwargs={"texts": data, "from_lang": from_lang, "to_lang": to_lang})
        return result
    return process


def text_postprocess(text):
    text = ftfy.fix_text(text)
    text = text.replace('ﬁ', 'fi')
    text = text.replace('´ı', 'í')
    text = text.replace('´a', 'á')
    text = text.replace('´e', 'é')
    text = text.replace('´i', 'í')
    text = text.replace('´o', 'ó')
    text = text.replace('´u', 'ú')
    text = text.replace('´ı', 'í')
    text = text.replace('`a', 'à')
    text = text.replace('`e', 'è')
    text = text.replace('`i', 'ì')
    text = text.replace('`o', 'ò')
    text = text.replace('`u', 'ù')
    text = text.replace('˜n', 'ñ')
    # text = text.replace('-\n', '')
    #text = text.replace('\n', ' ')
    text = escape(text)
    return text


def font_size_group(size):
    return math.floor(size/2)*2


def style_extraction(text_object):
    sizes = 0
    acc = 0
    for line in text_object:
        for char in line:
            if isinstance(char, LTChar):
                acc += char.size
                sizes += 1
    style = getSampleStyleSheet()["Normal"]
    style.alignment = TA_JUSTIFY
    style.fontSize = font_size_group(round(acc / sizes))
    style.leading = style.fontSize
    return style


def get_text_overlay(in_path, translate_callable) -> BytesIO:
    pdf_r = extract_pages(in_path, laparams=LAParams(detect_vertical=True))
    out_stream = BytesIO()
    pdf_w = canvas.Canvas(out_stream)

    for page_r in pdf_r:
        pdf_w.setPageSize((page_r.width, page_r.height))
        # extract all text data in page into a batch
        batched_data = []
        for element in page_r:
            if isinstance(element, LTTextContainer):
                text = text_postprocess(element.get_text())
                if len(text) > 0:
                    style = style_extraction(element)
                    batched_data.append({"style": style, "text": text, "width": element.width,
                                         "height": element.height, "x0": element.x0, "y0": element.y0})
        # translate
        text_batch = [d["text"] for d in batched_data]
        with allow_join_result():
            translated = translate_callable(text_batch).get()
            for i in range(len(text_batch)):
                batched_data[i]["text"] = translated[i]
        # write translations
        for data in batched_data:
            parag = Paragraph(data["text"], data["style"])
            x, y = parag.wrap(data["width"], data["height"])
            parag.drawOn(pdf_w, data["x0"], data["y0"])
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


def replace_text(in_path, out_path, translate_callable):
    with get_text_overlay(in_path, translate_callable) as a, get_background(in_path) as b:
        text_pdf = PdfFileReader(a)
        back_pdf = PdfFileReader(b)
        output_pdf = PdfFileWriter()

        for i in range(text_pdf.getNumPages()):
            p1 = text_pdf.getPage(i)
            p0 = back_pdf.getPage(i)
            rotation = p0.get('/Rotate')
            if rotation is None:
                p0.mergePage(p1)
            else:
                p0.mergeRotatedTranslatedPage(p1, rotation, p1.mediaBox.getHeight() / 2,
                                              p1.mediaBox.getHeight() / 2, True)
            output_pdf.addPage(p0)
        with open(out_path, "wb") as out:
            output_pdf.write(out)
