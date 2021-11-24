from tasks import app, STORAGE_DIR, unique_filename
from celery.result import allow_join_result
from celery.exceptions import Ignore
from celery import states
import os
from tasks.pdf import replace_text, translate_callable
from docx import Document
from tasks.docx import fold, unfold


@app.task(name='process_txt')
def process_txt(filename, from_lang, to_lang):
    try:
        with open(os.path.join(STORAGE_DIR, filename), "r") as input:
            data = input.read()
            result = app.send_task(
                name="translate_one",
                queue="translation_low",
                kwargs={"text": data, "from_lang": from_lang, "to_lang": to_lang})
            with allow_join_result():
                translated = result.get()
                new_filename = unique_filename() + ".txt"
                with open(os.path.join(STORAGE_DIR, new_filename), "w") as output:
                    output.write(translated)
                    return new_filename
    except:
        process_txt.update_state(
            state=states.FAILURE,
            meta='Exception while processing text file'
        )
        raise Ignore()


@app.task(name='process_pdf')
def process_pdf(filename, from_lang, to_lang):
    try:
        new_filename = unique_filename() + ".pdf"
        replace_text(os.path.join(STORAGE_DIR, filename),
                     os.path.join(STORAGE_DIR, new_filename),
                     translate_callable(app, from_lang, to_lang))
        return new_filename
    except:
        process_pdf.update_state(
            state=states.FAILURE,
            meta='Exception while processing pdf file'
        )
        raise Ignore()


@app.task(name='process_docx')
def process_docx(filename, from_lang, to_lang):
    try:
        document = Document(os.path.join(STORAGE_DIR, filename))
        texts = unfold(document)
        result = app.send_task(
            name="translate_multiple",
            queue="translation_low",
            kwargs={"texts": texts, "from_lang": from_lang, "to_lang": to_lang})
        new_filename = unique_filename() + ".docx"
        with allow_join_result():
            document = fold(document, result.get())
            document.save(os.path.join(STORAGE_DIR, new_filename))
        return new_filename
    except:
        process_docx.update_state(
            state=states.FAILURE,
            meta='Exception while processing docx file'
        )
        raise Ignore()
