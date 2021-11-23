from tasks import app, STORAGE_DIR, unique_filename
from celery.result import allow_join_result
import os
from tasks.pdf import replace_text, translate_callable

@app.task(name='process_txt')
def process_txt(filename, from_lang, to_lang):
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

@app.task(name='process_pdf')
def process_pdf(filename, from_lang, to_lang):
    new_filename = unique_filename() + ".pdf"
    replace_text(os.path.join(STORAGE_DIR, filename),
                 os.path.join(STORAGE_DIR, new_filename),
                 translate_callable(app, from_lang, to_lang))
    return new_filename