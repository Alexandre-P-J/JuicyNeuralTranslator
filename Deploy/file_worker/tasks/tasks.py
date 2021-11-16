from tasks import app
from celery.result import allow_join_result
import os
import uuid

STORAGE_DIR = os.getenv("STORAGE_MOUNT_DIR")


def unique_filename() -> str:
    return str(uuid.uuid4().hex)


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
