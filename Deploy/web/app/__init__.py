from flask import Flask, render_template, request, json, Markup, send_file
from celery import Celery
from celery.result import AsyncResult
import os
import uuid
from app import database

celery_app = Celery("tasks", backend="redis://redis:6379/0",
                    broker="redis://redis:6379/0")
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.getenv("STORAGE_MOUNT_DIR")
ALLOWED_EXTENSIONS = {'.txt', '.docx', '.pdf'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000  # 16MB

SOURCE_LANGS = celery_app.send_task(
    name="get_supported_sources",
    queue="translation_high")
SOURCE_LANGS = SOURCE_LANGS.get()
assert len(SOURCE_LANGS) > 0
SOURCE_LANGS.sort()

DEFAULT_SOURCE_LANG = "English (Standard)" if (
    "English (Standard)" in SOURCE_LANGS) else SOURCE_LANGS[0]
DB = database.Database()


def filename_extension(filename: str) -> str:
    if '.' in filename:
        return "." + filename.rsplit('.', 1)[1].lower()
    return ''


def allowed_filetype(filename: str) -> str:
    return filename_extension(filename) in ALLOWED_EXTENSIONS


def unique_filename(filename) -> str:
    return str(uuid.uuid4().hex) + filename_extension(filename)


@app.route('/')
def main():
    return render_template('index.html',
                           svg_logo=serve_svg("JT_logo_letra_horizontal.svg"),
                           source_langs=SOURCE_LANGS, default_source_lang=DEFAULT_SOURCE_LANG)


@app.route('/static/img/<svgFile>')
def serve_svg(svgFile):
    path = os.path.join(app.static_folder, "img", svgFile)
    return Markup(open(path).read())


@app.route('/languages', methods=['POST'])
def languages():
    from_lang = request.form["from_lang"]
    targets = celery_app.send_task(
        name="get_supported_targets",
        queue="translation_high",
        kwargs={"from_lang": from_lang})
    return json.dumps({'languages': targets.get()})


@app.route('/text_translate', methods=['POST'])
def text_translate():
    if "task_id" in request.form:
        task_id = request.form["task_id"]
        result = AsyncResult(task_id, app=celery_app)
        if result.successful():
            return json.dumps({'ready': True, 'success': True, 'result': result.get()})
        elif result.failed():
            return json.dumps({'ready': True, 'success': False})
        return json.dumps({'ready': False})
    else:
        in_text = request.form["in-text"]
        from_lang = request.form["from_lang"]
        to_lang = request.form["to_lang"]
        translation = celery_app.send_task(
            name="translate_one",
            queue="translation_high",
            kwargs={"text": in_text, "from_lang": from_lang, "to_lang": to_lang})
        return json.dumps({'success': True, 'result': translation.id})


@app.route('/upload', methods=['POST'])
def upload():
    if 'task_id' in request.form:
        task_id = request.form["task_id"]
        result = AsyncResult(task_id, app=celery_app)
        if result.successful():
            filename = result.get()
            return json.dumps({'ready': True, 'success': True, 'result': "/download/"+filename})
        elif result.failed():
            return json.dumps({'ready': True, 'success': False})
        return json.dumps({'ready': False})
    else:
        if 'file' not in request.files:
            return json.dumps({'success': False})
        file = request.files['file']
        if not allowed_filetype(file.filename):
            return json.dumps({'success': False, "filename_err": True})
        else:
            filename = unique_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            task_name = "process_txt"
            if filename_extension(file.filename) == ".docx":
                task_name = "process_docx"
            elif filename_extension(file.filename) == ".pdf":
                task_name = "process_pdf"
            result = celery_app.send_task(
                name=task_name,
                queue="file",
                kwargs={"filename": filename, "from_lang": request.form.get("from_lang"),
                        "to_lang": request.form.get("to_lang")})
            return json.dumps({'success': True, 'result': result.id})


@app.route('/download/<file>')
def download(file):
    filename_beauty = "translation" + filename_extension(file)
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], file),
                     as_attachment=True, download_name=filename_beauty)


@app.route('/text_correct', methods=['POST'])
def text_correct():
    in_text = request.form["in-text"]
    from_lang = request.form["from_lang"]
    to_lang = request.form["to_lang"]
    out_text = request.form['out-text']
    correct_text = request.form['correct-text']
    DB.save_correction(from_lang, to_lang, in_text, out_text, correct_text)
    return json.dumps({'success': True})


if __name__ == "__main__":
    app.run()
