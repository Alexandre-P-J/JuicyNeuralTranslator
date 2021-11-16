from tasks import app, models


@app.task(name="get_supported_sources")
def get_supported_sources():
    return list(models.get_supported_sources())


@app.task(name="get_supported_targets")
def get_supported_targets(from_lang):
    return list(models.get_supported_targets(from_lang))


@app.task(name="translate_one")
def translate_one(text, from_lang, to_lang):
    m = models.get_model(from_lang, to_lang)
    if m is not None:
        return m.batch_translate([text], from_lang, to_lang)[0]
    return None


@app.task(name="translate_multiple")
def translate_multiple(texts, from_lang, to_lang):
    m = models.get_model(from_lang, to_lang)
    if m is not None:
        return m.batch_translate(texts, from_lang, to_lang)
    return None
