from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time

languages = {"French": "fr", "French (Belgium)": "fr_BE", "French (Canadian)": "fr_CA",
             "French (France)": "fr_FR", "Walloon": "wa", "frp": "frp", "Occitan": "oc",
             "Catalan": "ca", "Romansh": "rm", "lld": "lld", "Friulian": "fur", "lij": "lij", "lmo": "lmo",
             "Spanish": "es", "Spanish (Argentina)": "es_AR", "Spanish (Chile)": "es_CL",
             "Spanish (Colombia)": "es_CO", "Spanish (Costa Rica)": "es_CR",
             "Spanish (Dominican Republic)": "es_DO", "Spanish (Ecuador)": "es_EC",
             "Spanish (Spain)": "es_ES", "Spanish (Guatemala)": "es_GT", "Spanish (Honduras)": "es_HN",
             "Spanish (Mexico)": "es_MX", "Spanish (Nicaragua)": "es_NI", "Spanish (Panama)": "es_PA",
             "Spanish (Peru)": "es_PE", "Spanish (Puerto Rico)": "es_PR", "Spanish (El Salvador)": "es_SV",
             "Spanish (Uruguay)": "es_UY", "Spanish (Venezuela)": "es_VE", "Portuguese": "pt",
             "pt_br": "pt_br", "Portuguese (Brasil)": "pt_BR", "Portuguese (Portugal)": "pt_PT",
             "Galician": "gl", "Ladin": "lad", "Aragonese": "an", "Mirandese": "mwl", "Italian": "it",
             "Italian (Italia)": "it_IT", "Corsican": "co", "Neapolitan": "nap", "Sicilian": "scn",
             "Venetian": "vec", "Sardinian": "sc", "Romanian": "ro", "Latin": "la"}



def translate(in_text, model, tokenizer, lang_id="es"):
    text = f'>>{lang_id}<<{in_text}'
    input_ids = tokenizer.encode(text, return_tensors="pt")
    outputs = model.generate(input_ids)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded


init_start = time.time()
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ROMANCE")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ROMANCE")
init_time = time.time() - init_start

compute_start = time.time()
text = 'On Unix, return the current processor time as a floating point number expressed in seconds. The precision, and in fact the very definition of the meaning of "processor time", depends on that of the C function of the same name.'

#   On Windows, this function returns wall-clock seconds elapsed since the first call to this function, as a floating point number, based on the Win32 function QueryPerformanceCounter(). The resolution is typically better than one microsecond."
text = translate(text, model, tokenizer, "es")
compute_time = time.time() - compute_start

print(init_time, compute_time)
print(text)

