from models import Model
from typing import List, Set
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class EnRomance(Model):
    __model = AutoModelForSeq2SeqLM.from_pretrained(
        "Helsinki-NLP/opus-mt-en-ROMANCE")
    __tokenizer = AutoTokenizer.from_pretrained(
        "Helsinki-NLP/opus-mt-en-ROMANCE")
    __sentencizer = Model.get_spacy_sentencecizer()
    __languages = {
        "French (Standard) [en-romance]": "fr",
        "French (Belgium) [en-romance]": "fr_BE",
        "French (Canadian) [en-romance]": "fr_CA",
        "Walloon (Standard) [en-romance]": "wa",
        "Occitan (Standard) [en-romance]": "oc",
        "Catalan (Standard) [en-romance]": "ca",
        "Romansh (Standard) [en-romance]": "rm",
        "Friulian (Standard) [en-romance]": "fur",
        "Ligurian (Standard) [en-romance]": "lij",
        "Lombard (Standard) [en-romance]": "lmo",
        "Spanish (Standard) [en-romance]": "es",
        "Spanish (Argentina) [en-romance]": "es_AR",
        "Spanish (Chile) [en-romance]": "es_CL",
        "Spanish (Colombia) [en-romance]": "es_CO",
        "Spanish (Costa Rica) [en-romance]": "es_CR",
        "Spanish (Dominican Republic) [en-romance]": "es_DO",
        "Spanish (Ecuador) [en-romance]": "es_EC",
        "Spanish (Spain) [en-romance]": "es_ES",
        "Spanish (Guatemala) [en-romance]": "es_GT",
        "Spanish (Honduras) [en-romance]": "es_HN",
        "Spanish (Mexico) [en-romance]": "es_MX",
        "Spanish (Nicaragua) [en-romance]": "es_NI",
        "Spanish (Panama) [en-romance]": "es_PA",
        "Spanish (Peru) [en-romance]": "es_PE",
        "Spanish (Puerto Rico) [en-romance]": "es_PR",
        "Spanish (El Salvador) [en-romance]": "es_SV",
        "Spanish (Uruguay) [en-romance]": "es_UY",
        "Spanish (Venezuela) [en-romance]": "es_VE",
        "Portuguese (Standard) [en-romance]": "pt",
        "Portuguese (Brasil) [en-romance]": "pt_BR",
        "Portuguese (Portugal) [en-romance]": "pt_PT",
        "Galician (Standard) [en-romance]": "gl",
        "Ladin (Standard) [en-romance]": "lad",
        "Aragonese (Standard) [en-romance]": "an",
        "Mirandese (Standard) [en-romance]": "mwl",
        "Italian (Standard) [en-romance]": "it",
        "Italian (Italy) [en-romance]": "it_IT",
        "Corsican (Standard) [en-romance]": "co",
        "Neapolitan (Standard) [en-romance]": "nap",
        "Sicilian (Standard) [en-romance]": "scn",
        "Venetian (Standard) [en-romance]": "vec",
        "Sardinian (Standard) [en-romance]": "sc",
        "Romanian (Standard) [en-romance]": "ro",
        "Latin (Standard) [en-romance]": "la",
    }

    @classmethod
    def get_source_langs(cls) -> Set[str]:
        return {"English (Standard)"}

    @classmethod
    def get_target_langs(cls) -> Set[str]:
        return set(cls.__languages.keys())

    @classmethod
    def batch_translate(cls, texts: List[str], source: str, target: str) -> List[str]:
        unpacked, metadata = cls.unpack(texts, cls.__sentencizer)
        if unpacked:
            to_lang = cls.__languages[target]
            unpacked = [f">>{to_lang}<<{text}" for text in unpacked]
            input_ids = cls.__tokenizer(
                unpacked, padding=True, truncation=True, return_tensors="pt"
            ).input_ids
            outputs = cls.__model.generate(input_ids)
            decoded = cls.__tokenizer.batch_decode(
                outputs, skip_special_tokens=True)
            return cls.pack(decoded, metadata)
        return texts
