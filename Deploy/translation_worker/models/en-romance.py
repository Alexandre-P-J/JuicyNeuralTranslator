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
        "French (Standard)": "fr",
        "French (Belgium)": "fr_BE",
        "French (Canadian)": "fr_CA",
        "Walloon (Standard)": "wa",
        "Occitan (Standard)": "oc",
        "Catalan (Standard)": "ca",
        "Romansh (Standard)": "rm",
        "Friulian (Standard)": "fur",
        "Ligurian (Standard)": "lij",
        "Lombard (Standard)": "lmo",
        "Spanish (Standard)": "es",
        "Spanish (Argentina)": "es_AR",
        "Spanish (Chile)": "es_CL",
        "Spanish (Colombia)": "es_CO",
        "Spanish (Costa Rica)": "es_CR",
        "Spanish (Dominican Republic)": "es_DO",
        "Spanish (Ecuador)": "es_EC",
        "Spanish (Spain)": "es_ES",
        "Spanish (Guatemala)": "es_GT",
        "Spanish (Honduras)": "es_HN",
        "Spanish (Mexico)": "es_MX",
        "Spanish (Nicaragua)": "es_NI",
        "Spanish (Panama)": "es_PA",
        "Spanish (Peru)": "es_PE",
        "Spanish (Puerto Rico)": "es_PR",
        "Spanish (El Salvador)": "es_SV",
        "Spanish (Uruguay)": "es_UY",
        "Spanish (Venezuela)": "es_VE",
        "Portuguese (Standard)": "pt",
        "Portuguese (Brasil)": "pt_BR",
        "Portuguese (Portugal)": "pt_PT",
        "Galician (Standard)": "gl",
        "Ladin (Standard)": "lad",
        "Aragonese (Standard)": "an",
        "Mirandese (Standard)": "mwl",
        "Italian (Standard)": "it",
        "Italian (Italy)": "it_IT",
        "Corsican (Standard)": "co",
        "Neapolitan (Standard)": "nap",
        "Sicilian (Standard)": "scn",
        "Venetian (Standard)": "vec",
        "Sardinian (Standard)": "sc",
        "Romanian (Standard)": "ro",
        "Latin (Standard)": "la",
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
