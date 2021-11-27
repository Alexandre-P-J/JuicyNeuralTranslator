from models import Model
from typing import List, Set
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class RomanceEn(Model):
    __model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ROMANCE-en")
    __tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ROMANCE-en")
    __sentencizer = Model.get_spacy_sentencecizer()
    __languages = {
        "French (Standard)",
        "French (Belgium)",
        "French (Canadian)",
        "Walloon (Standard)",
        "Occitan (Standard)",
        "Catalan (Standard)",
        "Romansh (Standard)",
        "Friulian (Standard)",
        "Ligurian (Standard)",
        "Lombard (Standard)",
        "Spanish (Standard)",
        "Spanish (Argentina)",
        "Spanish (Chile)",
        "Spanish (Colombia)",
        "Spanish (Costa Rica)",
        "Spanish (Dominican Republic)",
        "Spanish (Ecuador)",
        "Spanish (Spain)",
        "Spanish (Guatemala)",
        "Spanish (Honduras)",
        "Spanish (Mexico)",
        "Spanish (Nicaragua)",
        "Spanish (Panama)",
        "Spanish (Peru)",
        "Spanish (Puerto Rico)",
        "Spanish (El Salvador)",
        "Spanish (Uruguay)",
        "Spanish (Venezuela)",
        "Portuguese (Standard)",
        "Portuguese (Brasil)",
        "Portuguese (Portugal)",
        "Galician (Standard)",
        "Ladin (Standard)",
        "Aragonese (Standard)",
        "Mirandese (Standard)",
        "Italian (Standard)",
        "Italian (Italy)",
        "Corsican (Standard)",
        "Neapolitan (Standard)",
        "Sicilian (Standard)",
        "Venetian (Standard)",
        "Sardinian (Standard)",
        "Romanian (Standard)",
        "Latin (Standard)"
    }

    @classmethod
    def get_source_langs(cls) -> Set[str]:
        return cls.__languages

    @classmethod
    def get_target_langs(cls) -> Set[str]:
        return {"English (Standard) [romance-en]"}

    @classmethod
    def batch_translate(cls, texts: List[str], source: str, target: str) -> List[str]:
        unpacked, metadata = cls.unpack(texts, cls.__sentencizer)
        if unpacked:
            input_ids = cls.__tokenizer(
                unpacked, padding=True, truncation=True, return_tensors="pt"
            ).input_ids
            outputs = cls.__model.generate(input_ids)
            decoded = cls.__tokenizer.batch_decode(
                outputs, skip_special_tokens=True)
            return cls.pack(decoded, metadata)
        return texts
