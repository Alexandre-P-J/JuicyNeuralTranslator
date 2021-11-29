from models import Model
from typing import List, Set
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class CaEn(Model):
    __model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ca-en")
    __tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ca-en")
    __sentencizer = Model.get_spacy_sentencecizer()

    @classmethod
    def get_source_langs(cls) -> Set[str]:
        return {"Catalan (Standard)"}

    @classmethod
    def get_target_langs(cls) -> Set[str]:
        return {"English (Standard) [ca-en]"}

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
