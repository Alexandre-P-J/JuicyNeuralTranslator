from models import Model
from typing import List, Set
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pathlib


class CaEnTL(Model):
    __folder = pathlib.Path(__file__).parent.resolve()
    __model_path = str(pathlib.Path.joinpath(
        __folder, "Checkpoints/ca-en/run02/opus-mt-transfer-ca-to-en/checkpoint-95500"))
    __model = AutoModelForSeq2SeqLM.from_pretrained(__model_path)
    __tokenizer = AutoTokenizer.from_pretrained(__model_path)
    __sentencizer = Model.get_spacy_sentencecizer()

    @classmethod
    def get_source_langs(cls) -> Set[str]:
        return {"Catalan (Standard)"}

    @classmethod
    def get_target_langs(cls) -> Set[str]:
        return {"English (Standard) [Transfer from es-en to ca-en]"}

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
