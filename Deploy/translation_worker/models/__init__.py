from abc import ABC, abstractmethod
from typing import Iterable, List, Optional, Set, Type, Tuple, Callable
from spacy.language import Language


class Model(ABC):
    __registered_models = []
    __source_language_models = {}
    __target_language_models = {}

    @classmethod
    @abstractmethod
    def get_source_langs(cls) -> Set[str]:
        pass

    @classmethod
    @abstractmethod
    def get_target_langs(cls) -> Set[str]:
        pass

    @classmethod
    @abstractmethod
    def batch_translate(cls, texts: List[str], source: str, target: str) -> List[str]:
        pass

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        model_id = len(cls.__registered_models)
        cls.__registered_models.append(cls())
        for lang in cls.get_source_langs():
            cls.__source_language_models.setdefault(lang, [])
            cls.__source_language_models[lang].append(model_id)
        for lang in cls.get_target_langs():
            cls.__target_language_models.setdefault(lang, [])
            cls.__target_language_models[lang].append(model_id)

    @classmethod
    def get_supported_sources(cls) -> Iterable:
        return cls.__source_language_models.keys()

    @classmethod
    def get_supported_targets(cls, source: str) -> Set[str]:
        source_matching_models = cls.__source_language_models.get(source, [])
        targets = set()
        for model_id in source_matching_models:
            t = cls.__registered_models[model_id].get_target_langs()
            targets = targets | set(t)
        return targets

    @staticmethod
    def get_spacy_sentencecizer() -> Callable[[str], Tuple[List[str], List[str]]]:
        spacy_nlp = Language()
        spacy_nlp.add_pipe('sentencizer')

        def split_begining_spacing(text: str) -> Tuple[str, str]:
            for i, c in enumerate(text):
                if not c.isspace():
                    return text[:i], text[i:]
            return text, ""

        def sentencecizer(text: str) -> Tuple[List[str], List[Tuple[bool, str]]]:
            doc = spacy_nlp(text)
            sentences = []
            info = []
            for i, sent in enumerate(doc.sents):
                sent = str(sent)
                spacing, rest = split_begining_spacing(sent)
                spacing = spacing if spacing or (i == 0) else " "
                if rest:
                    info.append((True, spacing))
                    sentences.append(rest)
                else:
                    info.append((False, spacing))
            return sentences, info
        return sentencecizer

    @staticmethod
    def unpack(batch: List[str], sentencecizer: Callable) -> Tuple[List[str], List[List[Tuple[bool, str]]]]:
        infos = []
        result = []
        for text in batch:
            splits, info = sentencecizer(text)
            result.extend(splits)
            infos.append(info)
        return result, infos

    @staticmethod
    def pack(batch: List[str], infos: List[List[Tuple[bool, str]]]) -> List[str]:
        result = []
        index = 0
        for info in infos:
            current_text = ""
            for has_text, spacing in info:
                if has_text:
                    current_text += spacing + batch[index]
                    index += 1
                else:
                    current_text += spacing
            result.append(current_text)
        return result

    @classmethod
    def get_model(cls, source: str, target: str) -> Type['Model']:
        source_matching_models = cls.__source_language_models.get(source, [])
        target_matching_models = cls.__target_language_models.get(target, [])
        candidates = set(source_matching_models) & set(target_matching_models)
        # In case there're multiple valid models, the model with least supported
        # languages is selected. Criterion is, the more specialized the better.
        best_model_id = None
        best_model_score = len(cls.__source_language_models) + len(
            cls.__target_language_models
        )
        for model_id in candidates:
            num_source_langs = len(
                cls.__registered_models[model_id].get_source_langs())
            num_target_langs = len(
                cls.__registered_models[model_id].get_target_langs())
            score = num_source_langs + num_target_langs
            if score <= best_model_score:
                best_model_id = model_id
                best_model_score = score
        return None if best_model_id is None else cls.__registered_models[best_model_id]
