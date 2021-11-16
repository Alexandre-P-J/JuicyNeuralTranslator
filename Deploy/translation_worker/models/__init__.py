from abc import ABC, abstractmethod
from typing import Iterable, List, Set, Type, Tuple, Callable
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

        def sentencecizer(text: str) -> Tuple[List[str], List[str]]:
            doc = spacy_nlp(text)
            sentences = []
            joints = []
            for i, sent in enumerate(doc.sents):
                first_token = str(sent[0])
                if first_token.isspace():
                    joints.append(first_token)
                    sentences.append(str(sent[1:]))
                else:
                    joints.append(" " if i != 0 else "")
                    sentences.append(str(sent))
            return sentences, joints
        return sentencecizer

    @staticmethod
    def unpack(batch: List[str], sentencecizer: Callable[[str], Tuple[List[str], List[str]]]) -> Tuple[List[str], List[int], List[str]]:
        sizes = [1] * len(batch)
        result = []
        joints = []
        for i, text in enumerate(batch):
            splits, splits_joints = sentencecizer(text)
            result.extend(splits)
            joints.extend(splits_joints)
            sizes[i] = len(splits)
        return result, sizes, joints

    @staticmethod
    def pack(batch: List[str], sizes: List[int], joints: List[str]) -> List[str]:
        result = batch
        for start, size in enumerate(sizes):
            end = start + size
            merge = ""
            for joint, part in zip(joints[start:end], result[start:end]):
                merge += joint + part
            result[start:end] = [merge]
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
