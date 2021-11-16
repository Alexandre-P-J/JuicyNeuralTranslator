import os, sys, importlib
from typing import List


def smart_split(huggingface_tokenizer, spacy_model, text: str, max_tokens: int):
    doc = spacy_model(text)
    result = []

    sentences = [str(s) for s in doc.sents]
    tmp0 = ""
    tokens0 = []
    for s in sentences:
        tmp1 = tmp0 + (" " if len(tmp0) > 0 else "") + s
        tokens1 = huggingface_tokenizer(tmp1, truncation=True,
                                        max_length=max_tokens+1).input_ids
        if (len(tokens1) > max_tokens) and len(tokens0) > 0:
            result.append(tokens0)
            tmp0 = s
            tokens0 = []
        elif (len(tokens1) > max_tokens):
            result.append(tokens1[0:max_tokens])
            tmp0 = ""
        else:
            tmp0 = tmp1
            tokens0 = tokens1
    if (len(tokens0) > 0):
        result.append(tokens0)
    return result



def load_models() -> List:
    path = os.path.join(os.path.dirname(__file__), os.pardir, "models")
    sys.path.append(path)
    from models import Model
    for file in os.listdir(path):
        if file.endswith(".py"):
            module = os.path.splitext(file)[0]
            importlib.import_module(module)
    return Model