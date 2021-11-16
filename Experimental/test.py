import spacy
# import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from typing import List

tokenizer = AutoTokenizer.from_pretrained(
     "Helsinki-NLP/opus-mt-en-ROMANCE")


long_text = "Depending on your specified model and input sentence, the difference lies in the additionally encoded information, specifically the input mask. Since you are feeding in two sentences at a time, BERT (and likely other model variants), expect some form of masking, which allows the model to discern between the two sequences."
super_long_text = long_text + " " + long_text
xxx = super_long_text + " " + super_long_text + " " + super_long_text + " " + super_long_text + " " + super_long_text
test = "On offering to help the blind man, the man who then stole his car, had not, at that precise moment, had any evil intention, quite the contrary, what he did was nothing more than obey those feelings of generosity and altruism which, as everyone knows, are the two best traits of human nature and to be found in much more hardened criminals than this one, a simple car-thief without any hope of advancing in his profession, exploited by the real owners of this enterprise, for it is they who take advantage of the needs of the poor."

nlp = spacy.load("en_core_web_sm")


def smart_split(huggingface_tokenizer, spacy_model, text: str, max_tokens: int):
    doc = spacy_model(text)
    result = []

    sentences = [str(s) for s in doc.sents]
    tmp0 = ""
    tokens0 = []
    for s in sentences:
        tmp1 = tmp0 + s
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


result = smart_split(tokenizer, nlp, long_text, 20)
out = tokenizer.batch_decode(result, skip_special_tokens=True)

print(len(result), result)

print(out)

