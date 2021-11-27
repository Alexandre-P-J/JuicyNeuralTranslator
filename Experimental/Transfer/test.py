# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ca-en")
# model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ca-en")

# def freeze_output_embeddings(model):
#     out_embeddings = model.get_output_embeddings()
#     for p in out_embeddings.parameters():
#         p.requires_grad = False
#     model.set_output_embeddings(out_embeddings)
#     return model

# text = 'La Marta va fer galetes i les vàrem menjar entre tots'

# model = freeze_output_embeddings(model)

# input_ids = tokenizer.encode(text, return_tensors="pt")
# outputs = model.generate(input_ids)
# decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

# print(decoded)


from Data.Books_CaEn import Books_CaEn_Dataset
from Data.GlobalVoices_CaEn import GlobalVoices_CaEn_Dataset
from Data.OpenSubtitles_CaEn import OpenSubtitles_CaEn_Dataset
from Data.QED_CaEn import QED_CaEn_Dataset
import itertools

with Books_CaEn_Dataset() as a, GlobalVoices_CaEn_Dataset() as b, OpenSubtitles_CaEn_Dataset() as c, QED_CaEn_Dataset() as d:
    data = itertools.chain(a.translations(), b.translations(), c.translations(), d.translations())
    data = list(data)
    print(len(data))
    # for d in data.translations():
    #     print(d)
