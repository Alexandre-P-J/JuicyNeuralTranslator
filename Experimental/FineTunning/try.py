from transformers import MarianMTModel, MarianTokenizer

src_text = ["We must mitigate poverty"]

model_name = 'opus-mt-finetuned-en-to-es/checkpoint-31000'
tokenizer = MarianTokenizer.from_pretrained(model_name)

model = MarianMTModel.from_pretrained(model_name)
translated = model.generate(
    **tokenizer(src_text, return_tensors="pt", padding=True))
x = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
print(x)