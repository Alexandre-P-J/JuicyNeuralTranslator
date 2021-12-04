from transformers import MarianMTModel, MarianTokenizer

src_text = ["Bon dia! el meu nom es Alex", "llaminadura",
            "Si us plau, per a evitar malentesos, signeu sempre els vostres missatges al final de la vostra intervenció amb el codi"]

model_name = 'opus-mt-transfer-ca-to-en/checkpoint-58000'
#model_name ="Helsinki-NLP/opus-mt-es-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)

model = MarianMTModel.from_pretrained(model_name)
translated = model.generate(
    **tokenizer(src_text, return_tensors="pt", padding=True))
x = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
print(x)
