from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
  
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ROMANCE")

model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ROMANCE")

text = '>>es_ES<<Deformation is probably the most important benefit of separating the base mesh from fine detail'
input_ids = tokenizer.encode(text, return_tensors="pt")
outputs = model.generate(input_ids)
decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(decoded)