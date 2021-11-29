from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MarianTokenizer

tokenizer1 = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ca-en")
tokenizer2 = MarianTokenizer(vocab="Tokenizer/vocab.json", source_spm="Tokenizer/source.spm",
                            target_spm="Tokenizer/target.spm", source_lang="ca", target_lang="en")

#print(tokenizer1.vocab_size, tokenizer2.vocab_size)

print(len(tokenizer1(["La frase és l'objecte d'estudi de la sintaxi, en totes les seves escoles: generativisme, estructuralisme... Com a element de formació del paràgraf, interessa també a la pragmàtica i a la gramàtica del discurs, ja que la frase dona el context adequat per entendre el significat concret de cada paraula que la forma."]).input_ids[0]))