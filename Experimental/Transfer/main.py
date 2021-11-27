import numpy as np
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer
from datasets import Dataset, load_metric
import random
from Data.Books_CaEn import Books_CaEn_Dataset
from Data.GlobalVoices_CaEn import GlobalVoices_CaEn_Dataset
from Data.OpenSubtitles_CaEn import OpenSubtitles_CaEn_Dataset
from Data.QED_CaEn import QED_CaEn_Dataset
import itertools

random.seed(1234)

base_model_name = "Helsinki-NLP/opus-mt-es-en"
bleu_metric = load_metric("sacrebleu")
chrf_metric = load_metric("chrf")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
prefix = ""
max_input_length = 128
max_target_length = 128
source_lang = "ca"
target_lang = "en"
batch_size = 16
num_epoch = 3


def freeze_output_embeddings(model):
    out_embeddings = model.get_output_embeddings()
    for p in out_embeddings.parameters():
        p.requires_grad = False
    model.set_output_embeddings(out_embeddings)
    return model

def load_splits(shuffle=True, train_p=0.6, val_p=0.2, test_p=0.2):
    with Books_CaEn_Dataset() as a, GlobalVoices_CaEn_Dataset() as b, OpenSubtitles_CaEn_Dataset() as c, QED_CaEn_Dataset() as d:
        data = itertools.chain(
            a.translations(), b.translations(), c.translations(), d.translations())
        data = list(data)
        if shuffle:
            random.shuffle(data)
        total = len(data)
        train_interval = (0, int(total*train_p))
        val_interval = (train_interval[1],
                        train_interval[1] + int(total*val_p))
        test_interval = (val_interval[1], val_interval[1] + int(total*test_p))
        train = Dataset.from_dict(
            {"translation": data[train_interval[0]:train_interval[1]]})
        val = Dataset.from_dict(
            {"translation": data[val_interval[0]:val_interval[1]]})
        test = Dataset.from_dict(
            {"translation": data[test_interval[0]:test_interval[1]]})
        return train, val, test


def preprocess_function(examples):
    inputs = [prefix + ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]
    model_inputs = tokenizer(
        inputs, max_length=max_input_length, truncation=True)
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets, max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(
        decoded_preds, decoded_labels)
    # BLEU
    result = bleu_metric.compute(predictions=decoded_preds,
                                 references=decoded_labels)
    result = {"BLEU": result["score"]}
    # Character F-score
    chrf = chrf_metric.compute(predictions=decoded_preds,
                               references=decoded_labels)
    result["chr-F"] = chrf["score"]
    # Lenghts
    prediction_lens = [np.count_nonzero(
        pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


train, val, test = load_splits(shuffle=True,
                               train_p=0.6, val_p=0.2, test_p=0.2)


train_tokenized = train.map(preprocess_function, batched=True)
val_tokenized = val.map(preprocess_function, batched=True)
test_tokenized = test.map(preprocess_function, batched=True)


args = Seq2SeqTrainingArguments(
    f"opus-mt-transfer-{source_lang}-to-{target_lang}",
    evaluation_strategy="epoch",
    learning_rate=2e-3,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=5,
    num_train_epochs=num_epoch,
    predict_with_generate=True,
)

model = freeze_output_embeddings(model)
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# trainer.train(base_model_name) # start from interrupted train
trainer.train()

# # TEST
# _, _, metrics = trainer.predict(test_dataset=test_tokenized)
# print(f"TEST:\n{metrics}")

# # EXTRA TEST: METRICS WITH TATOEBA
# _, _, tatoeba_test = load_splits(
#     TatoebaTest_EnEs_Dataset, train_p=0, val_p=0, test_p=1)
# tatoeba_test_tokenized = tatoeba_test.map(preprocess_function, batched=True)
# _, _, metrics = trainer.predict(test_dataset=tatoeba_test_tokenized)
# print(f"TATOEBA:\n{metrics}")
