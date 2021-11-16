#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import spacy
import numpy as np
import time
import random
import math
import os

from model import Encoder, Decoder, Seq2Seq
from dataset import TextDataset
from inference import translate_sentence, calculate_bleu

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
spacy_es = spacy.load('es_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')
special_tokens = {'pad': '<PAD>', 'eos': '<EOS>',
                  'bos': '<BOS>', 'unk': '<unk>'}
BATCH_SIZE = 128
MAX_TOKENS_PER_SEQ = 80
MIN_TOKEN_FREQ = 3
# Train Settings
LEARNING_RATE = 0.0005

def tokenize_es(text):
    return [tok.text for tok in spacy_es.tokenizer(text.lower())]


def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text.lower())]


def sandwich_tokens(tokenizer, begin_token: str, end_token: str):
    def process(text):
        return [begin_token] + tokenizer(text) + [end_token]
    return process


def get_dataset(root, max_tokens):
    es_tok = sandwich_tokens(
        tokenize_es, special_tokens["bos"], special_tokens["eos"])
    en_tok = sandwich_tokens(
        tokenize_en, special_tokens["bos"], special_tokens["eos"])
    train_data = TextDataset([(os.path.join(root, "train.es"), es_tok),
                              (os.path.join(root, "train.en"), en_tok)], max_tokens)

    valid_data = TextDataset([(os.path.join(root, "val.es"), es_tok),
                              (os.path.join(root, "val.en"), en_tok)], max_tokens)

    test_data = TextDataset([(os.path.join(root, "test.es"), es_tok),
                             (os.path.join(root, "test.en"), en_tok)], max_tokens)
    return train_data, valid_data, test_data


def get_vocabs(train_data, min_freq):
    src_vocab, trg_vocab = train_data.get_vocab(
        min_freq=min_freq, append=tuple(special_tokens.values()))
    src_vocab.set_default_index(src_vocab[special_tokens['unk']])
    trg_vocab.set_default_index(trg_vocab[special_tokens['unk']])
    return src_vocab, trg_vocab


def collate_batch(src_vocab, trg_vocab, special_tokens, device):
    pad = special_tokens["pad"]

    def transform(tokens, vocab):
        return torch.tensor([vocab[token] for token in tokens], device=device)

    def process(batch):
        sources, targets = [], []
        for (src, trg) in batch:
            sources.append(transform(src, src_vocab))
            targets.append(transform(trg, trg_vocab))
        x = pad_sequence(
            sources, padding_value=src_vocab[pad], batch_first=True)
        y = pad_sequence(
            targets, padding_value=trg_vocab[pad], batch_first=True)
        return x, y
    return process


def get_dataloaders(train_data, valid_data, test_data, src_vocab, trg_vocab):
    train_iterator = DataLoader(train_data, batch_size=BATCH_SIZE,
                                collate_fn=collate_batch(src_vocab, trg_vocab, special_tokens, device))
    valid_iterator = DataLoader(valid_data, batch_size=BATCH_SIZE,
                                collate_fn=collate_batch(src_vocab, trg_vocab, special_tokens, device))
    test_iterator = DataLoader(test_data, batch_size=BATCH_SIZE,
                               collate_fn=collate_batch(src_vocab, trg_vocab, special_tokens, device))
    return train_iterator, valid_iterator, test_iterator


def get_model(src_vocab, trg_vocab, special_tokens, max_tokens):
    INPUT_DIM = len(src_vocab)
    OUTPUT_DIM = len(trg_vocab)
    MAX_SENTENCE_LENGTH = max_tokens  # in tokens
    HID_DIM = 256
    ENC_LAYERS = 3
    DEC_LAYERS = 3
    ENC_HEADS = 8
    DEC_HEADS = 8
    ENC_PF_DIM = 512
    DEC_PF_DIM = 512
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1

    enc = Encoder(INPUT_DIM,
                  HID_DIM,
                  ENC_LAYERS,
                  ENC_HEADS,
                  ENC_PF_DIM,
                  ENC_DROPOUT,
                  device,
                  MAX_SENTENCE_LENGTH)

    dec = Decoder(OUTPUT_DIM,
                  HID_DIM,
                  DEC_LAYERS,
                  DEC_HEADS,
                  DEC_PF_DIM,
                  DEC_DROPOUT,
                  device,
                  MAX_SENTENCE_LENGTH)

    model = Seq2Seq(enc, dec, src_vocab[special_tokens["pad"]],
                    trg_vocab[special_tokens["pad"]], device).to(device)
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


# As we want our model to predict the `<eos>` token but not have it be an input into our model we simply slice the `<eos>` token off the end of the sequence. Thus:
#
# $$\begin{align*}
# \text{trg} &= [sos, x_1, x_2, x_3, eos]\\
# \text{trg[:-1]} &= [sos, x_1, x_2, x_3]
# \end{align*}$$
#
# $x_i$ denotes actual target sequence element. We then feed this into the model to get a predicted sequence that should hopefully predict the `<eos>` token:
#
# $$\begin{align*}
# \text{output} &= [y_1, y_2, y_3, eos]
# \end{align*}$$
#
# $y_i$ denotes predicted target sequence element. We then calculate our loss using the original `trg` tensor with the `<sos>` token sliced off the front, leaving the `<eos>` token:
#
# $$\begin{align*}
# \text{output} &= [y_1, y_2, y_3, eos]\\
# \text{trg[1:]} &= [x_1, x_2, x_3, eos]
# \end{align*}$$
def train_step(model, iterator, optimizer, criterion, clip):

    model.train()

    epoch_loss = 0

    for _, batch in enumerate(iterator):

        src = batch[0]
        trg = batch[1]

        optimizer.zero_grad()

        output, _ = model(src, trg[:, :-1])

        # output = [batch size, trg len - 1, output dim]
        # trg = [batch size, trg len]

        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)

        # output = [batch size * trg len - 1, output dim]
        # trg = [batch size * trg len - 1]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss /  len(iterator)


# The evaluation loop is the same as the training loop, just without the gradient calculations and parameter updates.
def evaluate(model, iterator, criterion):

    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for _, batch in enumerate(iterator):

            src = batch[0]
            trg = batch[1]

            output, _ = model(src, trg[:, :-1])

            # output = [batch size, trg len - 1, output dim]
            # trg = [batch size, trg len]

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            # output = [batch size * trg len - 1, output dim]
            # trg = [batch size * trg len - 1]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train_model(model, train_iterator, validation_iterator, optimizer, loss_func):
    N_EPOCHS = 10
    CLIP = 1
    best_valid_loss = float('inf')
    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss = train_step(model, train_iterator,
                                optimizer, loss_func, CLIP)
        valid_loss = evaluate(model, validation_iterator, loss_func)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f'seq2sec_checkpoint-{epoch+1}.pt')

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(
            f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(
            f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
    return model


def interactive(model, src_vocab, trg_vocab, max_tokens):
    es_tok = sandwich_tokens(
        tokenize_es, special_tokens["bos"], special_tokens["eos"])
    while(True):
        line = input()
        tokens = es_tok(line)
        translation, _ = translate_sentence(tokens, special_tokens, src_vocab,
                                            trg_vocab, model, device, max_tokens)
        print(" ".join(translation[:-1]))


def main():
    train_data, valid_data, test_data = get_dataset(
        ".data/wikimedia", MAX_TOKENS_PER_SEQ)

    src_vocab, trg_vocab = get_vocabs(train_data, min_freq=MIN_TOKEN_FREQ)
    print(
        f'Source and target vocabs have {len(src_vocab)} and {len(trg_vocab)} tokens respectivelly')

    train_iterator, valid_iterator, test_iterator = get_dataloaders(
        train_data, valid_data, test_data, src_vocab, trg_vocab)

    model = get_model(src_vocab, trg_vocab, special_tokens, MAX_TOKENS_PER_SEQ)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    model.apply(initialize_weights)

    if os.path.isfile('seq2sec.pt'):
        model.load_state_dict(torch.load('seq2sec.pt'))
    else:
        # The optimizer used in the original Transformer paper uses Adam with a learning rate that has a "warm-up" and then a "cool-down" period. BERT and other Transformer models use Adam with a fixed learning rate, so we will implement that. Check [this](http://nlp.seas.harvard.edu/2018/04/03/attention.html#optimizer) link for more details about the original Transformer's learning rate schedule.
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        # loss function, making sure to ignore losses calculated over `<pad>` tokens.
        criterion = nn.CrossEntropyLoss(
            ignore_index=trg_vocab[special_tokens['pad']])

        model = train_model(model, train_iterator, valid_iterator,
                            optimizer, criterion)

        test_loss = evaluate(model, test_iterator, criterion)
        print(
            f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
        bleu_score = calculate_bleu(test_data, special_tokens,
                                    src_vocab, trg_vocab, model, device, MAX_TOKENS_PER_SEQ)
        print(f'BLEU score = {bleu_score*100:.2f}')

    interactive(model, src_vocab, trg_vocab, MAX_TOKENS_PER_SEQ)


if __name__ == "__main__":
    main()
