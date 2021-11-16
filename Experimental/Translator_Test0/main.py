#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import sentencepiece as spm

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

BATCH_SIZE = 128
MAX_TOKENS_PER_SEQ = 86
MAX_VOCAB_SIZE = 10000
# Train Settings
LEARNING_RATE = 0.0005
# Model Settings
HID_DIM = 256
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

def get_tokenizer(sentencepiece_model):
    def process(text):
        tokens = sentencepiece_model.Encode(text, out_type=str, enable_sampling=False,
                                            alpha=0.1, nbest_size=-1, add_bos=True, add_eos=True)
        ids = [sentencepiece_model.PieceToId(piece) for piece in tokens]
        return ids
    return process

# def tokenize_es(text):
#     return [tok.text for tok in spacy_es.tokenizer(text.lower())]


# def tokenize_en(text):
#     return [tok.text for tok in spacy_en.tokenizer(text.lower())]


# def sandwich_tokens(tokenizer, begin_token: str, end_token: str):
#     def process(text):
#         return [begin_token] + tokenizer(text) + [end_token]
#     return process


def get_dataset(root, max_tokens, src_vocab, trg_vocab):
    es_tok = get_tokenizer(src_vocab)
    en_tok = get_tokenizer(trg_vocab)
    train_data = TextDataset([(os.path.join(root, "train.es"), es_tok),
                              (os.path.join(root, "train.en"), en_tok)], max_tokens)

    valid_data = TextDataset([(os.path.join(root, "val.es"), es_tok),
                              (os.path.join(root, "val.en"), en_tok)], max_tokens)

    test_data = TextDataset([(os.path.join(root, "test.es"), es_tok),
                             (os.path.join(root, "test.en"), en_tok)], max_tokens)
    return train_data, valid_data, test_data


def get_vocabs(src_path, trg_path, size):
    spm.SentencePieceTrainer.train(input=src_path, model_prefix='src_sentence_piece', vocab_size=size,
                                   bos_piece="<bos>", eos_piece="<eos>", unk_piece="<unk>", pad_piece="<pad>", pad_id=3)
    src_vocab = spm.SentencePieceProcessor(
        model_file='src_sentence_piece.model')
    spm.SentencePieceTrainer.train(input=trg_path, model_prefix='trg_sentence_piece', vocab_size=size,
                                   bos_piece="<bos>", eos_piece="<eos>", unk_piece="<unk>", pad_piece="<pad>", pad_id=3)
    trg_vocab = spm.SentencePieceProcessor(
        model_file='trg_sentence_piece.model')
    return src_vocab, trg_vocab


def collate_batch(src_vocab, trg_vocab, device):
    def process(batch):
        sources, targets = [], []
        for (src, trg) in batch:
            sources.append(torch.tensor(src, device=device))
            targets.append(torch.tensor(trg, device=device))
        x = pad_sequence(
            sources, padding_value=src_vocab.pad_id(), batch_first=True)
        y = pad_sequence(
            targets, padding_value=trg_vocab.pad_id(), batch_first=True)
        return x, y
    return process


def get_dataloaders(train_data, valid_data, test_data, src_vocab, trg_vocab):
    collate_func = collate_batch(src_vocab, trg_vocab, device)
    train_iterator = DataLoader(train_data, batch_size=BATCH_SIZE,
                                collate_fn=collate_func)
    valid_iterator = DataLoader(valid_data, batch_size=BATCH_SIZE,
                                collate_fn=collate_func)
    test_iterator = DataLoader(test_data, batch_size=BATCH_SIZE,
                               collate_fn=collate_func)
    return train_iterator, valid_iterator, test_iterator


def get_model(src_vocab, trg_vocab):
    INPUT_DIM = len(src_vocab)
    OUTPUT_DIM = len(trg_vocab)
    
    enc = Encoder(INPUT_DIM,
                  HID_DIM,
                  ENC_LAYERS,
                  ENC_HEADS,
                  ENC_PF_DIM,
                  ENC_DROPOUT,
                  device,
                  MAX_TOKENS_PER_SEQ)

    dec = Decoder(OUTPUT_DIM,
                  HID_DIM,
                  DEC_LAYERS,
                  DEC_HEADS,
                  DEC_PF_DIM,
                  DEC_DROPOUT,
                  device,
                  MAX_TOKENS_PER_SEQ)

    model = Seq2Seq(enc, dec, src_vocab.pad_id(),
                    trg_vocab.pad_id(), device).to(device)
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

    return epoch_loss / len(iterator)


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
    N_EPOCHS = 100
    CLIP = 1
    best_valid_loss = float('inf')
    times_no_improve = 0
    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss = train_step(model, train_iterator,
                                optimizer, loss_func, CLIP)
        valid_loss = evaluate(model, validation_iterator, loss_func)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f'seq2sec.pt')
            times_no_improve = 0
        else:
            times_no_improve += 1
        if times_no_improve >= 3:
            print(f"No improvement for {times_no_improve}. Training stopped.")
            break

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(
            f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(
            f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
    model.load_state_dict(torch.load('seq2sec.pt'))
    return model


def interactive(model, src_vocab, trg_vocab, max_tokens):
    es_tok = get_tokenizer(src_vocab)
    while(True):
        line = input()
        tokens_ids = es_tok(line)
        translation, _ = translate_sentence(tokens_ids, trg_vocab, model,
                                            device, max_tokens)
        print(trg_vocab.Decode(translation))


def main():
    src_vocab, trg_vocab = get_vocabs(
        ".data/wikimedia/train.es", ".data/wikimedia/train.en", MAX_VOCAB_SIZE)
    print(
        f'Source and target vocabs have {len(src_vocab)} and {len(trg_vocab)} tokens respectivelly')

    train_data, valid_data, test_data = get_dataset(
        ".data/wikimedia", MAX_TOKENS_PER_SEQ, src_vocab, trg_vocab)

    train_iterator, valid_iterator, test_iterator = get_dataloaders(
        train_data, valid_data, test_data, src_vocab, trg_vocab)

    model = get_model(src_vocab, trg_vocab)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    model.apply(initialize_weights)

    if os.path.isfile('seq2sec.pt'):
        model.load_state_dict(torch.load('seq2sec.pt'))
    else:
        # The optimizer used in the original Transformer paper uses Adam with a learning rate that has a "warm-up" and then a "cool-down" period. BERT and other Transformer models use Adam with a fixed learning rate, so we will implement that. Check [this](http://nlp.seas.harvard.edu/2018/04/03/attention.html#optimizer) link for more details about the original Transformer's learning rate schedule.
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        # loss function, making sure to ignore losses calculated over `<pad>` tokens.
        criterion = nn.CrossEntropyLoss(ignore_index=trg_vocab.pad_id())

        model = train_model(model, train_iterator, valid_iterator,
                            optimizer, criterion)
        
        test_loss = evaluate(model, test_iterator, criterion)
        print(
            f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
    bleu_score = calculate_bleu(
                test_data, trg_vocab, model, device, MAX_TOKENS_PER_SEQ)
    print(f'BLEU score = {bleu_score*100:.2f}')

    interactive(model, src_vocab, trg_vocab, MAX_TOKENS_PER_SEQ)


if __name__ == "__main__":
    main()
