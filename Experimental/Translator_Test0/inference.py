#!/usr/bin/env python
# coding: utf-8

import torch
from torchtext.data.metrics import bleu_score

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# The steps taken to translate a sentence are:
# - append the `<sos>` and `<eos>` tokens
# - numericalize the source sentence
# - convert it to a tensor and add a batch dimension
# - create the source sentence mask
# - feed the source sentence and mask into the encoder
# - create a list to hold the output sentence, initialized with an `<sos>` token
# - while we have not hit a maximum length
#   - convert the current output sentence prediction into a tensor with a batch dimension
#   - create a target sentence mask
#   - place the current output, encoder output and both masks into the decoder
#   - get next output token prediction from decoder along with attention
#   - add prediction to current output sentence prediction
#   - break if the prediction was an `<eos>` token
# - convert the output sentence from indexes to tokens
# - return the output sentence (with the `<sos>` token removed) and the attention from the last layer
def translate_sentence(tokens_ids, trg_vocab, model, device, max_len):
    model.eval()

    src_tensor = torch.LongTensor(tokens_ids).unsqueeze(0).to(device)

    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [trg_vocab.bos_id()]

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():
            output, attention = model.decoder(
                trg_tensor, enc_src, trg_mask, src_mask)

        pred_token = output.argmax(2)[:, -1].item()

        trg_indexes.append(pred_token)

        if pred_token == trg_vocab.eos_id():
            break

    return trg_indexes, attention
    # trg_tokens = [trg_vocab.lookup_token(i) for i in trg_indexes]

    # return trg_tokens[1:], attention


# # Display the attention over the source sentence for each step of the decoding. As this model has 8 heads our model we can view the attention for each of the heads.
# def display_attention(sentence, translation, attention, n_heads=8, n_rows=4, n_cols=2):

#     assert n_rows * n_cols == n_heads

#     fig = plt.figure(figsize=(15, 25))

#     for i in range(n_heads):

#         ax = fig.add_subplot(n_rows, n_cols, i + 1)

#         _attention = attention.squeeze(0)[i].cpu().detach().numpy()

#         cax = ax.matshow(_attention, cmap='bone')

#         ax.tick_params(labelsize=12)
#         ax.set_xticklabels([''] + ['<sos>'] + [t.lower() for t in sentence] + ['<eos>'],
#                            rotation=45)
#         ax.set_yticklabels([''] + translation)

#         ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
#         ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

#     plt.show()
#     plt.close()


def calculate_bleu(data, trg_vocab, model, device, max_len):
    expected_trgs = []
    pred_trgs = []

    for tokens in data:

        src = tokens[0]
        trg = tokens[1]

        pred_trg, _ = translate_sentence(src, trg_vocab,
                                         model, device, max_len)

        # cut off <bos> and <eos> tokens. convert to list and store in list
        pred_trg = pred_trg[1:-1]
        pred_trg = [trg_vocab.IdToPiece(i) for i in pred_trg]
        pred_trgs.append(pred_trg)

        expected_trg = trg[1:-1]
        expected_trg = [trg_vocab.IdToPiece(i) for i in trg]
        expected_trgs.append([expected_trg])

    return bleu_score(pred_trgs, expected_trgs)
