import sentencepiece as spm


# spm.SentencePieceTrainer.train(input='.data/wikimedia/train.en', model_prefix='sentence_piece', vocab_size=20000,
#                                bos_piece="<bos>", eos_piece="<eos>", unk_piece="<unk>", pad_piece="<pad>")
# s = spm.SentencePieceProcessor(model_file='sentence_piece.model')
# t = s.encode('hungry and tired of vegetables, australopitecus',
#              out_type=str, enable_sampling=False, alpha=0.1, nbest_size=-1)
# print(t, len(t))
# t = s.Encode('hungry and tired Of Vegetables, australopitecus',
#              out_type=str, enable_sampling=False, alpha=0.1, nbest_size=-1, add_bos=True, add_eos=True)
# print(t, len(t))

# ids = [s.PieceToId(piece) for piece in t]
# print(ids, len(ids))

# result = s.Decode(ids[:-1])
# print(result)


# print(s.bos_id())

s = spm.SentencePieceProcessor(model_file='src_sentence_piece.model')
print(s.Encode('soy el más alto de mis hermanos',
              out_type=str, enable_sampling=False, alpha=0.1, nbest_size=-1, add_bos=True, add_eos=True))