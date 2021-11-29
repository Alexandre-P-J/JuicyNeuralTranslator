from tokenizers import SentencePieceUnigramTokenizer
from transformers import PreTrainedTokenizerFast

tokenizer = SentencePieceUnigramTokenizer()
tokenizer.train_from_iterator(iterator=iter([]), vocab_size=222)

tokenizer.save("out.json")

transformer_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer
)