from torch.utils.data import IterableDataset
from typing import Tuple, Callable, Sequence
from collections import Counter
import torchtext


class TextDataset(IterableDataset):

    def __init__(self, path_tokenizer_pairs: Sequence[Tuple[str, Callable[[str], Sequence[str]]]],
                 max_line_tokens=float('inf')):
        assert(len(path_tokenizer_pairs) >= 1)
        self.ft_pairs = path_tokenizer_pairs
        self.max_line_tokens = max_line_tokens
        counts = [self.count_lines(p) for (p, _) in self.ft_pairs]
        self.num_examples = min(counts)

    def preprocess(self, pairs):
        def process(line, tokenizer):
            return tokenizer(line.rstrip('\n'))
        files = [open(f, 'r') for (f, _) in pairs]
        tokenizers = [t for (_, t) in pairs]
        for lines in zip(*files):
            tokens = [process(l, t) for (l, t) in zip(lines, tokenizers)]
            sizes = [len(ts) for ts in tokens]
            if max(sizes) <= self.max_line_tokens:
                yield tuple(tokens)
        for f in files:
            f.close()

    def __iter__(self):
        return self.preprocess(self.ft_pairs)

    def __len__(self):
        return self.num_examples

    @classmethod
    def count_lines(cls, path):
        count = 0
        with open(path, 'r') as f:
            for _ in f:
                count += 1
        return count
