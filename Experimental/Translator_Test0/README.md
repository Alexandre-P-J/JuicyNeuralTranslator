# Translation with transformers

As of January 2020, Transformers are the dominant architecture in NLP and are used to achieve state-of-the-art results for many tasks and it appears as if they will be for the near future.

## Implementation details
Slightly modified version of the Transformer model from the [Attention is all you need](https://arxiv.org/abs/1706.03762) paper.
The differences between the implementation in this notebook and the paper are:

- Learned positional encoding instead of a static one
- Standard Adam optimizer with a static learning rate instead of one with warm-up and cool-down steps
- No label smoothing

## Resources
[Attention is all you need](https://arxiv.org/abs/1706.03762)

[Transformers attention in disguise](https://www.mihaileric.com/posts/transformers-attention-in-disguise/)

[Illustrated transformer](https://jalammar.github.io/illustrated-transformer/)

[Attention](http://nlp.seas.harvard.edu/2018/04/03/attention.html)