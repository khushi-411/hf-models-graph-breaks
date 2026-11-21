"""*ForCTC.forward: `if labels.max() >= self.config.vocab_size: raise ValueError` (wav2vec2, hubert, wavlm, ...)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
import torch
from _pattern import report

VOCAB = 32
labels = torch.randint(0, VOCAB, (2, 8))
logits = torch.randn(2, 8, VOCAB)


def original(logits, labels):
    if labels is not None and labels.max() >= VOCAB:
        raise ValueError(f"Label values must be <= vocab_size: {VOCAB}")
    return logits.log_softmax(-1).sum()


def fixed(logits, labels):
    if labels is not None:
        torch._assert_async(labels.max() < VOCAB, "Label values must be < vocab_size")
    return logits.log_softmax(-1).sum()


report("original", original, logits, labels)
report("torch._assert_async", fixed, logits, labels)
