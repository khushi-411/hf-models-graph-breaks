"""FSMT make_padding_mask: `if not padding_mask.any(): padding_mask = None` (modeling_fsmt.py)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
import torch
from _pattern import report

input_ids = torch.randint(2, 100, (2, 32))  # no pad tokens, like most batches


def make_padding_mask_original(input_ids, padding_idx=1):
    padding_mask = input_ids.eq(padding_idx)
    if not padding_mask.any():
        padding_mask = None
    return padding_mask


def make_padding_mask_fixed(input_ids, padding_idx=1):
    return input_ids.eq(padding_idx)   # an all-False mask masks nothing


def attn(scores, mask):
    if mask is not None:
        scores = scores.masked_fill(mask[:, None, :], float("-inf"))
    return scores.softmax(-1)


def original(input_ids, scores):
    return attn(scores, make_padding_mask_original(input_ids))


def fixed(input_ids, scores):
    return attn(scores, make_padding_mask_fixed(input_ids))


scores = torch.randn(2, 32, 32)
report("original", original, input_ids, scores)
report("always return mask", fixed, input_ids, scores)
torch.testing.assert_close(original(input_ids, scores), fixed(input_ids, scores))
print("outputs match")
