"""dynamic_frequency_update (modeling_rope_utils.py): `if seq_len > max_seq_len_cached:` where
seq_len = torch.max(position_ids) + 1 is a tensor -> data-dependent branch every forward."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
import torch
from _pattern import report

position_ids = torch.arange(32)[None].expand(2, -1)
inv_freq = torch.randn(8)
MAX_CACHED = 2048


def original(position_ids, inv_freq):
    seq_len = torch.max(position_ids) + 1
    if seq_len > MAX_CACHED:            # growth: recompute inv_freq
        inv_freq = inv_freq * 0.5
    return position_ids[..., None].float() * inv_freq


def fixed(position_ids, inv_freq):
    # the sequence length is known from the shape when positions are contiguous
    seq_len = position_ids.shape[-1]
    if seq_len > MAX_CACHED:
        inv_freq = inv_freq * 0.5
    return position_ids[..., None].float() * inv_freq


report("original (tensor seq_len)", original, position_ids, inv_freq)
report("shape-derived seq_len", fixed, position_ids, inv_freq)
