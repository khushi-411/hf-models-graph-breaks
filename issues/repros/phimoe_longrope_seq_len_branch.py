"""PhiMoE rotary embedding: `seq_len = torch.max(position_ids) + 1` then
`if ... and seq_len:` and `long if seq_len > original_max else short` (modeling_phimoe.py)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
import torch
from _pattern import report

position_ids = torch.arange(32)[None].expand(2, -1)
ORIG_MAX, LONG, SHORT = 4096, 1.2, 1.0


def original(position_ids):
    seq_len = torch.max(position_ids) + 1
    mscale = None
    if seq_len:
        mscale = LONG if seq_len > ORIG_MAX else SHORT
    return position_ids.float() * mscale


def fixed(position_ids):
    seq_len = torch.max(position_ids) + 1
    mscale = torch.where(seq_len > ORIG_MAX, LONG, SHORT)   # stays a tensor
    return position_ids.float() * mscale


report("original", original, position_ids)
report("torch.where", fixed, position_ids)
