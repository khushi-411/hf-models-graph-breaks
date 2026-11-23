"""Longformer / LED: `is_index_global_attn.flatten().any().item()` (modeling_longformer.py,
modeling_led.py) forces a .item() and a Python branch on it in every forward."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
import torch
from _pattern import report

attention_mask = torch.ones(2, 32)


def original(attention_mask, x):
    is_index_global_attn = attention_mask > 0
    is_global_attn = is_index_global_attn.flatten().any().item()
    if is_global_attn:                      # two different code paths in the real model
        return x * 2
    return x + 1


def fixed(attention_mask, x):
    # one option: compute both and select (the real fix is to run the global path masked,
    # or use torch.cond); this shows the break goes away once no Python bool is needed
    is_global_attn = (attention_mask > 0).flatten().any()
    return torch.where(is_global_attn, x * 2, x + 1)


x = torch.randn(2, 32)
report("original", original, attention_mask, x)
report("torch.where", fixed, attention_mask, x)
