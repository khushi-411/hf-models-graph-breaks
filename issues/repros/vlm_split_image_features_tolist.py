"""get_image_features (hy_v4_vl, granite_docling, lfm3_vl, inkling2, dots3_note, qwen2_vl):
`features.split((grid_thw.prod(-1) // merge**2).tolist())` only to return per-image tuples; the
language-model path concatenates them again right after. `.tolist()` on a tensor is a graph break."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
import torch
from _pattern import report

MERGE = 2
grid_thw = torch.tensor([[1, 8, 8], [1, 8, 8]])
features = torch.randn(32, 16)   # 2 images x 16 merged tokens


def original(features, grid_thw):
    split_sizes = (grid_thw.prod(-1) // MERGE**2).tolist()
    per_image = features.split(split_sizes)
    return torch.cat(per_image, dim=0) * 2           # what the LM path does next


def fixed(features, grid_thw):
    return features * 2                                # keep the concatenated tensor


report("split(...tolist()) then cat", original, features, grid_thw)
report("no split", fixed, features, grid_thw)
