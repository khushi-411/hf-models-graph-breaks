"""get_placeholder_mask (inkling2, dots3_note, omniasr): the token/feature count check is written as
`inputs_embeds[special_mask].numel() == features.numel()`. Boolean indexing has a data-dependent
output shape, so this check alone is a graph break. Counting with `.sum()` is equivalent."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
import torch
from _pattern import report

IMAGE_TOKEN, HIDDEN = 7, 16
input_ids = torch.randint(10, 100, (2, 32)); input_ids[:, :8] = IMAGE_TOKEN
inputs_embeds = torch.randn(2, 32, HIDDEN)
features = torch.randn(16, HIDDEN)


def original(input_ids, inputs_embeds, features):
    special_mask = input_ids == IMAGE_TOKEN
    special_mask = special_mask.unsqueeze(-1).expand_as(inputs_embeds)
    torch._check_tensor_all(torch.tensor(inputs_embeds[special_mask].numel() == features.numel()))
    return inputs_embeds.masked_scatter(special_mask, features)


def fixed(input_ids, inputs_embeds, features):
    special_mask = input_ids == IMAGE_TOKEN
    n_tokens = special_mask.sum()
    torch._assert_async(n_tokens * inputs_embeds.shape[-1] == features.numel())
    special_mask = special_mask.unsqueeze(-1).expand_as(inputs_embeds)
    return inputs_embeds.masked_scatter(special_mask, features)


report("original (bool-index numel)", original, input_ids, inputs_embeds, features)
report("count with sum()", fixed, input_ids, inputs_embeds, features)
torch.testing.assert_close(original(input_ids, inputs_embeds, features), fixed(input_ids, inputs_embeds, features))
print("outputs match")
