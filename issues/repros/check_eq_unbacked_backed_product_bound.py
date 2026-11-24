"""torch._check(u0 == 32 * s0) on an unbacked numel fails with
AssertionError('tgt_bound=VR[64, int_oo] not a subset of src_bound=VR[64, 9223372036854775806]').

u0 = numel of a boolean-indexed tensor (unbacked, size-like, upper bound sys.maxsize - 1).
s0 = a backed batch size (range [2, int_oo]). The _check asks ShapeEnv to replace u0 with 32*s0, whose upper
bound is int_oo, and _set_replacement refuses it because int_oo is not <= sys.maxsize - 1.
Seen in inkling2_vlm / inkling2_mini_vlm (transformers get_placeholder_mask -> torch_compilable_check) under
dynamic=True + capture_dynamic_output_shape_ops.
"""
import torch

torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True


def f(embeds, ids):
    mask = (ids == 7).unsqueeze(-1).expand_as(embeds)
    n = embeds[mask].numel()                      # unbacked u0
    torch._check(n == embeds.shape[0] * 32)       # 32 = image tokens per row * hidden, as in the model
    return embeds.sum()


ids = torch.full((2, 4), 7)
embeds = torch.randn(2, 4, 8)                     # 4 tokens * 8 hidden = 32 per row
print("eager:", f(embeds, ids))
print("compiled:", torch.compile(f, dynamic=True, fullgraph=True)(embeds, ids))
