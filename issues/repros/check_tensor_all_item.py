"""torch._check_tensor_all(cond) graph-breaks under torch.compile (default flags).

Its Python implementation does `cond._is_all_true().item()`, and Tensor.item() is
unsupported with capture_scalar_outputs=False. torch._check (python bool / SymBool)
compiles fine, so the tensor variant is the odd one out.

transformers' `torch_compilable_check` (utils/import_utils.py) uses it under tracing for
the image-placeholder check in every LLaVA-style VLM. It is the *only* break in llava and
gemma3 (VLM): fixing it takes them from 4 graphs to 1. Also hit by internvl, mistral3,
smolvlm, qwen2_5_vl, qwen3_vl.

Expected: 1 graph (a runtime assert, like torch._check / torch._assert_async).
Actual:   2 graphs, 1 break: "Unsupported Tensor.item() call with capture_scalar_outputs=False".
"""

import torch


def with_check(image_features, image_token_mask):
    torch._check_tensor_all(image_token_mask.sum() * image_features.shape[-1] == image_features.numel())
    return image_features * 2


def with_python_check(image_features, image_token_mask):
    torch._check(image_token_mask.shape[0] * image_features.shape[-1] == image_features.numel())
    return image_features * 2


feats, mask = torch.randn(4, 8), torch.ones(4, dtype=torch.bool)
for fn in (with_python_check, with_check):
    torch._dynamo.reset()
    e = torch._dynamo.explain(fn)(feats, mask)
    print(f"{fn.__name__:18s} graphs={e.graph_count} breaks={e.graph_break_count}",
          [str(r.reason).splitlines()[0] for r in e.break_reasons])
