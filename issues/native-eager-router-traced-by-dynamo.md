<!-- Draft for: pytorch/pytorch. Once filed, put the URL in known_issues.yaml and set status: open -->

# Dynamo compiles `torch/_native/registry.py:eager_router` as a user frame (guards on a thread-local and on duck-sized args → recompile)

### System info

- torch: `2.16.0a0+git57f02f8` (source build, CUDA 12.8). `eager_router` is unchanged on main (`41a828ca634`).
- transformers: `5.24.0.dev0` (git `b7fe69cb19`)
- Python 3.15, Linux, CUDA (B200)

### Description

`torch._native` installs a Python "eager router" kernel for some CUDA ops that have native overrides (e.g.
`aten::bmm`, `aten::topk`, `aten::scatter_add`). When a compiled model falls back to eager after a graph
break, the op is dispatched to that Python function. Dynamo's frame hook then treats `eager_router` as a
user frame and compiles it.

Its guards then cause recompiles:

```
Recompiling function eager_router in torch/_native/registry.py:1093
  - 12/0: not hasattr(G['_router_active'], 'on')   # and not getattr(_router_active, "on", False)  # _native/registry.py:1129
Recompiling function eager_router in torch/_native/registry.py:1093
  - 12/1: args[1].size()[2] == args[0].size()[2]   # duck sizing added this equality because these variables had the same size 64
  - 12/0: not hasattr(G['_router_active'], 'on')
```

The first guard is on the `_router_active` thread-local, which the router sets and clears itself.
The second is duck-sizing two operand dims that happen to match (64). Neither has anything to do with the
model.

Seen in DeBERTa (`microsoft/deberta-base` config, 2 layers). DeBERTa graph-breaks on `.item()` in every layer
(see `issues/hf/deberta-att-span-item.md`), so most of each layer runs eagerly. `torch.compile(dynamic=True)`,
Inductor, CUDA, seq 32 / 40 / 56: graphs per size 22 / 1 / 0 (eval) and 21 / 1 / 0 (train). On CPU
(where the router isn't installed) it's 20 / 0 / 0. The single extra recompile is the router frame.

### Expected behavior

`torch/_native/registry.py` should never be compiled as a user frame. It's PyTorch-internal dispatch code,
like the rest of `torch/` in `trace_rules`.

### Possible fix

Add `torch._native` to the Dynamo skip list (`torch/_dynamo/trace_rules.py`), or wrap the router with
`torch._dynamo.disable` / `torch.compiler.disable`. The `is_dynamo_compiling()` shortcut in the router only
needs to run while Dynamo is *tracing a caller*, not while it's compiling the router itself.

Repro: not yet reduced. Needs an op that `torch._native` routes on CUDA, called from eager code that Dynamo is
intercepting (e.g. after a graph break).

<!-- affected-models:manual -->
### Affected models

All runs use `torch.compile(backend="inductor", dynamic=True)` over (batch, seq, audio samples) =
(2, 32, 16000), (3, 40, 17600), (4, 56, 19200), on a model built from the checkpoint's config with reduced
layers. "Graphs per size" is the number of graphs compiled on the 1st / 2nd / 3rd size (anything after the
1st size is a recompile).

| model | config from | scenario · device | graphs per size |
|---|---|---|---|
| deberta | `microsoft/deberta-base` | eval · CUDA | 22 / 1 / 0 |
| deberta | | train · CUDA | 21 / 1 / 0 |

On CPU the same runs are 20 / 0 / 0 (the `_native` overrides are CUDA-only). It's the only model where
`eager_router` shows up. Any model with a graph break before a routed CUDA op (`bmm`, `topk`, `scatter_add`,
...) can hit it.
