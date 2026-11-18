<!-- Draft for: huggingface/transformers. Once filed, put the URL in known_issues.yaml and set status: open -->

# `torch.compile`: SwitchTransformers sparse MLP graph-breaks in every sparse layer (`nonzero()` expert loop)

### System info

- transformers: `5.24.0.dev0` (main, git `b7fe69cb19`)
- torch: `2.16.0a0+git57f02f8` (source build)
- Python 3.15, Linux; checked on CPU and on CUDA (B200)

### Description

The sparse MLP loops over the experts that received tokens
(`modeling_switch_transformers.py:174`):

```python
final_hidden_states = torch.zeros_like(hidden_states)
expert_mask = selected_experts.permute(2, 1, 0)
expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
for expert_idx in expert_hit:
    ...
```

`nonzero()` has a data-dependent output shape (`Dynamic shape operator`) and the Python loop over it
can't be traced, so every sparse layer splits the graph, in **eval and train**.

The router also does `self.classifier = self.classifier.to(self.dtype)` inside `forward`. With the default
`router_dtype="float32"` and a float32 model that is a no-op, but whenever the model runs in another dtype
it rebuilds `nn.Parameter`s inside forward, which Dynamo breaks on by design (pytorch#157452). The harness
run records that break as well (`torch/nn/modules/module.py`, "Attempted to use `torch.nn.Parameter()`
constructor").

Harness (config-only, reduced layers): `eval_forward` 18 graphs / 17 breaks, `train_forward` 19 / 18.

### Reproduction

`issues/repros/switch_router_module_to.py` (real `SwitchTransformersModel`, 2+2 layers, 4 experts, all
layers sparse, eval):

```
router_dtype: float32
graphs=13 breaks=12          (14 / 13 on another run: the count depends on which experts the
   - Dynamic shape operator   random weights route tokens to)
```

### Possible fix

- Loop over all experts (`for expert_idx in range(self.num_experts)`) and let empty experts process zero
  tokens, or use a grouped-GEMM / dense-dispatch path when compiling, as the newer MoE implementations in
  the library do.
- Cast the router classifier once (at init / after loading) instead of reassigning it in `forward`.

### Affected

switch_transformers (eval_forward, train_forward).
