<!-- Draft for: huggingface/transformers. Once filed, put the URL in known_issues.yaml and set status: open -->

# `torch.compile`: Inkling2Loop (Inkling 2 Mini) `copy.deepcopy`s cache layers inside `forward`, so the whole text model falls back to eager

### System info

- transformers: `5.24.0.dev0` (main, git `046bdb8cc2`, "Add Inkling2 Mini (#51050)")
- torch: `2.16.0a0+git57f02f8` (source build)
- Python 3.15, Linux, CPU

### Description

`Inkling2LoopTextModel.forward` (`modeling_inkling2_loop.py:1357`) adds the cache layers for the extra
loop passes the first time it sees a cache:

```python
if cache is not None and len(cache.layers) == self.config.num_hidden_layers:
    # Append extra caches for loop layers
    loop_cache_layers = cache.layers[loop_start:loop_end]
    cache.layers += [copy.deepcopy(layer) for layer in loop_cache_layers * (self.config.loop_num_passes - 1)]
```

Dynamo does not support `copy.deepcopy` on tensors (`Attempted to copy.deepcopy a tensor`, gb4198). The
break happens inside a list comprehension, which Dynamo cannot resume from here ("Unsafe value live across
comprehension graph break", gb1891), so it skips the whole `forward` frame and runs it in eager. Every decoder
layer is then compiled as its own graph.

This hits every call that builds a cache: generation, `use_cache=True` and training, since `use_cache`
defaults to `True`. Plain `use_cache=False` forwards are not affected.

When this branch runs, the layers being copied were created a few lines earlier by
`DynamicCache(config=self.config)` and are still empty. So the copies can be built directly, without
`deepcopy`.

### Numbers

From the graph-break harness: model built from its config with random weights, 7 layers (layers 3-5 looped
3 times), one `torch._dynamo.explain` call per scenario.

| scenario | graphs | breaks |
|---|---|---|
| inkling2_mini eval_forward (no cache) | 1 | 0 |
| inkling2_mini eval_with_cache | 20 | 19 |
| inkling2_mini train_forward | 20 | 19 |
| inkling2_mini eval_with_cache, cache passed in with the loop layers already added | 1 | 0 |

With `torch.compile(dynamic=True)`, training at three sizes compiles 22 graphs on the first size, because
each decoder layer call becomes its own frame. It then recompiles once, when duck sizing ties
`block_states._base.size(2)` to the batch size in `Inkling2LoopMessageAttentionResidual.forward`. With the loop
cache layers passed in, the same run is 1 frame and 0 recompiles. That also needs the PyTorch conv-backward fix,
which is tracked separately.

`inkling2_mini_vlm` gets the same 17 extra breaks in eval_with_cache and train_forward, on top of the
vision-tower breaks it shares with inkling2_vlm.

### Reproduction

Pass a cache that already has the loop layers, so the `deepcopy` branch is skipped:

```python
cache = DynamicCache(config=cfg)
s, e = cfg.loop_layer_index_start, cfg.loop_layer_index_end
for _ in range(cfg.loop_num_passes - 1):
    cache.layers += DynamicCache(config=cfg).layers[s:e]
torch._dynamo.explain(model)(input_ids=ids, attention_mask=mask, cache=cache)  # 1 graph, 0 breaks
```

Without `cache=...`: 20 graphs, 19 breaks.

### Possible fix

Create the extra cache layers when the cache is created, so `forward` never copies them: for example, have
the cache derive its layer list from `total_num_hidden_layers` / the looped layer types. Alternatively, build
fresh layers instead of copying, e.g. `DynamicCache(config=self.config).layers[loop_start:loop_end]` once per
extra pass. The second option is untested inside a compiled region.
