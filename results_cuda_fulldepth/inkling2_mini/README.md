# inkling2_mini — graph breaks

- **HF id:** `thinkingmachines/Inkling-2-Mini`
- **Class:** `AutoModelForCausalLM`
- **Config overrides:** `{'num_hidden_layers': 7, 'layer_types': ['hybrid_sliding', 'hybrid_sliding', 'hybrid_sliding', 'hybrid_sliding', 'hybrid_sliding', 'hybrid', 'hybrid_sliding'], 'n_layers_per_attn_res_block': 3, 'loop_layer_index_start': 3, 'loop_layer_index_end': 6, 'loop_num_passes': 3, 'loop_gate_rank': 8, 'hidden_size': 512, 'num_attention_heads': 8, 'num_key_value_heads': 2, 'head_dim': 64, 'swa_num_attention_heads': 8, 'swa_num_key_value_heads': 2, 'swa_head_dim': 64, 'sliding_window_size': 16, 'rel_extent': 16, 'moe_intermediate_size': 256, 'n_routed_experts': 8, 'num_experts_per_tok': 2, 'vocab_size': 32000, 'unpadded_vocab_size': 31000, 'quantization_config': None, 'dtype': 'float32'}`
- **torch:** `2.16.0a0+gitf29b6d5`  |  **transformers:** `5.24.0.dev0` (git `046bdb8cc2`)
- **Notes:** Text model of Inkling 2 Mini (model_type inkling2_loop). Inkling-2 layers with a block of layers looped several times and message passing between passes.

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ok | 1 | 0 | 0 |
| eval_with_cache | ok | 20 | 19 | 2 |
| train_forward | ok | 20 | 19 | 2 |

## eval_forward

No graph breaks. ✅

## eval_with_cache

### 1. Attempted to copy.deepcopy a tensor

- **Location:** `python/copy.py:137`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_FUNCTION_EX: a variadic function call, e.g. f(*args):

Attempted to copy.deepcopy a tensor
  Explanation: Dynamo does not support copy.deepcopy() on tensors.
  Hint: Avoid calling copy.deepcopy() on tensors inside compiled regions.

  Developer debug context: copy.deepcopy(TensorVariable())

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb4198.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb4198.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling2_loop/modeling_inkling2_loop.py:1523 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling2_loop/modeling_inkling2_loop.py:1357 in forward
```
</details>

### 2. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/inkling2_loop/modeling_inkling2_loop.py:1357`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Unsafe value live across comprehension graph break
  Explanation: A context manager or local generator cannot be reconstructed safely after a comprehension graph break, so the frame falls back to eager.
  Hint: Close the generator first, or move the comprehension into a helper.

  Developer debug context: LocalGeneratorObjectVariable(__iter__), frame skipped: forward (/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/inkling2_loop/modeling_inkling2_loop.py line 1304)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb1891.html

Stack variable source attribution:
  UserDefinedObjectVariable(DynamicCache) originated from:
  File "/admin/home/khushi_agrawal/Documents/pytorch/torch/_dynamo/polyfills/__init__.py", line 472
        obj = cls.__new__(cls, *args, **kwargs)
              ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^
  ListVariable(length=24) originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/cache_utils.py", line 2065
                layers = [
                         ^
                ...<2 lines>...
                ]
                ^
  NullVariable originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/inkling2_loop/modeling_inkling2_loop.py", line 1357
                cache.layers += [copy.deepcopy(layer) for layer in loop_cache_layers * (self.config.loop_num_passes - 1)]
                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  ListVariable(length=24) originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/inkling2_loop/modeling_inkling2_loop.py", line 1357
                cache.layers += [copy.deepcopy(layer) for layer in loop_cache_layers * (self.config.loop_num_passes - 1)]
                                                                   ~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb1891.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling2_loop/modeling_inkling2_loop.py:1523 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling2_loop/modeling_inkling2_loop.py:1357 in forward
```
</details>


## train_forward

### 1. Attempted to copy.deepcopy a tensor

- **Location:** `python/copy.py:137`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_FUNCTION_EX: a variadic function call, e.g. f(*args):

Attempted to copy.deepcopy a tensor
  Explanation: Dynamo does not support copy.deepcopy() on tensors.
  Hint: Avoid calling copy.deepcopy() on tensors inside compiled regions.

  Developer debug context: copy.deepcopy(TensorVariable())

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb4198.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb4198.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling2_loop/modeling_inkling2_loop.py:1523 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling2_loop/modeling_inkling2_loop.py:1357 in forward
```
</details>

### 2. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/inkling2_loop/modeling_inkling2_loop.py:1357`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Unsafe value live across comprehension graph break
  Explanation: A context manager or local generator cannot be reconstructed safely after a comprehension graph break, so the frame falls back to eager.
  Hint: Close the generator first, or move the comprehension into a helper.

  Developer debug context: LocalGeneratorObjectVariable(__iter__), frame skipped: forward (/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/inkling2_loop/modeling_inkling2_loop.py line 1304)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb1891.html

Stack variable source attribution:
  UserDefinedObjectVariable(DynamicCache) originated from:
  File "/admin/home/khushi_agrawal/Documents/pytorch/torch/_dynamo/polyfills/__init__.py", line 472
        obj = cls.__new__(cls, *args, **kwargs)
              ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^
  ListVariable(length=24) originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/cache_utils.py", line 2065
                layers = [
                         ^
                ...<2 lines>...
                ]
                ^
  NullVariable originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/inkling2_loop/modeling_inkling2_loop.py", line 1357
                cache.layers += [copy.deepcopy(layer) for layer in loop_cache_layers * (self.config.loop_num_passes - 1)]
                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  ListVariable(length=24) originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/inkling2_loop/modeling_inkling2_loop.py", line 1357
                cache.layers += [copy.deepcopy(layer) for layer in loop_cache_layers * (self.config.loop_num_passes - 1)]
                                                                   ~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb1891.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling2_loop/modeling_inkling2_loop.py:1523 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling2_loop/modeling_inkling2_loop.py:1357 in forward
```
</details>
