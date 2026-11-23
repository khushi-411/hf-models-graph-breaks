# dory — graph breaks

- **HF id:** `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16`
- **Class:** `AutoModelForCausalLM`
- **Config overrides:** `{'num_hidden_layers': 6, 'n_input_layers': 2, 'n_recurrent_layers': 2, 'n_output_layers': 2, 'n_recurrent_loops': 2, 'layer_types': ['sliding_attention', 'full_attention', 'sliding_attention', 'full_attention', 'sliding_attention', 'full_attention'], 'sliding_window': 16, 'hidden_size': 512, 'intermediate_size': 1024, 'num_attention_heads': 8, 'num_key_value_heads': 2, 'head_dim': 64, 'vocab_size': 32000}`
- **torch:** `2.16.0a0+git57f02f8`  |  **transformers:** `5.24.0.dev0` (git `046bdb8cc2`)
- **Notes:** Nemotron-Labs Dory. Looped (recurrent-depth) transformer, nGPT-style normalization, per-loop KV cache.

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ok | 1 | 0 | 0 |
| eval_with_cache | ok | 15 | 14 | 2 |
| train_forward | ok | 15 | 14 | 2 |

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
transformers/models/dory/modeling_dory.py:582 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/dory/modeling_dory.py:491 in forward
```
</details>

### 2. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/dory/modeling_dory.py:491`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Unsafe value live across comprehension graph break
  Explanation: A context manager or local generator cannot be reconstructed safely after a comprehension graph break, so the frame falls back to eager.
  Hint: Close the generator first, or move the comprehension into a helper.

  Developer debug context: LocalGeneratorObjectVariable(__iter__), frame skipped: forward (/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/dory/modeling_dory.py line 434)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb1891.html

Stack variable source attribution:
  UserDefinedObjectVariable(DynamicCache) originated from:
  File "/admin/home/khushi_agrawal/Documents/pytorch/torch/_dynamo/polyfills/__init__.py", line 471
        obj = cls.__new__(cls, *args, **kwargs)
              ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^
  ListVariable(length=6) originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/cache_utils.py", line 2065
                layers = [
                         ^
                ...<2 lines>...
                ]
                ^
  NullVariable originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/dory/modeling_dory.py", line 491
                cache.layers += [copy.deepcopy(c) for c in cache.layers[recurrent_start:recurrent_end] * (n_loops - 1)]
                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  ListVariable(length=2) originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/dory/modeling_dory.py", line 491
                cache.layers += [copy.deepcopy(c) for c in cache.layers[recurrent_start:recurrent_end] * (n_loops - 1)]
                                                           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb1891.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/models/dory/modeling_dory.py:582 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/dory/modeling_dory.py:491 in forward
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
transformers/models/dory/modeling_dory.py:582 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/dory/modeling_dory.py:491 in forward
```
</details>

### 2. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/dory/modeling_dory.py:491`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Unsafe value live across comprehension graph break
  Explanation: A context manager or local generator cannot be reconstructed safely after a comprehension graph break, so the frame falls back to eager.
  Hint: Close the generator first, or move the comprehension into a helper.

  Developer debug context: LocalGeneratorObjectVariable(__iter__), frame skipped: forward (/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/dory/modeling_dory.py line 434)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb1891.html

Stack variable source attribution:
  UserDefinedObjectVariable(DynamicCache) originated from:
  File "/admin/home/khushi_agrawal/Documents/pytorch/torch/_dynamo/polyfills/__init__.py", line 471
        obj = cls.__new__(cls, *args, **kwargs)
              ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^
  ListVariable(length=6) originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/cache_utils.py", line 2065
                layers = [
                         ^
                ...<2 lines>...
                ]
                ^
  NullVariable originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/dory/modeling_dory.py", line 491
                cache.layers += [copy.deepcopy(c) for c in cache.layers[recurrent_start:recurrent_end] * (n_loops - 1)]
                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  ListVariable(length=2) originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/dory/modeling_dory.py", line 491
                cache.layers += [copy.deepcopy(c) for c in cache.layers[recurrent_start:recurrent_end] * (n_loops - 1)]
                                                           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb1891.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/models/dory/modeling_dory.py:582 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/dory/modeling_dory.py:491 in forward
```
</details>
