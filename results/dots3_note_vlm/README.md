# dots3_note_vlm — graph breaks

- **HF id:** `dots-studio/dots3-note-prev`
- **Class:** `AutoModelForImageTextToText`
- **Config overrides:** `{'text_config.num_hidden_layers': 3, 'text_config.layer_types': ['indexed_attention', 'indexed_attention', 'sliding_attention'], 'text_config.mlp_layer_types': ['dense', 'sparse', 'sparse'], 'text_config.hidden_size': 512, 'text_config.intermediate_size': 1024, 'text_config.moe_intermediate_size': 128, 'text_config.shared_experts_intermediate_size': 128, 'text_config.num_attention_heads': 8, 'text_config.num_key_value_heads': 8, 'text_config.head_dim': 128, 'text_config.q_lora_rank': 128, 'text_config.kv_lora_rank': 64, 'text_config.qk_rope_head_dim': 32, 'text_config.qk_nope_head_dim': 32, 'text_config.v_head_dim': 32, 'text_config.index_topk': 16, 'text_config.index_head_dim': 64, 'text_config.index_n_heads': 4, 'text_config.n_routed_experts': 8, 'text_config.num_experts_per_tok': 2, 'text_config.sliding_window': 16, 'vision_config.num_hidden_layers': 2}`
- **torch:** `2.16.0a0+git9933f27`  |  **transformers:** `5.24.0.dev0` (git `84fe6ceb8a`)
- **Notes:** Full dots3-note with the image path (vision tower + adapter). Audio input not exercised. Text overrides match `dots3_note`.

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ok | 19 | 18 | 6 |
| eval_with_cache | ok | 19 | 18 | 6 |
| train_forward | ok | 19 | 18 | 6 |

## eval_forward

### 1. Data dependent operator

- **Location:** `transformers/vision_utils.py:112`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_FUNCTION_EX: a variadic function call, e.g. f(*args):

Data dependent operator
  Explanation: Operator `aten._local_scalar_dense.default` has a non-Tensor output whose value is dependent on the data of Tensor inputs.
  Hint: Enable tracing of data-dependent output operators with `torch._dynamo.config.capture_scalar_outputs = True`

  Developer debug context: aten._local_scalar_dense.default

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html

<details><summary>User stack</summary>

```
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/dots3_note/modeling_dots3_note.py:1768 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/dots3_note/modeling_dots3_note.py:1677 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/dots3_note/modeling_dots3_note.py:1587 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/dots3_note/modeling_dots3_note.py:1545 in forward
transformers/vision_utils.py:112 in get_vision_position_ids
```
</details>

### 2. Dynamic shape operator

- **Location:** `transformers/vision_utils.py:64`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_KW: a function call with keyword arguments, e.g. f(x=True):

Dynamic shape operator
  Explanation: Operator `aten.repeat_interleave.Tensor`'s output shape depends on input Tensor data.
  Hint: Enable tracing of dynamic shape operators with `torch._dynamo.config.capture_dynamic_output_shape_ops = True`

  Developer debug context: aten.repeat_interleave.Tensor

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

<details><summary>User stack</summary>

```
transformers/models/dots3_note/modeling_dots3_note.py:1768 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/dots3_note/modeling_dots3_note.py:1677 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/dots3_note/modeling_dots3_note.py:1587 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/dots3_note/modeling_dots3_note.py:1546 in forward
transformers/vision_utils.py:76 in get_vision_attention_seqlens
transformers/vision_utils.py:64 in get_vision_cu_seqlens
```
</details>

### 3. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/dots3_note/modeling_dots3_note.py:1413`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Data dependent operator
  Explanation: Operator `aten._local_scalar_dense.default` has a non-Tensor output whose value is dependent on the data of Tensor inputs.
  Hint: Enable tracing of data-dependent output operators with `torch._dynamo.config.capture_scalar_outputs = True`

  Developer debug context: aten._local_scalar_dense.default

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html

*** While handling this graph break, another graph break occurred: ***

graph break in loop
  Explanation: torch.compile detected a graph break in a for/while loop. Skipping the frame and falling back to eager, as graph breaks in loops are not supported.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: frame skipped: torch_dynamo_resume_in_forward_at_1546 (/home/khushi/Documents/transformers/src/transformers/models/dots3_note/modeling_dots3_note.py line 1546)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  DictValuesIterator() originated from:
  File "/home/khushi/Documents/osc/work/pytorch/torch/nn/modules/container.py", line 416
            return iter(self._modules.values())
                   ~~~~^^^^^^^^^^^^^^^^^^^^^^^^
  NullVariable originated from:
  File "/home/khushi/Documents/transformers/src/transformers/models/dots3_note/modeling_dots3_note.py", line 1550
            for block in self.blocks:
                         ^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html

<details><summary>User stack</summary>

```
transformers/models/dots3_note/modeling_dots3_note.py:1768 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/dots3_note/modeling_dots3_note.py:1677 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/dots3_note/modeling_dots3_note.py:1587 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/dots3_note/modeling_dots3_note.py:1551 in forward
transformers/modeling_layers.py:122 in forward_with_checkpointing
transformers/models/dots3_note/modeling_dots3_note.py:1413 in forward
```
</details>

### 4. Data dependent operator

- **Location:** `transformers/models/dots3_note/modeling_dots3_note.py:1413`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_FUNCTION_EX: a variadic function call, e.g. f(*args):

Data dependent operator
  Explanation: Operator `aten._local_scalar_dense.default` has a non-Tensor output whose value is dependent on the data of Tensor inputs.
  Hint: Enable tracing of data-dependent output operators with `torch._dynamo.config.capture_scalar_outputs = True`

  Developer debug context: aten._local_scalar_dense.default

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html

<details><summary>User stack</summary>

```
transformers/models/dots3_note/modeling_dots3_note.py:1768 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/dots3_note/modeling_dots3_note.py:1677 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/dots3_note/modeling_dots3_note.py:1587 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/dots3_note/modeling_dots3_note.py:1551 in forward
transformers/modeling_layers.py:122 in forward_with_checkpointing
transformers/models/dots3_note/modeling_dots3_note.py:1413 in forward
```
</details>

### 5. Data dependent operator

- **Location:** `transformers/models/dots3_note/modeling_dots3_note.py:1594`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Data dependent operator
  Explanation: Operator `aten._local_scalar_dense.default` has a non-Tensor output whose value is dependent on the data of Tensor inputs.
  Hint: Enable tracing of data-dependent output operators with `torch._dynamo.config.capture_scalar_outputs = True`

  Developer debug context: aten._local_scalar_dense.default

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/dots3_note/modeling_dots3_note.py:1768 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/dots3_note/modeling_dots3_note.py:1677 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/dots3_note/modeling_dots3_note.py:1594 in get_image_features
```
</details>

### 6. Dynamic shape operator

- **Location:** `transformers/models/dots3_note/modeling_dots3_note.py:1639`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Dynamic shape operator
  Explanation: Operator `aten.nonzero.default`'s output shape depends on input Tensor data.
  Hint: Enable tracing of dynamic shape operators with `torch._dynamo.config.capture_dynamic_output_shape_ops = True`

  Developer debug context: aten.nonzero.default

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/dots3_note/modeling_dots3_note.py:1768 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/dots3_note/modeling_dots3_note.py:1682 in forward
transformers/models/dots3_note/modeling_dots3_note.py:1639 in get_placeholder_mask
```
</details>


## eval_with_cache

### 1. Data dependent operator

- **Location:** `transformers/vision_utils.py:112`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_FUNCTION_EX: a variadic function call, e.g. f(*args):

Data dependent operator
  Explanation: Operator `aten._local_scalar_dense.default` has a non-Tensor output whose value is dependent on the data of Tensor inputs.
  Hint: Enable tracing of data-dependent output operators with `torch._dynamo.config.capture_scalar_outputs = True`

  Developer debug context: aten._local_scalar_dense.default

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html

<details><summary>User stack</summary>

```
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/dots3_note/modeling_dots3_note.py:1768 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/dots3_note/modeling_dots3_note.py:1677 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/dots3_note/modeling_dots3_note.py:1587 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/dots3_note/modeling_dots3_note.py:1545 in forward
transformers/vision_utils.py:112 in get_vision_position_ids
```
</details>

### 2. Dynamic shape operator

- **Location:** `transformers/vision_utils.py:64`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_KW: a function call with keyword arguments, e.g. f(x=True):

Dynamic shape operator
  Explanation: Operator `aten.repeat_interleave.Tensor`'s output shape depends on input Tensor data.
  Hint: Enable tracing of dynamic shape operators with `torch._dynamo.config.capture_dynamic_output_shape_ops = True`

  Developer debug context: aten.repeat_interleave.Tensor

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

<details><summary>User stack</summary>

```
transformers/models/dots3_note/modeling_dots3_note.py:1768 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/dots3_note/modeling_dots3_note.py:1677 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/dots3_note/modeling_dots3_note.py:1587 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/dots3_note/modeling_dots3_note.py:1546 in forward
transformers/vision_utils.py:76 in get_vision_attention_seqlens
transformers/vision_utils.py:64 in get_vision_cu_seqlens
```
</details>

### 3. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/dots3_note/modeling_dots3_note.py:1413`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Data dependent operator
  Explanation: Operator `aten._local_scalar_dense.default` has a non-Tensor output whose value is dependent on the data of Tensor inputs.
  Hint: Enable tracing of data-dependent output operators with `torch._dynamo.config.capture_scalar_outputs = True`

  Developer debug context: aten._local_scalar_dense.default

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html

*** While handling this graph break, another graph break occurred: ***

graph break in loop
  Explanation: torch.compile detected a graph break in a for/while loop. Skipping the frame and falling back to eager, as graph breaks in loops are not supported.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: frame skipped: torch_dynamo_resume_in_forward_at_1546 (/home/khushi/Documents/transformers/src/transformers/models/dots3_note/modeling_dots3_note.py line 1546)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  DictValuesIterator() originated from:
  File "/home/khushi/Documents/osc/work/pytorch/torch/nn/modules/container.py", line 416
            return iter(self._modules.values())
                   ~~~~^^^^^^^^^^^^^^^^^^^^^^^^
  NullVariable originated from:
  File "/home/khushi/Documents/transformers/src/transformers/models/dots3_note/modeling_dots3_note.py", line 1550
            for block in self.blocks:
                         ^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html

<details><summary>User stack</summary>

```
transformers/models/dots3_note/modeling_dots3_note.py:1768 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/dots3_note/modeling_dots3_note.py:1677 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/dots3_note/modeling_dots3_note.py:1587 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/dots3_note/modeling_dots3_note.py:1551 in forward
transformers/modeling_layers.py:122 in forward_with_checkpointing
transformers/models/dots3_note/modeling_dots3_note.py:1413 in forward
```
</details>

### 4. Data dependent operator

- **Location:** `transformers/models/dots3_note/modeling_dots3_note.py:1413`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_FUNCTION_EX: a variadic function call, e.g. f(*args):

Data dependent operator
  Explanation: Operator `aten._local_scalar_dense.default` has a non-Tensor output whose value is dependent on the data of Tensor inputs.
  Hint: Enable tracing of data-dependent output operators with `torch._dynamo.config.capture_scalar_outputs = True`

  Developer debug context: aten._local_scalar_dense.default

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html

<details><summary>User stack</summary>

```
transformers/models/dots3_note/modeling_dots3_note.py:1768 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/dots3_note/modeling_dots3_note.py:1677 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/dots3_note/modeling_dots3_note.py:1587 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/dots3_note/modeling_dots3_note.py:1551 in forward
transformers/modeling_layers.py:122 in forward_with_checkpointing
transformers/models/dots3_note/modeling_dots3_note.py:1413 in forward
```
</details>

### 5. Data dependent operator

- **Location:** `transformers/models/dots3_note/modeling_dots3_note.py:1594`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Data dependent operator
  Explanation: Operator `aten._local_scalar_dense.default` has a non-Tensor output whose value is dependent on the data of Tensor inputs.
  Hint: Enable tracing of data-dependent output operators with `torch._dynamo.config.capture_scalar_outputs = True`

  Developer debug context: aten._local_scalar_dense.default

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/dots3_note/modeling_dots3_note.py:1768 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/dots3_note/modeling_dots3_note.py:1677 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/dots3_note/modeling_dots3_note.py:1594 in get_image_features
```
</details>

### 6. Dynamic shape operator

- **Location:** `transformers/models/dots3_note/modeling_dots3_note.py:1639`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Dynamic shape operator
  Explanation: Operator `aten.nonzero.default`'s output shape depends on input Tensor data.
  Hint: Enable tracing of dynamic shape operators with `torch._dynamo.config.capture_dynamic_output_shape_ops = True`

  Developer debug context: aten.nonzero.default

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/dots3_note/modeling_dots3_note.py:1768 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/dots3_note/modeling_dots3_note.py:1682 in forward
transformers/models/dots3_note/modeling_dots3_note.py:1639 in get_placeholder_mask
```
</details>


## train_forward

### 1. Data dependent operator

- **Location:** `transformers/vision_utils.py:112`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_FUNCTION_EX: a variadic function call, e.g. f(*args):

Data dependent operator
  Explanation: Operator `aten._local_scalar_dense.default` has a non-Tensor output whose value is dependent on the data of Tensor inputs.
  Hint: Enable tracing of data-dependent output operators with `torch._dynamo.config.capture_scalar_outputs = True`

  Developer debug context: aten._local_scalar_dense.default

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html

<details><summary>User stack</summary>

```
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/dots3_note/modeling_dots3_note.py:1768 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/dots3_note/modeling_dots3_note.py:1677 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/dots3_note/modeling_dots3_note.py:1587 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/dots3_note/modeling_dots3_note.py:1545 in forward
transformers/vision_utils.py:112 in get_vision_position_ids
```
</details>

### 2. Dynamic shape operator

- **Location:** `transformers/vision_utils.py:64`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_KW: a function call with keyword arguments, e.g. f(x=True):

Dynamic shape operator
  Explanation: Operator `aten.repeat_interleave.Tensor`'s output shape depends on input Tensor data.
  Hint: Enable tracing of dynamic shape operators with `torch._dynamo.config.capture_dynamic_output_shape_ops = True`

  Developer debug context: aten.repeat_interleave.Tensor

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

<details><summary>User stack</summary>

```
transformers/models/dots3_note/modeling_dots3_note.py:1768 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/dots3_note/modeling_dots3_note.py:1677 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/dots3_note/modeling_dots3_note.py:1587 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/dots3_note/modeling_dots3_note.py:1546 in forward
transformers/vision_utils.py:76 in get_vision_attention_seqlens
transformers/vision_utils.py:64 in get_vision_cu_seqlens
```
</details>

### 3. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/dots3_note/modeling_dots3_note.py:1413`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Data dependent operator
  Explanation: Operator `aten._local_scalar_dense.default` has a non-Tensor output whose value is dependent on the data of Tensor inputs.
  Hint: Enable tracing of data-dependent output operators with `torch._dynamo.config.capture_scalar_outputs = True`

  Developer debug context: aten._local_scalar_dense.default

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html

*** While handling this graph break, another graph break occurred: ***

graph break in loop
  Explanation: torch.compile detected a graph break in a for/while loop. Skipping the frame and falling back to eager, as graph breaks in loops are not supported.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: frame skipped: torch_dynamo_resume_in_forward_at_1546 (/home/khushi/Documents/transformers/src/transformers/models/dots3_note/modeling_dots3_note.py line 1546)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  DictValuesIterator() originated from:
  File "/home/khushi/Documents/osc/work/pytorch/torch/nn/modules/container.py", line 416
            return iter(self._modules.values())
                   ~~~~^^^^^^^^^^^^^^^^^^^^^^^^
  NullVariable originated from:
  File "/home/khushi/Documents/transformers/src/transformers/models/dots3_note/modeling_dots3_note.py", line 1550
            for block in self.blocks:
                         ^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html

<details><summary>User stack</summary>

```
transformers/models/dots3_note/modeling_dots3_note.py:1768 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/dots3_note/modeling_dots3_note.py:1677 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/dots3_note/modeling_dots3_note.py:1587 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/dots3_note/modeling_dots3_note.py:1551 in forward
transformers/modeling_layers.py:122 in forward_with_checkpointing
transformers/models/dots3_note/modeling_dots3_note.py:1413 in forward
```
</details>

### 4. Data dependent operator

- **Location:** `transformers/models/dots3_note/modeling_dots3_note.py:1413`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_FUNCTION_EX: a variadic function call, e.g. f(*args):

Data dependent operator
  Explanation: Operator `aten._local_scalar_dense.default` has a non-Tensor output whose value is dependent on the data of Tensor inputs.
  Hint: Enable tracing of data-dependent output operators with `torch._dynamo.config.capture_scalar_outputs = True`

  Developer debug context: aten._local_scalar_dense.default

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html

<details><summary>User stack</summary>

```
transformers/models/dots3_note/modeling_dots3_note.py:1768 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/dots3_note/modeling_dots3_note.py:1677 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/dots3_note/modeling_dots3_note.py:1587 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/dots3_note/modeling_dots3_note.py:1551 in forward
transformers/modeling_layers.py:122 in forward_with_checkpointing
transformers/models/dots3_note/modeling_dots3_note.py:1413 in forward
```
</details>

### 5. Data dependent operator

- **Location:** `transformers/models/dots3_note/modeling_dots3_note.py:1594`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Data dependent operator
  Explanation: Operator `aten._local_scalar_dense.default` has a non-Tensor output whose value is dependent on the data of Tensor inputs.
  Hint: Enable tracing of data-dependent output operators with `torch._dynamo.config.capture_scalar_outputs = True`

  Developer debug context: aten._local_scalar_dense.default

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/dots3_note/modeling_dots3_note.py:1768 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/dots3_note/modeling_dots3_note.py:1677 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/dots3_note/modeling_dots3_note.py:1594 in get_image_features
```
</details>

### 6. Dynamic shape operator

- **Location:** `transformers/models/dots3_note/modeling_dots3_note.py:1639`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Dynamic shape operator
  Explanation: Operator `aten.nonzero.default`'s output shape depends on input Tensor data.
  Hint: Enable tracing of dynamic shape operators with `torch._dynamo.config.capture_dynamic_output_shape_ops = True`

  Developer debug context: aten.nonzero.default

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/dots3_note/modeling_dots3_note.py:1768 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/dots3_note/modeling_dots3_note.py:1682 in forward
transformers/models/dots3_note/modeling_dots3_note.py:1639 in get_placeholder_mask
```
</details>
