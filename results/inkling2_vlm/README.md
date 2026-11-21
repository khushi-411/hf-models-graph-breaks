# inkling2_vlm — graph breaks

- **HF id:** `thinkingmachines/Inkling-2-Small-MXFP8`
- **Class:** `AutoModelForImageTextToText`
- **Config overrides:** `{'text_config.num_hidden_layers': 7, 'text_config.layer_types': ['hybrid_sliding', 'hybrid_sliding', 'hybrid_sliding', 'hybrid_sliding', 'hybrid_sliding', 'hybrid', 'hybrid_sliding'], 'text_config.hidden_size': 512, 'text_config.num_attention_heads': 8, 'text_config.num_key_value_heads': 2, 'text_config.head_dim': 64, 'text_config.swa_num_attention_heads': 8, 'text_config.swa_num_key_value_heads': 2, 'text_config.swa_head_dim': 64, 'text_config.sliding_window_size': 16, 'text_config.rel_extent': 16, 'text_config.moe_intermediate_size': 256, 'text_config.n_routed_experts': 8, 'text_config.num_experts_per_tok': 2, 'text_config.dtype': 'float32', 'quantization_config': None, 'dtype': 'float32', 'vision_config.num_hidden_layers': 2, 'vision_config.text_hidden_size': 512, 'vision_config.dtype': 'float32'}`
- **torch:** `2.16.0a0+git57f02f8`  |  **transformers:** `5.24.0.dev0` (git `b7fe69cb19`)
- **Notes:** Full Inkling-2 (vision tower + LM). Text overrides match the `inkling2` entry.

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ok | 26 | 25 | 10 |
| eval_with_cache | ok | 26 | 25 | 10 |
| train_forward | ok | 26 | 25 | 10 |

## eval_forward

### 1. Dynamic shape operator

- **Location:** `transformers/models/inkling2/modeling_inkling2.py:1444`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_FUNCTION_EX: a variadic function call, e.g. f(*args):

Dynamic shape operator
  Explanation: Operator `aten.repeat_interleave.Tensor`'s output shape depends on input Tensor data.
  Hint: Enable tracing of dynamic shape operators with `torch._dynamo.config.capture_dynamic_output_shape_ops = True`

  Developer debug context: aten.repeat_interleave.Tensor

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling2/modeling_inkling2.py:1704 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling2/modeling_inkling2.py:1587 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/inkling2/modeling_inkling2.py:1489 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/inkling2/modeling_inkling2.py:1444 in forward
```
</details>

### 2. Dynamic shape operator

- **Location:** `transformers/vision_utils.py:276`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_FUNCTION_EX: a variadic function call, e.g. f(*args):

Dynamic shape operator
  Explanation: Operator `aten.repeat_interleave.Tensor`'s output shape depends on input Tensor data.
  Hint: Enable tracing of dynamic shape operators with `torch._dynamo.config.capture_dynamic_output_shape_ops = True`

  Developer debug context: aten.repeat_interleave.Tensor

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

<details><summary>User stack</summary>

```
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling2/modeling_inkling2.py:1704 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling2/modeling_inkling2.py:1587 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/inkling2/modeling_inkling2.py:1489 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/inkling2/modeling_inkling2.py:1121 in forward
transformers/vision_utils.py:276 in get_vision_interpolation_indices_and_weights
```
</details>

### 3. Dynamic shape operator

- **Location:** `transformers/vision_utils.py:277`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Dynamic shape operator
  Explanation: Operator `aten.repeat_interleave.Tensor`'s output shape depends on input Tensor data.
  Hint: Enable tracing of dynamic shape operators with `torch._dynamo.config.capture_dynamic_output_shape_ops = True`

  Developer debug context: aten.repeat_interleave.Tensor

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

<details><summary>User stack</summary>

```
transformers/models/inkling2/modeling_inkling2.py:1704 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling2/modeling_inkling2.py:1587 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/inkling2/modeling_inkling2.py:1489 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/inkling2/modeling_inkling2.py:1448 in forward
transformers/models/inkling2/modeling_inkling2.py:1121 in forward
transformers/vision_utils.py:277 in get_vision_interpolation_indices_and_weights
```
</details>

### 4. Dynamic shape operator

- **Location:** `transformers/vision_utils.py:278`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Dynamic shape operator
  Explanation: Operator `aten.repeat_interleave.Tensor`'s output shape depends on input Tensor data.
  Hint: Enable tracing of dynamic shape operators with `torch._dynamo.config.capture_dynamic_output_shape_ops = True`

  Developer debug context: aten.repeat_interleave.Tensor

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

<details><summary>User stack</summary>

```
transformers/models/inkling2/modeling_inkling2.py:1704 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling2/modeling_inkling2.py:1587 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/inkling2/modeling_inkling2.py:1489 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/inkling2/modeling_inkling2.py:1448 in forward
transformers/models/inkling2/modeling_inkling2.py:1121 in forward
transformers/vision_utils.py:278 in get_vision_interpolation_indices_and_weights
```
</details>

### 5. Data dependent operator

- **Location:** `transformers/vision_utils.py:280`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_KW: a function call with keyword arguments, e.g. f(x=True):

Data dependent operator
  Explanation: Operator `aten._local_scalar_dense.default` has a non-Tensor output whose value is dependent on the data of Tensor inputs.
  Hint: Enable tracing of data-dependent output operators with `torch._dynamo.config.capture_scalar_outputs = True`

  Developer debug context: aten._local_scalar_dense.default

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html

<details><summary>User stack</summary>

```
transformers/models/inkling2/modeling_inkling2.py:1704 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling2/modeling_inkling2.py:1587 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/inkling2/modeling_inkling2.py:1489 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/inkling2/modeling_inkling2.py:1448 in forward
transformers/models/inkling2/modeling_inkling2.py:1121 in forward
transformers/vision_utils.py:280 in get_vision_interpolation_indices_and_weights
```
</details>

### 6. Data dependent operator

- **Location:** `transformers/vision_utils.py:112`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_KW: a function call with keyword arguments, e.g. f(x=True):

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
transformers/models/inkling2/modeling_inkling2.py:1704 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling2/modeling_inkling2.py:1587 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/inkling2/modeling_inkling2.py:1489 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/inkling2/modeling_inkling2.py:1452 in forward
transformers/vision_utils.py:112 in get_vision_position_ids
```
</details>

### 7. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/inkling2/modeling_inkling2.py:1027`  (hit ×1)

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

  Developer debug context: frame skipped: torch_dynamo_resume_in_forward_at_1452 (/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/inkling2/modeling_inkling2.py line 1452)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  DictValuesIterator() originated from:
  File "/admin/home/khushi_agrawal/miniforge3/envs/torch-snap-57f02f8/lib/python3.15/site-packages/torch/nn/modules/container.py", line 416
            return iter(self._modules.values())
                   ~~~~^^^^^^^^^^^^^^^^^^^^^^^^
  NullVariable originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/inkling2/modeling_inkling2.py", line 1456
            for layer in self.layers:
                         ^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html

<details><summary>User stack</summary>

```
transformers/models/inkling2/modeling_inkling2.py:1704 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling2/modeling_inkling2.py:1587 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/inkling2/modeling_inkling2.py:1489 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/inkling2/modeling_inkling2.py:1457 in forward
transformers/modeling_layers.py:122 in forward_with_checkpointing
transformers/models/inkling2/modeling_inkling2.py:1027 in forward
```
</details>

### 8. Data dependent operator

- **Location:** `transformers/models/inkling2/modeling_inkling2.py:1027`  (hit ×1)

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
transformers/models/inkling2/modeling_inkling2.py:1704 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling2/modeling_inkling2.py:1587 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/inkling2/modeling_inkling2.py:1489 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/inkling2/modeling_inkling2.py:1457 in forward
transformers/modeling_layers.py:122 in forward_with_checkpointing
transformers/models/inkling2/modeling_inkling2.py:1027 in forward
```
</details>

### 9. Data dependent operator

- **Location:** `transformers/models/inkling2/modeling_inkling2.py:1490`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Data dependent operator
  Explanation: Operator `aten._local_scalar_dense.default` has a non-Tensor output whose value is dependent on the data of Tensor inputs.
  Hint: Enable tracing of data-dependent output operators with `torch._dynamo.config.capture_scalar_outputs = True`

  Developer debug context: aten._local_scalar_dense.default

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html

Stack variable source attribution:
  GetAttrVariable(TensorVariable(), split) originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/inkling2/modeling_inkling2.py", line 1490
            image_outputs.pooler_output = image_outputs.pooler_output.split(image_grid_thw.prod(-1).tolist())
                                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling2/modeling_inkling2.py:1704 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling2/modeling_inkling2.py:1587 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/inkling2/modeling_inkling2.py:1490 in get_image_features
```
</details>

### 10. Dynamic shape operator

- **Location:** `transformers/models/inkling2/modeling_inkling2.py:1515`  (hit ×1)

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
transformers/models/inkling2/modeling_inkling2.py:1704 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling2/modeling_inkling2.py:1593 in forward
transformers/models/inkling2/modeling_inkling2.py:1515 in get_placeholder_mask
```
</details>


## eval_with_cache

### 1. Dynamic shape operator

- **Location:** `transformers/models/inkling2/modeling_inkling2.py:1444`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_FUNCTION_EX: a variadic function call, e.g. f(*args):

Dynamic shape operator
  Explanation: Operator `aten.repeat_interleave.Tensor`'s output shape depends on input Tensor data.
  Hint: Enable tracing of dynamic shape operators with `torch._dynamo.config.capture_dynamic_output_shape_ops = True`

  Developer debug context: aten.repeat_interleave.Tensor

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling2/modeling_inkling2.py:1704 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling2/modeling_inkling2.py:1587 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/inkling2/modeling_inkling2.py:1489 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/inkling2/modeling_inkling2.py:1444 in forward
```
</details>

### 2. Dynamic shape operator

- **Location:** `transformers/vision_utils.py:276`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_FUNCTION_EX: a variadic function call, e.g. f(*args):

Dynamic shape operator
  Explanation: Operator `aten.repeat_interleave.Tensor`'s output shape depends on input Tensor data.
  Hint: Enable tracing of dynamic shape operators with `torch._dynamo.config.capture_dynamic_output_shape_ops = True`

  Developer debug context: aten.repeat_interleave.Tensor

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

<details><summary>User stack</summary>

```
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling2/modeling_inkling2.py:1704 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling2/modeling_inkling2.py:1587 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/inkling2/modeling_inkling2.py:1489 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/inkling2/modeling_inkling2.py:1121 in forward
transformers/vision_utils.py:276 in get_vision_interpolation_indices_and_weights
```
</details>

### 3. Dynamic shape operator

- **Location:** `transformers/vision_utils.py:277`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Dynamic shape operator
  Explanation: Operator `aten.repeat_interleave.Tensor`'s output shape depends on input Tensor data.
  Hint: Enable tracing of dynamic shape operators with `torch._dynamo.config.capture_dynamic_output_shape_ops = True`

  Developer debug context: aten.repeat_interleave.Tensor

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

<details><summary>User stack</summary>

```
transformers/models/inkling2/modeling_inkling2.py:1704 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling2/modeling_inkling2.py:1587 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/inkling2/modeling_inkling2.py:1489 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/inkling2/modeling_inkling2.py:1448 in forward
transformers/models/inkling2/modeling_inkling2.py:1121 in forward
transformers/vision_utils.py:277 in get_vision_interpolation_indices_and_weights
```
</details>

### 4. Dynamic shape operator

- **Location:** `transformers/vision_utils.py:278`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Dynamic shape operator
  Explanation: Operator `aten.repeat_interleave.Tensor`'s output shape depends on input Tensor data.
  Hint: Enable tracing of dynamic shape operators with `torch._dynamo.config.capture_dynamic_output_shape_ops = True`

  Developer debug context: aten.repeat_interleave.Tensor

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

<details><summary>User stack</summary>

```
transformers/models/inkling2/modeling_inkling2.py:1704 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling2/modeling_inkling2.py:1587 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/inkling2/modeling_inkling2.py:1489 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/inkling2/modeling_inkling2.py:1448 in forward
transformers/models/inkling2/modeling_inkling2.py:1121 in forward
transformers/vision_utils.py:278 in get_vision_interpolation_indices_and_weights
```
</details>

### 5. Data dependent operator

- **Location:** `transformers/vision_utils.py:280`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_KW: a function call with keyword arguments, e.g. f(x=True):

Data dependent operator
  Explanation: Operator `aten._local_scalar_dense.default` has a non-Tensor output whose value is dependent on the data of Tensor inputs.
  Hint: Enable tracing of data-dependent output operators with `torch._dynamo.config.capture_scalar_outputs = True`

  Developer debug context: aten._local_scalar_dense.default

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html

<details><summary>User stack</summary>

```
transformers/models/inkling2/modeling_inkling2.py:1704 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling2/modeling_inkling2.py:1587 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/inkling2/modeling_inkling2.py:1489 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/inkling2/modeling_inkling2.py:1448 in forward
transformers/models/inkling2/modeling_inkling2.py:1121 in forward
transformers/vision_utils.py:280 in get_vision_interpolation_indices_and_weights
```
</details>

### 6. Data dependent operator

- **Location:** `transformers/vision_utils.py:112`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_KW: a function call with keyword arguments, e.g. f(x=True):

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
transformers/models/inkling2/modeling_inkling2.py:1704 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling2/modeling_inkling2.py:1587 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/inkling2/modeling_inkling2.py:1489 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/inkling2/modeling_inkling2.py:1452 in forward
transformers/vision_utils.py:112 in get_vision_position_ids
```
</details>

### 7. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/inkling2/modeling_inkling2.py:1027`  (hit ×1)

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

  Developer debug context: frame skipped: torch_dynamo_resume_in_forward_at_1452 (/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/inkling2/modeling_inkling2.py line 1452)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  DictValuesIterator() originated from:
  File "/admin/home/khushi_agrawal/miniforge3/envs/torch-snap-57f02f8/lib/python3.15/site-packages/torch/nn/modules/container.py", line 416
            return iter(self._modules.values())
                   ~~~~^^^^^^^^^^^^^^^^^^^^^^^^
  NullVariable originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/inkling2/modeling_inkling2.py", line 1456
            for layer in self.layers:
                         ^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html

<details><summary>User stack</summary>

```
transformers/models/inkling2/modeling_inkling2.py:1704 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling2/modeling_inkling2.py:1587 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/inkling2/modeling_inkling2.py:1489 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/inkling2/modeling_inkling2.py:1457 in forward
transformers/modeling_layers.py:122 in forward_with_checkpointing
transformers/models/inkling2/modeling_inkling2.py:1027 in forward
```
</details>

### 8. Data dependent operator

- **Location:** `transformers/models/inkling2/modeling_inkling2.py:1027`  (hit ×1)

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
transformers/models/inkling2/modeling_inkling2.py:1704 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling2/modeling_inkling2.py:1587 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/inkling2/modeling_inkling2.py:1489 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/inkling2/modeling_inkling2.py:1457 in forward
transformers/modeling_layers.py:122 in forward_with_checkpointing
transformers/models/inkling2/modeling_inkling2.py:1027 in forward
```
</details>

### 9. Data dependent operator

- **Location:** `transformers/models/inkling2/modeling_inkling2.py:1490`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Data dependent operator
  Explanation: Operator `aten._local_scalar_dense.default` has a non-Tensor output whose value is dependent on the data of Tensor inputs.
  Hint: Enable tracing of data-dependent output operators with `torch._dynamo.config.capture_scalar_outputs = True`

  Developer debug context: aten._local_scalar_dense.default

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html

Stack variable source attribution:
  GetAttrVariable(TensorVariable(), split) originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/inkling2/modeling_inkling2.py", line 1490
            image_outputs.pooler_output = image_outputs.pooler_output.split(image_grid_thw.prod(-1).tolist())
                                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling2/modeling_inkling2.py:1704 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling2/modeling_inkling2.py:1587 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/inkling2/modeling_inkling2.py:1490 in get_image_features
```
</details>

### 10. Dynamic shape operator

- **Location:** `transformers/models/inkling2/modeling_inkling2.py:1515`  (hit ×1)

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
transformers/models/inkling2/modeling_inkling2.py:1704 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling2/modeling_inkling2.py:1593 in forward
transformers/models/inkling2/modeling_inkling2.py:1515 in get_placeholder_mask
```
</details>


## train_forward

### 1. Dynamic shape operator

- **Location:** `transformers/models/inkling2/modeling_inkling2.py:1444`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_FUNCTION_EX: a variadic function call, e.g. f(*args):

Dynamic shape operator
  Explanation: Operator `aten.repeat_interleave.Tensor`'s output shape depends on input Tensor data.
  Hint: Enable tracing of dynamic shape operators with `torch._dynamo.config.capture_dynamic_output_shape_ops = True`

  Developer debug context: aten.repeat_interleave.Tensor

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling2/modeling_inkling2.py:1704 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling2/modeling_inkling2.py:1587 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/inkling2/modeling_inkling2.py:1489 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/inkling2/modeling_inkling2.py:1444 in forward
```
</details>

### 2. Dynamic shape operator

- **Location:** `transformers/vision_utils.py:276`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_FUNCTION_EX: a variadic function call, e.g. f(*args):

Dynamic shape operator
  Explanation: Operator `aten.repeat_interleave.Tensor`'s output shape depends on input Tensor data.
  Hint: Enable tracing of dynamic shape operators with `torch._dynamo.config.capture_dynamic_output_shape_ops = True`

  Developer debug context: aten.repeat_interleave.Tensor

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

<details><summary>User stack</summary>

```
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling2/modeling_inkling2.py:1704 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling2/modeling_inkling2.py:1587 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/inkling2/modeling_inkling2.py:1489 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/inkling2/modeling_inkling2.py:1121 in forward
transformers/vision_utils.py:276 in get_vision_interpolation_indices_and_weights
```
</details>

### 3. Dynamic shape operator

- **Location:** `transformers/vision_utils.py:277`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Dynamic shape operator
  Explanation: Operator `aten.repeat_interleave.Tensor`'s output shape depends on input Tensor data.
  Hint: Enable tracing of dynamic shape operators with `torch._dynamo.config.capture_dynamic_output_shape_ops = True`

  Developer debug context: aten.repeat_interleave.Tensor

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

<details><summary>User stack</summary>

```
transformers/models/inkling2/modeling_inkling2.py:1704 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling2/modeling_inkling2.py:1587 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/inkling2/modeling_inkling2.py:1489 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/inkling2/modeling_inkling2.py:1448 in forward
transformers/models/inkling2/modeling_inkling2.py:1121 in forward
transformers/vision_utils.py:277 in get_vision_interpolation_indices_and_weights
```
</details>

### 4. Dynamic shape operator

- **Location:** `transformers/vision_utils.py:278`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Dynamic shape operator
  Explanation: Operator `aten.repeat_interleave.Tensor`'s output shape depends on input Tensor data.
  Hint: Enable tracing of dynamic shape operators with `torch._dynamo.config.capture_dynamic_output_shape_ops = True`

  Developer debug context: aten.repeat_interleave.Tensor

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

<details><summary>User stack</summary>

```
transformers/models/inkling2/modeling_inkling2.py:1704 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling2/modeling_inkling2.py:1587 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/inkling2/modeling_inkling2.py:1489 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/inkling2/modeling_inkling2.py:1448 in forward
transformers/models/inkling2/modeling_inkling2.py:1121 in forward
transformers/vision_utils.py:278 in get_vision_interpolation_indices_and_weights
```
</details>

### 5. Data dependent operator

- **Location:** `transformers/vision_utils.py:280`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_KW: a function call with keyword arguments, e.g. f(x=True):

Data dependent operator
  Explanation: Operator `aten._local_scalar_dense.default` has a non-Tensor output whose value is dependent on the data of Tensor inputs.
  Hint: Enable tracing of data-dependent output operators with `torch._dynamo.config.capture_scalar_outputs = True`

  Developer debug context: aten._local_scalar_dense.default

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html

<details><summary>User stack</summary>

```
transformers/models/inkling2/modeling_inkling2.py:1704 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling2/modeling_inkling2.py:1587 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/inkling2/modeling_inkling2.py:1489 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/inkling2/modeling_inkling2.py:1448 in forward
transformers/models/inkling2/modeling_inkling2.py:1121 in forward
transformers/vision_utils.py:280 in get_vision_interpolation_indices_and_weights
```
</details>

### 6. Data dependent operator

- **Location:** `transformers/vision_utils.py:112`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_KW: a function call with keyword arguments, e.g. f(x=True):

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
transformers/models/inkling2/modeling_inkling2.py:1704 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling2/modeling_inkling2.py:1587 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/inkling2/modeling_inkling2.py:1489 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/inkling2/modeling_inkling2.py:1452 in forward
transformers/vision_utils.py:112 in get_vision_position_ids
```
</details>

### 7. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/inkling2/modeling_inkling2.py:1027`  (hit ×1)

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

  Developer debug context: frame skipped: torch_dynamo_resume_in_forward_at_1452 (/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/inkling2/modeling_inkling2.py line 1452)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  DictValuesIterator() originated from:
  File "/admin/home/khushi_agrawal/miniforge3/envs/torch-snap-57f02f8/lib/python3.15/site-packages/torch/nn/modules/container.py", line 416
            return iter(self._modules.values())
                   ~~~~^^^^^^^^^^^^^^^^^^^^^^^^
  NullVariable originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/inkling2/modeling_inkling2.py", line 1456
            for layer in self.layers:
                         ^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html

<details><summary>User stack</summary>

```
transformers/models/inkling2/modeling_inkling2.py:1704 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling2/modeling_inkling2.py:1587 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/inkling2/modeling_inkling2.py:1489 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/inkling2/modeling_inkling2.py:1457 in forward
transformers/modeling_layers.py:122 in forward_with_checkpointing
transformers/models/inkling2/modeling_inkling2.py:1027 in forward
```
</details>

### 8. Data dependent operator

- **Location:** `transformers/models/inkling2/modeling_inkling2.py:1027`  (hit ×1)

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
transformers/models/inkling2/modeling_inkling2.py:1704 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling2/modeling_inkling2.py:1587 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/inkling2/modeling_inkling2.py:1489 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/inkling2/modeling_inkling2.py:1457 in forward
transformers/modeling_layers.py:122 in forward_with_checkpointing
transformers/models/inkling2/modeling_inkling2.py:1027 in forward
```
</details>

### 9. Data dependent operator

- **Location:** `transformers/models/inkling2/modeling_inkling2.py:1490`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Data dependent operator
  Explanation: Operator `aten._local_scalar_dense.default` has a non-Tensor output whose value is dependent on the data of Tensor inputs.
  Hint: Enable tracing of data-dependent output operators with `torch._dynamo.config.capture_scalar_outputs = True`

  Developer debug context: aten._local_scalar_dense.default

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html

Stack variable source attribution:
  GetAttrVariable(TensorVariable(), split) originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/inkling2/modeling_inkling2.py", line 1490
            image_outputs.pooler_output = image_outputs.pooler_output.split(image_grid_thw.prod(-1).tolist())
                                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling2/modeling_inkling2.py:1704 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling2/modeling_inkling2.py:1587 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/inkling2/modeling_inkling2.py:1490 in get_image_features
```
</details>

### 10. Dynamic shape operator

- **Location:** `transformers/models/inkling2/modeling_inkling2.py:1515`  (hit ×1)

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
transformers/models/inkling2/modeling_inkling2.py:1704 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling2/modeling_inkling2.py:1593 in forward
transformers/models/inkling2/modeling_inkling2.py:1515 in get_placeholder_mask
```
</details>
