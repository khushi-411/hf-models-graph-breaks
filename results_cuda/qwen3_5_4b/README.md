# qwen3_5_4b — graph breaks

- **HF id:** `Qwen/Qwen3.5-4B`
- **Class:** `AutoModelForImageTextToText`
- **Config overrides:** `{'text_config.num_hidden_layers': 4, 'text_config.layer_types': ['linear_attention', 'linear_attention', 'linear_attention', 'full_attention'], 'text_config.hidden_size': 512, 'text_config.intermediate_size': 1024, 'text_config.num_attention_heads': 4, 'text_config.num_key_value_heads': 2, 'text_config.head_dim': 64, 'text_config.linear_num_key_heads': 4, 'text_config.linear_num_value_heads': 8, 'text_config.linear_key_head_dim': 32, 'text_config.linear_value_head_dim': 32, 'vision_config.depth': 2, 'vision_config.out_hidden_size': 512}`
- **torch:** `2.16.0a0+git57f02f8`  |  **transformers:** `5.24.0.dev0` (git `046bdb8cc2`)
- **Notes:** Top-100 #98. Qwen3.5 (same code as qwen3_8). CUDA only (gated delta net needs triangular_solve).

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ok | 30 | 29 | 15 |
| eval_with_cache | ok | 30 | 29 | 15 |
| train_forward | ok | 30 | 29 | 15 |

## eval_forward

### 1. Dynamic shape operator

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
transformers/models/qwen3_5/modeling_qwen3_5.py:1833 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1602 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/qwen3_5/modeling_qwen3_5.py:1476 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/qwen3_5/modeling_qwen3_5.py:1167 in forward
transformers/vision_utils.py:276 in get_vision_interpolation_indices_and_weights
```
</details>

### 2. Dynamic shape operator

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
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1833 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1602 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/qwen3_5/modeling_qwen3_5.py:1476 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/qwen3_5/modeling_qwen3_5.py:1167 in forward
transformers/vision_utils.py:277 in get_vision_interpolation_indices_and_weights
```
</details>

### 3. Dynamic shape operator

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
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1833 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1602 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/qwen3_5/modeling_qwen3_5.py:1476 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/qwen3_5/modeling_qwen3_5.py:1167 in forward
transformers/vision_utils.py:278 in get_vision_interpolation_indices_and_weights
```
</details>

### 4. Data dependent operator

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
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1833 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1602 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/qwen3_5/modeling_qwen3_5.py:1476 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/qwen3_5/modeling_qwen3_5.py:1167 in forward
transformers/vision_utils.py:280 in get_vision_interpolation_indices_and_weights
```
</details>

### 5. Data dependent operator

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
transformers/models/qwen3_5/modeling_qwen3_5.py:1833 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1602 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/qwen3_5/modeling_qwen3_5.py:1476 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/qwen3_5/modeling_qwen3_5.py:1175 in forward
transformers/vision_utils.py:112 in get_vision_position_ids
```
</details>

### 6. Dynamic shape operator

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
transformers/models/qwen3_5/modeling_qwen3_5.py:1833 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1602 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/qwen3_5/modeling_qwen3_5.py:1476 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/qwen3_5/modeling_qwen3_5.py:1176 in forward
transformers/vision_utils.py:76 in get_vision_attention_seqlens
transformers/vision_utils.py:64 in get_vision_cu_seqlens
```
</details>

### 7. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/qwen3_5/modeling_qwen3_5.py:1063`  (hit ×1)

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

  Developer debug context: frame skipped: torch_dynamo_resume_in_forward_at_1176 (/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/qwen3_5/modeling_qwen3_5.py line 1176)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  DictValuesIterator() originated from:
  File "/admin/home/khushi_agrawal/Documents/pytorch/torch/nn/modules/container.py", line 416
            return iter(self._modules.values())
                   ~~~~^^^^^^^^^^^^^^^^^^^^^^^^
  NullVariable originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1186
            for blk in self.blocks:
                       ^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html

<details><summary>User stack</summary>

```
transformers/models/qwen3_5/modeling_qwen3_5.py:1833 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1602 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/qwen3_5/modeling_qwen3_5.py:1476 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/qwen3_5/modeling_qwen3_5.py:1187 in forward
transformers/modeling_layers.py:124 in forward_with_checkpointing
transformers/models/qwen3_5/modeling_qwen3_5.py:1063 in forward
```
</details>

### 8. Data dependent operator

- **Location:** `transformers/models/qwen3_5/modeling_qwen3_5.py:1063`  (hit ×1)

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
transformers/models/qwen3_5/modeling_qwen3_5.py:1833 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1602 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/qwen3_5/modeling_qwen3_5.py:1476 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/qwen3_5/modeling_qwen3_5.py:1187 in forward
transformers/modeling_layers.py:124 in forward_with_checkpointing
transformers/models/qwen3_5/modeling_qwen3_5.py:1063 in forward
```
</details>

### 9. Data dependent operator

- **Location:** `transformers/models/qwen3_5/modeling_qwen3_5.py:1477`  (hit ×1)

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
transformers/models/qwen3_5/modeling_qwen3_5.py:1833 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1602 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/qwen3_5/modeling_qwen3_5.py:1477 in get_image_features
```
</details>

### 10. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/utils/import_utils.py:1993`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_KW: a function call with keyword arguments, e.g. f(x=True):

Unsupported Tensor.item() call with capture_scalar_outputs=False
  Explanation: Dynamo does not support tracing `Tensor.item()` with config.capture_scalar_outputs=False.
  Hint: Set `torch._dynamo.config.capture_scalar_outputs = True` or `export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` to include these operations in the captured graph.

  Developer debug context: call_method TensorVariable() item () {}

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1833 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1615 in forward
transformers/models/qwen3_5/modeling_qwen3_5.py:1509 in get_placeholder_mask
transformers/utils/import_utils.py:1993 in torch_compilable_check
```
</details>

### 11. Dynamic shape operator

- **Location:** `transformers/models/qwen3_5/modeling_qwen3_5.py:1417`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_KW: a function call with keyword arguments, e.g. f(x=True):

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
transformers/models/qwen3_5/modeling_qwen3_5.py:1833 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1630 in forward
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1545 in compute_3d_position_ids
transformers/models/qwen3_5/modeling_qwen3_5.py:1417 in get_rope_index
```
</details>

### 12. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/qwen3_5/modeling_qwen3_5.py:1417`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Dynamic shape operator
  Explanation: Operator `aten.nonzero.default`'s output shape depends on input Tensor data.
  Hint: Enable tracing of dynamic shape operators with `torch._dynamo.config.capture_dynamic_output_shape_ops = True`

  Developer debug context: aten.nonzero.default

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

*** While handling this graph break, another graph break occurred: ***

graph break in loop
  Explanation: torch.compile detected a graph break in a for/while loop. Skipping the frame and falling back to eager, as graph breaks in loops are not supported.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: frame skipped: get_rope_index (/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/qwen3_5/modeling_qwen3_5.py line 1361)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  LocalGeneratorObjectVariable(enumerate) originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1414
            for batch_idx, current_input_ids in enumerate(input_ids):
                                                ~~~~~~~~~^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1833 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1630 in forward
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1545 in compute_3d_position_ids
transformers/models/qwen3_5/modeling_qwen3_5.py:1417 in get_rope_index
```
</details>

### 13. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/models/qwen3_5/modeling_qwen3_5.py:1347`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Unsupported Tensor.item() call with capture_scalar_outputs=False
  Explanation: Dynamo does not support tracing `Tensor.item()` with config.capture_scalar_outputs=False.
  Hint: Set `torch._dynamo.config.capture_scalar_outputs = True` or `export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` to include these operations in the captured graph.

  Developer debug context: call_method TensorVariable() item () {}

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1833 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1630 in forward
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1545 in compute_3d_position_ids
transformers/models/qwen3_5/modeling_qwen3_5.py:1440 in get_rope_index
transformers/models/qwen3_5/modeling_qwen3_5.py:1347 in get_vision_position_ids
```
</details>

### 14. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/models/qwen3_5/modeling_qwen3_5.py:1348`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Unsupported Tensor.item() call with capture_scalar_outputs=False
  Explanation: Dynamo does not support tracing `Tensor.item()` with config.capture_scalar_outputs=False.
  Hint: Set `torch._dynamo.config.capture_scalar_outputs = True` or `export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` to include these operations in the captured graph.

  Developer debug context: call_method TensorVariable() item () {}

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

Stack variable source attribution:
  ConstantVariable(int: 1) originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1347
                grid_thw[0].item() // temp_merge_size,
                ~~~~~~~~~~~~~~~~~~~^^~~~~~~~~~~~~~~~~
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1833 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1630 in forward
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1545 in compute_3d_position_ids
transformers/models/qwen3_5/modeling_qwen3_5.py:1440 in get_rope_index
transformers/models/qwen3_5/modeling_qwen3_5.py:1348 in get_vision_position_ids
```
</details>

### 15. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/models/qwen3_5/modeling_qwen3_5.py:1349`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Unsupported Tensor.item() call with capture_scalar_outputs=False
  Explanation: Dynamo does not support tracing `Tensor.item()` with config.capture_scalar_outputs=False.
  Hint: Set `torch._dynamo.config.capture_scalar_outputs = True` or `export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` to include these operations in the captured graph.

  Developer debug context: call_method TensorVariable() item () {}

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

Stack variable source attribution:
  LazyVariableTracker(unrealized: <class 'int'>) originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1348
                grid_thw[1].item() // spatial_merge_size,
  ConstantVariable(int: 4) originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1348
                grid_thw[1].item() // spatial_merge_size,
                ~~~~~~~~~~~~~~~~~~~^^~~~~~~~~~~~~~~~~~~~
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1833 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1630 in forward
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1545 in compute_3d_position_ids
transformers/models/qwen3_5/modeling_qwen3_5.py:1440 in get_rope_index
transformers/models/qwen3_5/modeling_qwen3_5.py:1349 in get_vision_position_ids
```
</details>


## eval_with_cache

### 1. Dynamic shape operator

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
transformers/models/qwen3_5/modeling_qwen3_5.py:1833 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1602 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/qwen3_5/modeling_qwen3_5.py:1476 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/qwen3_5/modeling_qwen3_5.py:1167 in forward
transformers/vision_utils.py:276 in get_vision_interpolation_indices_and_weights
```
</details>

### 2. Dynamic shape operator

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
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1833 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1602 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/qwen3_5/modeling_qwen3_5.py:1476 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/qwen3_5/modeling_qwen3_5.py:1167 in forward
transformers/vision_utils.py:277 in get_vision_interpolation_indices_and_weights
```
</details>

### 3. Dynamic shape operator

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
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1833 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1602 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/qwen3_5/modeling_qwen3_5.py:1476 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/qwen3_5/modeling_qwen3_5.py:1167 in forward
transformers/vision_utils.py:278 in get_vision_interpolation_indices_and_weights
```
</details>

### 4. Data dependent operator

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
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1833 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1602 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/qwen3_5/modeling_qwen3_5.py:1476 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/qwen3_5/modeling_qwen3_5.py:1167 in forward
transformers/vision_utils.py:280 in get_vision_interpolation_indices_and_weights
```
</details>

### 5. Data dependent operator

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
transformers/models/qwen3_5/modeling_qwen3_5.py:1833 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1602 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/qwen3_5/modeling_qwen3_5.py:1476 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/qwen3_5/modeling_qwen3_5.py:1175 in forward
transformers/vision_utils.py:112 in get_vision_position_ids
```
</details>

### 6. Dynamic shape operator

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
transformers/models/qwen3_5/modeling_qwen3_5.py:1833 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1602 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/qwen3_5/modeling_qwen3_5.py:1476 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/qwen3_5/modeling_qwen3_5.py:1176 in forward
transformers/vision_utils.py:76 in get_vision_attention_seqlens
transformers/vision_utils.py:64 in get_vision_cu_seqlens
```
</details>

### 7. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/qwen3_5/modeling_qwen3_5.py:1063`  (hit ×1)

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

  Developer debug context: frame skipped: torch_dynamo_resume_in_forward_at_1176 (/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/qwen3_5/modeling_qwen3_5.py line 1176)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  DictValuesIterator() originated from:
  File "/admin/home/khushi_agrawal/Documents/pytorch/torch/nn/modules/container.py", line 416
            return iter(self._modules.values())
                   ~~~~^^^^^^^^^^^^^^^^^^^^^^^^
  NullVariable originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1186
            for blk in self.blocks:
                       ^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html

<details><summary>User stack</summary>

```
transformers/models/qwen3_5/modeling_qwen3_5.py:1833 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1602 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/qwen3_5/modeling_qwen3_5.py:1476 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/qwen3_5/modeling_qwen3_5.py:1187 in forward
transformers/modeling_layers.py:124 in forward_with_checkpointing
transformers/models/qwen3_5/modeling_qwen3_5.py:1063 in forward
```
</details>

### 8. Data dependent operator

- **Location:** `transformers/models/qwen3_5/modeling_qwen3_5.py:1063`  (hit ×1)

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
transformers/models/qwen3_5/modeling_qwen3_5.py:1833 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1602 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/qwen3_5/modeling_qwen3_5.py:1476 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/qwen3_5/modeling_qwen3_5.py:1187 in forward
transformers/modeling_layers.py:124 in forward_with_checkpointing
transformers/models/qwen3_5/modeling_qwen3_5.py:1063 in forward
```
</details>

### 9. Data dependent operator

- **Location:** `transformers/models/qwen3_5/modeling_qwen3_5.py:1477`  (hit ×1)

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
transformers/models/qwen3_5/modeling_qwen3_5.py:1833 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1602 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/qwen3_5/modeling_qwen3_5.py:1477 in get_image_features
```
</details>

### 10. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/utils/import_utils.py:1993`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_KW: a function call with keyword arguments, e.g. f(x=True):

Unsupported Tensor.item() call with capture_scalar_outputs=False
  Explanation: Dynamo does not support tracing `Tensor.item()` with config.capture_scalar_outputs=False.
  Hint: Set `torch._dynamo.config.capture_scalar_outputs = True` or `export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` to include these operations in the captured graph.

  Developer debug context: call_method TensorVariable() item () {}

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1833 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1615 in forward
transformers/models/qwen3_5/modeling_qwen3_5.py:1509 in get_placeholder_mask
transformers/utils/import_utils.py:1993 in torch_compilable_check
```
</details>

### 11. Dynamic shape operator

- **Location:** `transformers/models/qwen3_5/modeling_qwen3_5.py:1417`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_KW: a function call with keyword arguments, e.g. f(x=True):

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
transformers/models/qwen3_5/modeling_qwen3_5.py:1833 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1630 in forward
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1545 in compute_3d_position_ids
transformers/models/qwen3_5/modeling_qwen3_5.py:1417 in get_rope_index
```
</details>

### 12. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/qwen3_5/modeling_qwen3_5.py:1417`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Dynamic shape operator
  Explanation: Operator `aten.nonzero.default`'s output shape depends on input Tensor data.
  Hint: Enable tracing of dynamic shape operators with `torch._dynamo.config.capture_dynamic_output_shape_ops = True`

  Developer debug context: aten.nonzero.default

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

*** While handling this graph break, another graph break occurred: ***

graph break in loop
  Explanation: torch.compile detected a graph break in a for/while loop. Skipping the frame and falling back to eager, as graph breaks in loops are not supported.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: frame skipped: get_rope_index (/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/qwen3_5/modeling_qwen3_5.py line 1361)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  LocalGeneratorObjectVariable(enumerate) originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1414
            for batch_idx, current_input_ids in enumerate(input_ids):
                                                ~~~~~~~~~^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1833 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1630 in forward
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1545 in compute_3d_position_ids
transformers/models/qwen3_5/modeling_qwen3_5.py:1417 in get_rope_index
```
</details>

### 13. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/models/qwen3_5/modeling_qwen3_5.py:1347`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Unsupported Tensor.item() call with capture_scalar_outputs=False
  Explanation: Dynamo does not support tracing `Tensor.item()` with config.capture_scalar_outputs=False.
  Hint: Set `torch._dynamo.config.capture_scalar_outputs = True` or `export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` to include these operations in the captured graph.

  Developer debug context: call_method TensorVariable() item () {}

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1833 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1630 in forward
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1545 in compute_3d_position_ids
transformers/models/qwen3_5/modeling_qwen3_5.py:1440 in get_rope_index
transformers/models/qwen3_5/modeling_qwen3_5.py:1347 in get_vision_position_ids
```
</details>

### 14. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/models/qwen3_5/modeling_qwen3_5.py:1348`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Unsupported Tensor.item() call with capture_scalar_outputs=False
  Explanation: Dynamo does not support tracing `Tensor.item()` with config.capture_scalar_outputs=False.
  Hint: Set `torch._dynamo.config.capture_scalar_outputs = True` or `export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` to include these operations in the captured graph.

  Developer debug context: call_method TensorVariable() item () {}

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

Stack variable source attribution:
  ConstantVariable(int: 1) originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1347
                grid_thw[0].item() // temp_merge_size,
                ~~~~~~~~~~~~~~~~~~~^^~~~~~~~~~~~~~~~~
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1833 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1630 in forward
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1545 in compute_3d_position_ids
transformers/models/qwen3_5/modeling_qwen3_5.py:1440 in get_rope_index
transformers/models/qwen3_5/modeling_qwen3_5.py:1348 in get_vision_position_ids
```
</details>

### 15. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/models/qwen3_5/modeling_qwen3_5.py:1349`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Unsupported Tensor.item() call with capture_scalar_outputs=False
  Explanation: Dynamo does not support tracing `Tensor.item()` with config.capture_scalar_outputs=False.
  Hint: Set `torch._dynamo.config.capture_scalar_outputs = True` or `export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` to include these operations in the captured graph.

  Developer debug context: call_method TensorVariable() item () {}

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

Stack variable source attribution:
  LazyVariableTracker(unrealized: <class 'int'>) originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1348
                grid_thw[1].item() // spatial_merge_size,
  ConstantVariable(int: 4) originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1348
                grid_thw[1].item() // spatial_merge_size,
                ~~~~~~~~~~~~~~~~~~~^^~~~~~~~~~~~~~~~~~~~
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1833 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1630 in forward
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1545 in compute_3d_position_ids
transformers/models/qwen3_5/modeling_qwen3_5.py:1440 in get_rope_index
transformers/models/qwen3_5/modeling_qwen3_5.py:1349 in get_vision_position_ids
```
</details>


## train_forward

### 1. Dynamic shape operator

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
transformers/models/qwen3_5/modeling_qwen3_5.py:1833 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1602 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/qwen3_5/modeling_qwen3_5.py:1476 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/qwen3_5/modeling_qwen3_5.py:1167 in forward
transformers/vision_utils.py:276 in get_vision_interpolation_indices_and_weights
```
</details>

### 2. Dynamic shape operator

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
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1833 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1602 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/qwen3_5/modeling_qwen3_5.py:1476 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/qwen3_5/modeling_qwen3_5.py:1167 in forward
transformers/vision_utils.py:277 in get_vision_interpolation_indices_and_weights
```
</details>

### 3. Dynamic shape operator

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
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1833 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1602 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/qwen3_5/modeling_qwen3_5.py:1476 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/qwen3_5/modeling_qwen3_5.py:1167 in forward
transformers/vision_utils.py:278 in get_vision_interpolation_indices_and_weights
```
</details>

### 4. Data dependent operator

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
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1833 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1602 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/qwen3_5/modeling_qwen3_5.py:1476 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/qwen3_5/modeling_qwen3_5.py:1167 in forward
transformers/vision_utils.py:280 in get_vision_interpolation_indices_and_weights
```
</details>

### 5. Data dependent operator

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
transformers/models/qwen3_5/modeling_qwen3_5.py:1833 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1602 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/qwen3_5/modeling_qwen3_5.py:1476 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/qwen3_5/modeling_qwen3_5.py:1175 in forward
transformers/vision_utils.py:112 in get_vision_position_ids
```
</details>

### 6. Dynamic shape operator

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
transformers/models/qwen3_5/modeling_qwen3_5.py:1833 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1602 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/qwen3_5/modeling_qwen3_5.py:1476 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/qwen3_5/modeling_qwen3_5.py:1176 in forward
transformers/vision_utils.py:76 in get_vision_attention_seqlens
transformers/vision_utils.py:64 in get_vision_cu_seqlens
```
</details>

### 7. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/qwen3_5/modeling_qwen3_5.py:1063`  (hit ×1)

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

  Developer debug context: frame skipped: torch_dynamo_resume_in_forward_at_1176 (/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/qwen3_5/modeling_qwen3_5.py line 1176)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  DictValuesIterator() originated from:
  File "/admin/home/khushi_agrawal/Documents/pytorch/torch/nn/modules/container.py", line 416
            return iter(self._modules.values())
                   ~~~~^^^^^^^^^^^^^^^^^^^^^^^^
  NullVariable originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1186
            for blk in self.blocks:
                       ^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html

<details><summary>User stack</summary>

```
transformers/models/qwen3_5/modeling_qwen3_5.py:1833 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1602 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/qwen3_5/modeling_qwen3_5.py:1476 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/qwen3_5/modeling_qwen3_5.py:1187 in forward
transformers/modeling_layers.py:124 in forward_with_checkpointing
transformers/models/qwen3_5/modeling_qwen3_5.py:1063 in forward
```
</details>

### 8. Data dependent operator

- **Location:** `transformers/models/qwen3_5/modeling_qwen3_5.py:1063`  (hit ×1)

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
transformers/models/qwen3_5/modeling_qwen3_5.py:1833 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1602 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/qwen3_5/modeling_qwen3_5.py:1476 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/qwen3_5/modeling_qwen3_5.py:1187 in forward
transformers/modeling_layers.py:124 in forward_with_checkpointing
transformers/models/qwen3_5/modeling_qwen3_5.py:1063 in forward
```
</details>

### 9. Data dependent operator

- **Location:** `transformers/models/qwen3_5/modeling_qwen3_5.py:1477`  (hit ×1)

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
transformers/models/qwen3_5/modeling_qwen3_5.py:1833 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1602 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/qwen3_5/modeling_qwen3_5.py:1477 in get_image_features
```
</details>

### 10. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/utils/import_utils.py:1993`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_KW: a function call with keyword arguments, e.g. f(x=True):

Unsupported Tensor.item() call with capture_scalar_outputs=False
  Explanation: Dynamo does not support tracing `Tensor.item()` with config.capture_scalar_outputs=False.
  Hint: Set `torch._dynamo.config.capture_scalar_outputs = True` or `export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` to include these operations in the captured graph.

  Developer debug context: call_method TensorVariable() item () {}

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1833 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1615 in forward
transformers/models/qwen3_5/modeling_qwen3_5.py:1509 in get_placeholder_mask
transformers/utils/import_utils.py:1993 in torch_compilable_check
```
</details>

### 11. Dynamic shape operator

- **Location:** `transformers/models/qwen3_5/modeling_qwen3_5.py:1417`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_KW: a function call with keyword arguments, e.g. f(x=True):

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
transformers/models/qwen3_5/modeling_qwen3_5.py:1833 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1630 in forward
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1545 in compute_3d_position_ids
transformers/models/qwen3_5/modeling_qwen3_5.py:1417 in get_rope_index
```
</details>

### 12. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/qwen3_5/modeling_qwen3_5.py:1417`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Dynamic shape operator
  Explanation: Operator `aten.nonzero.default`'s output shape depends on input Tensor data.
  Hint: Enable tracing of dynamic shape operators with `torch._dynamo.config.capture_dynamic_output_shape_ops = True`

  Developer debug context: aten.nonzero.default

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

*** While handling this graph break, another graph break occurred: ***

graph break in loop
  Explanation: torch.compile detected a graph break in a for/while loop. Skipping the frame and falling back to eager, as graph breaks in loops are not supported.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: frame skipped: get_rope_index (/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/qwen3_5/modeling_qwen3_5.py line 1361)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  LocalGeneratorObjectVariable(enumerate) originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1414
            for batch_idx, current_input_ids in enumerate(input_ids):
                                                ~~~~~~~~~^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1833 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1630 in forward
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1545 in compute_3d_position_ids
transformers/models/qwen3_5/modeling_qwen3_5.py:1417 in get_rope_index
```
</details>

### 13. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/models/qwen3_5/modeling_qwen3_5.py:1347`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Unsupported Tensor.item() call with capture_scalar_outputs=False
  Explanation: Dynamo does not support tracing `Tensor.item()` with config.capture_scalar_outputs=False.
  Hint: Set `torch._dynamo.config.capture_scalar_outputs = True` or `export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` to include these operations in the captured graph.

  Developer debug context: call_method TensorVariable() item () {}

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1833 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1630 in forward
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1545 in compute_3d_position_ids
transformers/models/qwen3_5/modeling_qwen3_5.py:1440 in get_rope_index
transformers/models/qwen3_5/modeling_qwen3_5.py:1347 in get_vision_position_ids
```
</details>

### 14. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/models/qwen3_5/modeling_qwen3_5.py:1348`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Unsupported Tensor.item() call with capture_scalar_outputs=False
  Explanation: Dynamo does not support tracing `Tensor.item()` with config.capture_scalar_outputs=False.
  Hint: Set `torch._dynamo.config.capture_scalar_outputs = True` or `export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` to include these operations in the captured graph.

  Developer debug context: call_method TensorVariable() item () {}

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

Stack variable source attribution:
  ConstantVariable(int: 1) originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1347
                grid_thw[0].item() // temp_merge_size,
                ~~~~~~~~~~~~~~~~~~~^^~~~~~~~~~~~~~~~~
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1833 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1630 in forward
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1545 in compute_3d_position_ids
transformers/models/qwen3_5/modeling_qwen3_5.py:1440 in get_rope_index
transformers/models/qwen3_5/modeling_qwen3_5.py:1348 in get_vision_position_ids
```
</details>

### 15. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/models/qwen3_5/modeling_qwen3_5.py:1349`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Unsupported Tensor.item() call with capture_scalar_outputs=False
  Explanation: Dynamo does not support tracing `Tensor.item()` with config.capture_scalar_outputs=False.
  Hint: Set `torch._dynamo.config.capture_scalar_outputs = True` or `export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` to include these operations in the captured graph.

  Developer debug context: call_method TensorVariable() item () {}

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

Stack variable source attribution:
  LazyVariableTracker(unrealized: <class 'int'>) originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1348
                grid_thw[1].item() // spatial_merge_size,
  ConstantVariable(int: 4) originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1348
                grid_thw[1].item() // spatial_merge_size,
                ~~~~~~~~~~~~~~~~~~~^^~~~~~~~~~~~~~~~~~~~
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1833 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1630 in forward
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen3_5/modeling_qwen3_5.py:1545 in compute_3d_position_ids
transformers/models/qwen3_5/modeling_qwen3_5.py:1440 in get_rope_index
transformers/models/qwen3_5/modeling_qwen3_5.py:1349 in get_vision_position_ids
```
</details>
