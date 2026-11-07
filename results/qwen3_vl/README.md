# qwen3_vl — graph breaks

- **HF id:** `Qwen/Qwen3-VL-2B-Instruct`
- **Class:** `AutoModelForImageTextToText`
- **Config overrides:** `{'text_config.num_hidden_layers': 2, 'text_config.hidden_size': 512, 'text_config.intermediate_size': 1024, 'text_config.num_attention_heads': 4, 'text_config.num_key_value_heads': 2, 'vision_config.depth': 2, 'vision_config.deepstack_visual_indexes': [0], 'vision_config.out_hidden_size': 512}`
- **torch:** `2.16.0a0+git9933f27`  |  **transformers:** `5.22.1`

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ok | 36 | 35 | 17 |
| eval_with_cache | ok | 37 | 36 | 17 |
| train_forward | ok | 37 | 36 | 17 |

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
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1364 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1174 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1040 in get_image_features
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:690 in forward
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
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1364 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1174 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1040 in get_image_features
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:690 in forward
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
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1364 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1174 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1040 in get_image_features
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:690 in forward
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
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1364 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1174 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1040 in get_image_features
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:690 in forward
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
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1364 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1174 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1040 in get_image_features
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:698 in forward
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
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1364 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1174 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1040 in get_image_features
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:699 in forward
transformers/vision_utils.py:76 in get_vision_attention_seqlens
transformers/vision_utils.py:64 in get_vision_cu_seqlens
```
</details>

### 7. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/qwen3_vl/modeling_qwen3_vl.py:299`  (hit ×1)

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

  Developer debug context: frame skipped: torch_dynamo_resume_in_forward_at_699 (/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py line 699)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  LocalGeneratorObjectVariable(enumerate) originated from:
  File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py", line 710
            for layer_num, blk in enumerate(self.blocks):
                                  ~~~~~~~~~^^^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1364 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1174 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1040 in get_image_features
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:711 in forward
transformers/modeling_layers.py:111 in __call__
transformers/models/qwen3_vl/modeling_qwen3_vl.py:299 in forward
```
</details>

### 8. Data dependent operator

- **Location:** `transformers/models/qwen3_vl/modeling_qwen3_vl.py:299`  (hit ×1)

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
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1364 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1174 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1040 in get_image_features
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:711 in forward
transformers/modeling_layers.py:111 in __call__
transformers/models/qwen3_vl/modeling_qwen3_vl.py:299 in forward
```
</details>

### 9. Data dependent operator

- **Location:** `transformers/models/qwen3_vl/modeling_qwen3_vl.py:1044`  (hit ×1)

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
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1364 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1174 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1044 in get_image_features
```
</details>

### 10. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/utils/import_utils.py:1897`  (hit ×1)

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
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1364 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1187 in forward
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1078 in get_placeholder_mask
transformers/utils/import_utils.py:1897 in torch_compilable_check
```
</details>

### 11. Dynamic shape operator

- **Location:** `transformers/models/qwen3_vl/modeling_qwen3_vl.py:981`  (hit ×1)

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
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1364 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1229 in forward
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1113 in compute_3d_position_ids
transformers/models/qwen3_vl/modeling_qwen3_vl.py:981 in get_rope_index
```
</details>

### 12. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/qwen3_vl/modeling_qwen3_vl.py:981`  (hit ×1)

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

  Developer debug context: frame skipped: get_rope_index (/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py line 925)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  LocalGeneratorObjectVariable(enumerate) originated from:
  File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py", line 978
            for batch_idx, current_input_ids in enumerate(input_ids):
                                                ~~~~~~~~~^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1364 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1229 in forward
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1113 in compute_3d_position_ids
transformers/models/qwen3_vl/modeling_qwen3_vl.py:981 in get_rope_index
```
</details>

### 13. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/models/qwen3_vl/modeling_qwen3_vl.py:911`  (hit ×1)

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
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1364 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1229 in forward
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1113 in compute_3d_position_ids
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1004 in get_rope_index
transformers/models/qwen3_vl/modeling_qwen3_vl.py:911 in get_vision_position_ids
```
</details>

### 14. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/models/qwen3_vl/modeling_qwen3_vl.py:912`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Unsupported Tensor.item() call with capture_scalar_outputs=False
  Explanation: Dynamo does not support tracing `Tensor.item()` with config.capture_scalar_outputs=False.
  Hint: Set `torch._dynamo.config.capture_scalar_outputs = True` or `export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` to include these operations in the captured graph.

  Developer debug context: call_method TensorVariable() item () {}

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

Stack variable source attribution:
  ConstantVariable(int: 1) originated from:
  File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py", line 911
                grid_thw[0].item() // temp_merge_size,
                ~~~~~~~~~~~~~~~~~~~^^~~~~~~~~~~~~~~~~
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1364 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1229 in forward
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1113 in compute_3d_position_ids
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1004 in get_rope_index
transformers/models/qwen3_vl/modeling_qwen3_vl.py:912 in get_vision_position_ids
```
</details>

### 15. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/models/qwen3_vl/modeling_qwen3_vl.py:913`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Unsupported Tensor.item() call with capture_scalar_outputs=False
  Explanation: Dynamo does not support tracing `Tensor.item()` with config.capture_scalar_outputs=False.
  Hint: Set `torch._dynamo.config.capture_scalar_outputs = True` or `export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` to include these operations in the captured graph.

  Developer debug context: call_method TensorVariable() item () {}

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

Stack variable source attribution:
  LazyVariableTracker(unrealized: <class 'int'>) originated from:
  File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py", line 912
                grid_thw[1].item() // spatial_merge_size,
  ConstantVariable(int: 4) originated from:
  File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py", line 912
                grid_thw[1].item() // spatial_merge_size,
                ~~~~~~~~~~~~~~~~~~~^^~~~~~~~~~~~~~~~~~~~
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1364 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1229 in forward
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1113 in compute_3d_position_ids
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1004 in get_rope_index
transformers/models/qwen3_vl/modeling_qwen3_vl.py:913 in get_vision_position_ids
```
</details>

### 16. Dynamic shape operator

- **Location:** `transformers/models/qwen3_vl/modeling_qwen3_vl.py:853`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_FUNCTION_EX: a variadic function call, e.g. f(*args):

Dynamic shape operator
  Explanation: Operator `aten.nonzero.default`'s output shape depends on input Tensor data.
  Hint: Enable tracing of dynamic shape operators with `torch._dynamo.config.capture_dynamic_output_shape_ops = True`

  Developer debug context: aten.nonzero.default

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1364 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1239 in forward
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:834 in forward
transformers/models/qwen3_vl/modeling_qwen3_vl.py:853 in _deepstack_process
```
</details>

### 17. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/qwen3_vl/modeling_qwen3_vl.py:853`  (hit ×1)

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

  Developer debug context: frame skipped: forward (/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py line 759)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  LocalGeneratorObjectVariable(enumerate) originated from:
  File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py", line 821
            for layer_idx, decoder_layer in enumerate(self.layers):
                                            ~~~~~~~~~^^^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1364 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1239 in forward
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:834 in forward
transformers/models/qwen3_vl/modeling_qwen3_vl.py:853 in _deepstack_process
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
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1364 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1174 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1040 in get_image_features
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:690 in forward
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
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1364 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1174 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1040 in get_image_features
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:690 in forward
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
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1364 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1174 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1040 in get_image_features
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:690 in forward
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
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1364 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1174 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1040 in get_image_features
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:690 in forward
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
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1364 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1174 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1040 in get_image_features
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:698 in forward
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
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1364 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1174 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1040 in get_image_features
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:699 in forward
transformers/vision_utils.py:76 in get_vision_attention_seqlens
transformers/vision_utils.py:64 in get_vision_cu_seqlens
```
</details>

### 7. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/qwen3_vl/modeling_qwen3_vl.py:299`  (hit ×1)

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

  Developer debug context: frame skipped: torch_dynamo_resume_in_forward_at_699 (/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py line 699)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  LocalGeneratorObjectVariable(enumerate) originated from:
  File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py", line 710
            for layer_num, blk in enumerate(self.blocks):
                                  ~~~~~~~~~^^^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1364 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1174 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1040 in get_image_features
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:711 in forward
transformers/modeling_layers.py:111 in __call__
transformers/models/qwen3_vl/modeling_qwen3_vl.py:299 in forward
```
</details>

### 8. Data dependent operator

- **Location:** `transformers/models/qwen3_vl/modeling_qwen3_vl.py:299`  (hit ×1)

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
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1364 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1174 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1040 in get_image_features
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:711 in forward
transformers/modeling_layers.py:111 in __call__
transformers/models/qwen3_vl/modeling_qwen3_vl.py:299 in forward
```
</details>

### 9. Data dependent operator

- **Location:** `transformers/models/qwen3_vl/modeling_qwen3_vl.py:1044`  (hit ×1)

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
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1364 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1174 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1044 in get_image_features
```
</details>

### 10. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/utils/import_utils.py:1897`  (hit ×1)

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
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1364 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1187 in forward
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1078 in get_placeholder_mask
transformers/utils/import_utils.py:1897 in torch_compilable_check
```
</details>

### 11. Dynamic shape operator

- **Location:** `transformers/models/qwen3_vl/modeling_qwen3_vl.py:981`  (hit ×1)

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
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1364 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1229 in forward
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1113 in compute_3d_position_ids
transformers/models/qwen3_vl/modeling_qwen3_vl.py:981 in get_rope_index
```
</details>

### 12. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/qwen3_vl/modeling_qwen3_vl.py:981`  (hit ×1)

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

  Developer debug context: frame skipped: get_rope_index (/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py line 925)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  LocalGeneratorObjectVariable(enumerate) originated from:
  File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py", line 978
            for batch_idx, current_input_ids in enumerate(input_ids):
                                                ~~~~~~~~~^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1364 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1229 in forward
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1113 in compute_3d_position_ids
transformers/models/qwen3_vl/modeling_qwen3_vl.py:981 in get_rope_index
```
</details>

### 13. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/models/qwen3_vl/modeling_qwen3_vl.py:911`  (hit ×1)

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
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1364 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1229 in forward
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1113 in compute_3d_position_ids
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1004 in get_rope_index
transformers/models/qwen3_vl/modeling_qwen3_vl.py:911 in get_vision_position_ids
```
</details>

### 14. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/models/qwen3_vl/modeling_qwen3_vl.py:912`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Unsupported Tensor.item() call with capture_scalar_outputs=False
  Explanation: Dynamo does not support tracing `Tensor.item()` with config.capture_scalar_outputs=False.
  Hint: Set `torch._dynamo.config.capture_scalar_outputs = True` or `export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` to include these operations in the captured graph.

  Developer debug context: call_method TensorVariable() item () {}

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

Stack variable source attribution:
  ConstantVariable(int: 1) originated from:
  File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py", line 911
                grid_thw[0].item() // temp_merge_size,
                ~~~~~~~~~~~~~~~~~~~^^~~~~~~~~~~~~~~~~
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1364 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1229 in forward
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1113 in compute_3d_position_ids
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1004 in get_rope_index
transformers/models/qwen3_vl/modeling_qwen3_vl.py:912 in get_vision_position_ids
```
</details>

### 15. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/models/qwen3_vl/modeling_qwen3_vl.py:913`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Unsupported Tensor.item() call with capture_scalar_outputs=False
  Explanation: Dynamo does not support tracing `Tensor.item()` with config.capture_scalar_outputs=False.
  Hint: Set `torch._dynamo.config.capture_scalar_outputs = True` or `export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` to include these operations in the captured graph.

  Developer debug context: call_method TensorVariable() item () {}

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

Stack variable source attribution:
  LazyVariableTracker(unrealized: <class 'int'>) originated from:
  File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py", line 912
                grid_thw[1].item() // spatial_merge_size,
  ConstantVariable(int: 4) originated from:
  File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py", line 912
                grid_thw[1].item() // spatial_merge_size,
                ~~~~~~~~~~~~~~~~~~~^^~~~~~~~~~~~~~~~~~~~
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1364 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1229 in forward
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1113 in compute_3d_position_ids
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1004 in get_rope_index
transformers/models/qwen3_vl/modeling_qwen3_vl.py:913 in get_vision_position_ids
```
</details>

### 16. Dynamic shape operator

- **Location:** `transformers/models/qwen3_vl/modeling_qwen3_vl.py:853`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_FUNCTION_EX: a variadic function call, e.g. f(*args):

Dynamic shape operator
  Explanation: Operator `aten.nonzero.default`'s output shape depends on input Tensor data.
  Hint: Enable tracing of dynamic shape operators with `torch._dynamo.config.capture_dynamic_output_shape_ops = True`

  Developer debug context: aten.nonzero.default

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1364 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1239 in forward
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:834 in forward
transformers/models/qwen3_vl/modeling_qwen3_vl.py:853 in _deepstack_process
```
</details>

### 17. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/qwen3_vl/modeling_qwen3_vl.py:853`  (hit ×1)

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

  Developer debug context: frame skipped: forward (/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py line 759)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  LocalGeneratorObjectVariable(enumerate) originated from:
  File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py", line 821
            for layer_idx, decoder_layer in enumerate(self.layers):
                                            ~~~~~~~~~^^^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1364 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1239 in forward
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:834 in forward
transformers/models/qwen3_vl/modeling_qwen3_vl.py:853 in _deepstack_process
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
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1364 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1174 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1040 in get_image_features
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:690 in forward
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
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1364 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1174 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1040 in get_image_features
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:690 in forward
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
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1364 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1174 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1040 in get_image_features
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:690 in forward
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
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1364 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1174 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1040 in get_image_features
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:690 in forward
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
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1364 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1174 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1040 in get_image_features
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:698 in forward
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
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1364 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1174 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1040 in get_image_features
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:699 in forward
transformers/vision_utils.py:76 in get_vision_attention_seqlens
transformers/vision_utils.py:64 in get_vision_cu_seqlens
```
</details>

### 7. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/qwen3_vl/modeling_qwen3_vl.py:299`  (hit ×1)

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

  Developer debug context: frame skipped: torch_dynamo_resume_in_forward_at_699 (/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py line 699)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  LocalGeneratorObjectVariable(enumerate) originated from:
  File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py", line 710
            for layer_num, blk in enumerate(self.blocks):
                                  ~~~~~~~~~^^^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1364 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1174 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1040 in get_image_features
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:711 in forward
transformers/modeling_layers.py:111 in __call__
transformers/models/qwen3_vl/modeling_qwen3_vl.py:299 in forward
```
</details>

### 8. Data dependent operator

- **Location:** `transformers/models/qwen3_vl/modeling_qwen3_vl.py:299`  (hit ×1)

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
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1364 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1174 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1040 in get_image_features
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:711 in forward
transformers/modeling_layers.py:111 in __call__
transformers/models/qwen3_vl/modeling_qwen3_vl.py:299 in forward
```
</details>

### 9. Data dependent operator

- **Location:** `transformers/models/qwen3_vl/modeling_qwen3_vl.py:1044`  (hit ×1)

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
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1364 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1174 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1044 in get_image_features
```
</details>

### 10. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/utils/import_utils.py:1897`  (hit ×1)

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
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1364 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1187 in forward
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1078 in get_placeholder_mask
transformers/utils/import_utils.py:1897 in torch_compilable_check
```
</details>

### 11. Dynamic shape operator

- **Location:** `transformers/models/qwen3_vl/modeling_qwen3_vl.py:981`  (hit ×1)

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
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1364 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1229 in forward
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1113 in compute_3d_position_ids
transformers/models/qwen3_vl/modeling_qwen3_vl.py:981 in get_rope_index
```
</details>

### 12. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/qwen3_vl/modeling_qwen3_vl.py:981`  (hit ×1)

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

  Developer debug context: frame skipped: get_rope_index (/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py line 925)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  LocalGeneratorObjectVariable(enumerate) originated from:
  File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py", line 978
            for batch_idx, current_input_ids in enumerate(input_ids):
                                                ~~~~~~~~~^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1364 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1229 in forward
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1113 in compute_3d_position_ids
transformers/models/qwen3_vl/modeling_qwen3_vl.py:981 in get_rope_index
```
</details>

### 13. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/models/qwen3_vl/modeling_qwen3_vl.py:911`  (hit ×1)

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
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1364 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1229 in forward
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1113 in compute_3d_position_ids
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1004 in get_rope_index
transformers/models/qwen3_vl/modeling_qwen3_vl.py:911 in get_vision_position_ids
```
</details>

### 14. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/models/qwen3_vl/modeling_qwen3_vl.py:912`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Unsupported Tensor.item() call with capture_scalar_outputs=False
  Explanation: Dynamo does not support tracing `Tensor.item()` with config.capture_scalar_outputs=False.
  Hint: Set `torch._dynamo.config.capture_scalar_outputs = True` or `export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` to include these operations in the captured graph.

  Developer debug context: call_method TensorVariable() item () {}

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

Stack variable source attribution:
  ConstantVariable(int: 1) originated from:
  File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py", line 911
                grid_thw[0].item() // temp_merge_size,
                ~~~~~~~~~~~~~~~~~~~^^~~~~~~~~~~~~~~~~
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1364 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1229 in forward
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1113 in compute_3d_position_ids
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1004 in get_rope_index
transformers/models/qwen3_vl/modeling_qwen3_vl.py:912 in get_vision_position_ids
```
</details>

### 15. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/models/qwen3_vl/modeling_qwen3_vl.py:913`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Unsupported Tensor.item() call with capture_scalar_outputs=False
  Explanation: Dynamo does not support tracing `Tensor.item()` with config.capture_scalar_outputs=False.
  Hint: Set `torch._dynamo.config.capture_scalar_outputs = True` or `export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` to include these operations in the captured graph.

  Developer debug context: call_method TensorVariable() item () {}

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

Stack variable source attribution:
  LazyVariableTracker(unrealized: <class 'int'>) originated from:
  File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py", line 912
                grid_thw[1].item() // spatial_merge_size,
  ConstantVariable(int: 4) originated from:
  File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py", line 912
                grid_thw[1].item() // spatial_merge_size,
                ~~~~~~~~~~~~~~~~~~~^^~~~~~~~~~~~~~~~~~~~
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1364 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1229 in forward
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1113 in compute_3d_position_ids
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1004 in get_rope_index
transformers/models/qwen3_vl/modeling_qwen3_vl.py:913 in get_vision_position_ids
```
</details>

### 16. Dynamic shape operator

- **Location:** `transformers/models/qwen3_vl/modeling_qwen3_vl.py:853`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_FUNCTION_EX: a variadic function call, e.g. f(*args):

Dynamic shape operator
  Explanation: Operator `aten.nonzero.default`'s output shape depends on input Tensor data.
  Hint: Enable tracing of dynamic shape operators with `torch._dynamo.config.capture_dynamic_output_shape_ops = True`

  Developer debug context: aten.nonzero.default

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1364 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1239 in forward
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:834 in forward
transformers/models/qwen3_vl/modeling_qwen3_vl.py:853 in _deepstack_process
```
</details>

### 17. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/qwen3_vl/modeling_qwen3_vl.py:853`  (hit ×1)

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

  Developer debug context: frame skipped: forward (/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py line 759)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  LocalGeneratorObjectVariable(enumerate) originated from:
  File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py", line 821
            for layer_idx, decoder_layer in enumerate(self.layers):
                                            ~~~~~~~~~^^^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1364 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1239 in forward
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/qwen3_vl/modeling_qwen3_vl.py:834 in forward
transformers/models/qwen3_vl/modeling_qwen3_vl.py:853 in _deepstack_process
```
</details>
