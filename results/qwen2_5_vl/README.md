# qwen2_5_vl — graph breaks

- **HF id:** `Qwen/Qwen2.5-VL-3B-Instruct`
- **Class:** `AutoModelForImageTextToText`
- **Config overrides:** `{'text_config.num_hidden_layers': 2, 'text_config.hidden_size': 512, 'text_config.intermediate_size': 1024, 'text_config.num_attention_heads': 4, 'text_config.num_key_value_heads': 2, 'vision_config.depth': 2, 'vision_config.fullatt_block_indexes': [1], 'vision_config.out_hidden_size': 512}`
- **torch:** `2.16.0a0+git9933f27`  |  **transformers:** `5.22.1`

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ok | 25 | 24 | 13 |
| eval_with_cache | ok | 25 | 24 | 13 |
| train_forward | ok | 25 | 24 | 13 |

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
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1382 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1211 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1083 in get_image_features
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:440 in forward
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
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1382 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1211 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1083 in get_image_features
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:441 in forward
transformers/vision_utils.py:76 in get_vision_attention_seqlens
transformers/vision_utils.py:64 in get_vision_cu_seqlens
```
</details>

### 3. Data dependent operator

- **Location:** `transformers/vision_utils.py:161`  (hit ×1)

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
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1382 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1211 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1083 in get_image_features
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:442 in forward
transformers/vision_utils.py:161 in get_vision_window_index
```
</details>

### 4. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/vision_utils.py:179`  (hit ×1)

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

  Developer debug context: frame skipped: torch_dynamo_resume_in_get_vision_window_index_at_161 (/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/vision_utils.py line 161)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  ListIteratorVariable(length=2, index=1) originated from:
  File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/vision_utils.py", line 161
        for grid_t, grid_h, grid_w in grid_thw.tolist():
                                      ~~~~~~~~~~~~~~~^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1382 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1211 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1083 in get_image_features
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:442 in forward
transformers/vision_utils.py:179 in get_vision_window_index
```
</details>

### 5. Attempted to read undefined local variable

- **Location:** `transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:459`  (hit ×1)

```
Encountered graph break that we cannot resume from. Compiling up to the previous resumable state, then skipping the rest of the function. Graph break encountered:

Attempted to read undefined local variable
  Explanation: Could not find a local variable with name `self`
  Hint: Your code may result in an error when running in eager. Please double check that your code doesn't contain a similar error when actually running eager/uncompiled. You can do this by removing the `torch.compile` call, or by using `torch.compiler.set_stance("force_eager")`. 

  Developer debug context: LOAD_FAST self

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0011.html

Stack variable source attribution:
  ListVariable(length=0) originated from:
  File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py", line 459
            position_embeddings = tuple(
                                  ^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0011.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1382 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1211 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1083 in get_image_features
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:459 in forward
```
</details>

### 6. Data dependent operator

- **Location:** `transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:314`  (hit ×1)

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
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1382 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1211 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1083 in get_image_features
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:471 in forward
transformers/modeling_layers.py:111 in __call__
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:314 in forward
```
</details>

### 7. Data dependent operator

- **Location:** `transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1084`  (hit ×1)

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
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1382 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1211 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1084 in get_image_features
```
</details>

### 8. Unsupported Tensor.item() call with capture_scalar_outputs=False

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
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1382 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1224 in forward
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1116 in get_placeholder_mask
transformers/utils/import_utils.py:1897 in torch_compilable_check
```
</details>

### 9. Dynamic shape operator

- **Location:** `transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1014`  (hit ×1)

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
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1382 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1239 in forward
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1149 in compute_3d_position_ids
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1014 in get_rope_index
```
</details>

### 10. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1014`  (hit ×1)

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

  Developer debug context: frame skipped: get_rope_index (/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py line 941)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  LocalGeneratorObjectVariable(enumerate) originated from:
  File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py", line 1011
            for batch_idx, current_input_ids in enumerate(input_ids):
                                                ~~~~~~~~~^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1382 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1239 in forward
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1149 in compute_3d_position_ids
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1014 in get_rope_index
```
</details>

### 11. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:927`  (hit ×1)

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
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1382 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1239 in forward
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1149 in compute_3d_position_ids
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1043 in get_rope_index
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:927 in get_vision_position_ids
```
</details>

### 12. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:928`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Unsupported Tensor.item() call with capture_scalar_outputs=False
  Explanation: Dynamo does not support tracing `Tensor.item()` with config.capture_scalar_outputs=False.
  Hint: Set `torch._dynamo.config.capture_scalar_outputs = True` or `export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` to include these operations in the captured graph.

  Developer debug context: call_method TensorVariable() item () {}

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

Stack variable source attribution:
  ConstantVariable(int: 1) originated from:
  File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py", line 927
                grid_thw[0].item() // temp_merge_size,
                ~~~~~~~~~~~~~~~~~~~^^~~~~~~~~~~~~~~~~
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1382 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1239 in forward
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1149 in compute_3d_position_ids
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1043 in get_rope_index
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:928 in get_vision_position_ids
```
</details>

### 13. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:929`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Unsupported Tensor.item() call with capture_scalar_outputs=False
  Explanation: Dynamo does not support tracing `Tensor.item()` with config.capture_scalar_outputs=False.
  Hint: Set `torch._dynamo.config.capture_scalar_outputs = True` or `export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` to include these operations in the captured graph.

  Developer debug context: call_method TensorVariable() item () {}

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

Stack variable source attribution:
  LazyVariableTracker(unrealized: <class 'int'>) originated from:
  File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py", line 928
                grid_thw[1].item() // spatial_merge_size,
  ConstantVariable(int: 4) originated from:
  File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py", line 928
                grid_thw[1].item() // spatial_merge_size,
                ~~~~~~~~~~~~~~~~~~~^^~~~~~~~~~~~~~~~~~~~
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1382 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1239 in forward
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1149 in compute_3d_position_ids
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1043 in get_rope_index
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:929 in get_vision_position_ids
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
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1382 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1211 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1083 in get_image_features
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:440 in forward
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
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1382 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1211 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1083 in get_image_features
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:441 in forward
transformers/vision_utils.py:76 in get_vision_attention_seqlens
transformers/vision_utils.py:64 in get_vision_cu_seqlens
```
</details>

### 3. Data dependent operator

- **Location:** `transformers/vision_utils.py:161`  (hit ×1)

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
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1382 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1211 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1083 in get_image_features
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:442 in forward
transformers/vision_utils.py:161 in get_vision_window_index
```
</details>

### 4. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/vision_utils.py:179`  (hit ×1)

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

  Developer debug context: frame skipped: torch_dynamo_resume_in_get_vision_window_index_at_161 (/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/vision_utils.py line 161)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  ListIteratorVariable(length=2, index=1) originated from:
  File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/vision_utils.py", line 161
        for grid_t, grid_h, grid_w in grid_thw.tolist():
                                      ~~~~~~~~~~~~~~~^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1382 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1211 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1083 in get_image_features
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:442 in forward
transformers/vision_utils.py:179 in get_vision_window_index
```
</details>

### 5. Attempted to read undefined local variable

- **Location:** `transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:459`  (hit ×1)

```
Encountered graph break that we cannot resume from. Compiling up to the previous resumable state, then skipping the rest of the function. Graph break encountered:

Attempted to read undefined local variable
  Explanation: Could not find a local variable with name `self`
  Hint: Your code may result in an error when running in eager. Please double check that your code doesn't contain a similar error when actually running eager/uncompiled. You can do this by removing the `torch.compile` call, or by using `torch.compiler.set_stance("force_eager")`. 

  Developer debug context: LOAD_FAST self

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0011.html

Stack variable source attribution:
  ListVariable(length=0) originated from:
  File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py", line 459
            position_embeddings = tuple(
                                  ^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0011.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1382 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1211 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1083 in get_image_features
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:459 in forward
```
</details>

### 6. Data dependent operator

- **Location:** `transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:314`  (hit ×1)

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
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1382 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1211 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1083 in get_image_features
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:471 in forward
transformers/modeling_layers.py:111 in __call__
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:314 in forward
```
</details>

### 7. Data dependent operator

- **Location:** `transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1084`  (hit ×1)

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
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1382 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1211 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1084 in get_image_features
```
</details>

### 8. Unsupported Tensor.item() call with capture_scalar_outputs=False

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
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1382 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1224 in forward
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1116 in get_placeholder_mask
transformers/utils/import_utils.py:1897 in torch_compilable_check
```
</details>

### 9. Dynamic shape operator

- **Location:** `transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1014`  (hit ×1)

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
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1382 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1239 in forward
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1149 in compute_3d_position_ids
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1014 in get_rope_index
```
</details>

### 10. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1014`  (hit ×1)

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

  Developer debug context: frame skipped: get_rope_index (/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py line 941)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  LocalGeneratorObjectVariable(enumerate) originated from:
  File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py", line 1011
            for batch_idx, current_input_ids in enumerate(input_ids):
                                                ~~~~~~~~~^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1382 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1239 in forward
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1149 in compute_3d_position_ids
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1014 in get_rope_index
```
</details>

### 11. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:927`  (hit ×1)

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
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1382 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1239 in forward
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1149 in compute_3d_position_ids
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1043 in get_rope_index
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:927 in get_vision_position_ids
```
</details>

### 12. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:928`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Unsupported Tensor.item() call with capture_scalar_outputs=False
  Explanation: Dynamo does not support tracing `Tensor.item()` with config.capture_scalar_outputs=False.
  Hint: Set `torch._dynamo.config.capture_scalar_outputs = True` or `export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` to include these operations in the captured graph.

  Developer debug context: call_method TensorVariable() item () {}

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

Stack variable source attribution:
  ConstantVariable(int: 1) originated from:
  File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py", line 927
                grid_thw[0].item() // temp_merge_size,
                ~~~~~~~~~~~~~~~~~~~^^~~~~~~~~~~~~~~~~
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1382 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1239 in forward
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1149 in compute_3d_position_ids
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1043 in get_rope_index
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:928 in get_vision_position_ids
```
</details>

### 13. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:929`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Unsupported Tensor.item() call with capture_scalar_outputs=False
  Explanation: Dynamo does not support tracing `Tensor.item()` with config.capture_scalar_outputs=False.
  Hint: Set `torch._dynamo.config.capture_scalar_outputs = True` or `export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` to include these operations in the captured graph.

  Developer debug context: call_method TensorVariable() item () {}

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

Stack variable source attribution:
  LazyVariableTracker(unrealized: <class 'int'>) originated from:
  File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py", line 928
                grid_thw[1].item() // spatial_merge_size,
  ConstantVariable(int: 4) originated from:
  File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py", line 928
                grid_thw[1].item() // spatial_merge_size,
                ~~~~~~~~~~~~~~~~~~~^^~~~~~~~~~~~~~~~~~~~
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1382 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1239 in forward
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1149 in compute_3d_position_ids
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1043 in get_rope_index
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:929 in get_vision_position_ids
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
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1382 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1211 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1083 in get_image_features
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:440 in forward
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
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1382 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1211 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1083 in get_image_features
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:441 in forward
transformers/vision_utils.py:76 in get_vision_attention_seqlens
transformers/vision_utils.py:64 in get_vision_cu_seqlens
```
</details>

### 3. Data dependent operator

- **Location:** `transformers/vision_utils.py:161`  (hit ×1)

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
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1382 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1211 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1083 in get_image_features
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:442 in forward
transformers/vision_utils.py:161 in get_vision_window_index
```
</details>

### 4. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/vision_utils.py:179`  (hit ×1)

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

  Developer debug context: frame skipped: torch_dynamo_resume_in_get_vision_window_index_at_161 (/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/vision_utils.py line 161)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  ListIteratorVariable(length=2, index=1) originated from:
  File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/vision_utils.py", line 161
        for grid_t, grid_h, grid_w in grid_thw.tolist():
                                      ~~~~~~~~~~~~~~~^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1382 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1211 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1083 in get_image_features
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:442 in forward
transformers/vision_utils.py:179 in get_vision_window_index
```
</details>

### 5. Attempted to read undefined local variable

- **Location:** `transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:459`  (hit ×1)

```
Encountered graph break that we cannot resume from. Compiling up to the previous resumable state, then skipping the rest of the function. Graph break encountered:

Attempted to read undefined local variable
  Explanation: Could not find a local variable with name `self`
  Hint: Your code may result in an error when running in eager. Please double check that your code doesn't contain a similar error when actually running eager/uncompiled. You can do this by removing the `torch.compile` call, or by using `torch.compiler.set_stance("force_eager")`. 

  Developer debug context: LOAD_FAST self

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0011.html

Stack variable source attribution:
  ListVariable(length=0) originated from:
  File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py", line 459
            position_embeddings = tuple(
                                  ^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0011.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1382 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1211 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1083 in get_image_features
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:459 in forward
```
</details>

### 6. Data dependent operator

- **Location:** `transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:314`  (hit ×1)

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
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1382 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1211 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1083 in get_image_features
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:471 in forward
transformers/modeling_layers.py:111 in __call__
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:314 in forward
```
</details>

### 7. Data dependent operator

- **Location:** `transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1084`  (hit ×1)

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
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1382 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1211 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1084 in get_image_features
```
</details>

### 8. Unsupported Tensor.item() call with capture_scalar_outputs=False

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
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1382 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1224 in forward
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1116 in get_placeholder_mask
transformers/utils/import_utils.py:1897 in torch_compilable_check
```
</details>

### 9. Dynamic shape operator

- **Location:** `transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1014`  (hit ×1)

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
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1382 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1239 in forward
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1149 in compute_3d_position_ids
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1014 in get_rope_index
```
</details>

### 10. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1014`  (hit ×1)

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

  Developer debug context: frame skipped: get_rope_index (/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py line 941)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  LocalGeneratorObjectVariable(enumerate) originated from:
  File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py", line 1011
            for batch_idx, current_input_ids in enumerate(input_ids):
                                                ~~~~~~~~~^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1382 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1239 in forward
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1149 in compute_3d_position_ids
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1014 in get_rope_index
```
</details>

### 11. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:927`  (hit ×1)

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
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1382 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1239 in forward
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1149 in compute_3d_position_ids
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1043 in get_rope_index
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:927 in get_vision_position_ids
```
</details>

### 12. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:928`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Unsupported Tensor.item() call with capture_scalar_outputs=False
  Explanation: Dynamo does not support tracing `Tensor.item()` with config.capture_scalar_outputs=False.
  Hint: Set `torch._dynamo.config.capture_scalar_outputs = True` or `export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` to include these operations in the captured graph.

  Developer debug context: call_method TensorVariable() item () {}

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

Stack variable source attribution:
  ConstantVariable(int: 1) originated from:
  File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py", line 927
                grid_thw[0].item() // temp_merge_size,
                ~~~~~~~~~~~~~~~~~~~^^~~~~~~~~~~~~~~~~
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1382 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1239 in forward
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1149 in compute_3d_position_ids
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1043 in get_rope_index
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:928 in get_vision_position_ids
```
</details>

### 13. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:929`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Unsupported Tensor.item() call with capture_scalar_outputs=False
  Explanation: Dynamo does not support tracing `Tensor.item()` with config.capture_scalar_outputs=False.
  Hint: Set `torch._dynamo.config.capture_scalar_outputs = True` or `export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` to include these operations in the captured graph.

  Developer debug context: call_method TensorVariable() item () {}

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

Stack variable source attribution:
  LazyVariableTracker(unrealized: <class 'int'>) originated from:
  File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py", line 928
                grid_thw[1].item() // spatial_merge_size,
  ConstantVariable(int: 4) originated from:
  File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py", line 928
                grid_thw[1].item() // spatial_merge_size,
                ~~~~~~~~~~~~~~~~~~~^^~~~~~~~~~~~~~~~~~~~
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1382 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1239 in forward
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1149 in compute_3d_position_ids
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1043 in get_rope_index
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:929 in get_vision_position_ids
```
</details>
