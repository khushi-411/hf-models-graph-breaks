# llava_next — graph breaks

- **HF id:** `llava-hf/llava-v1.6-mistral-7b-hf`
- **Class:** `AutoModelForImageTextToText`
- **Config overrides:** `{'text_config.num_hidden_layers': 2, 'text_config.hidden_size': 1024, 'text_config.intermediate_size': 2048, 'text_config.num_attention_heads': 16, 'text_config.num_key_value_heads': 8, 'text_config.head_dim': 64, 'vision_config.num_hidden_layers': 2}`
- **torch:** `2.16.0a0+git57f02f8`  |  **transformers:** `5.24.0.dev0` (git `b7fe69cb19`)
- **Notes:** LLaVA-NeXT (anyres image tiling).

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ok | 7 | 6 | 5 |
| eval_with_cache | ok | 7 | 6 | 5 |
| train_forward | ok | 7 | 6 | 5 |

## eval_forward

### 1. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/llava_next/modeling_llava_next.py:97`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

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
transformers/models/llava_next/modeling_llava_next.py:610 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/llava_next/modeling_llava_next.py:466 in forward
transformers/utils/output_capturing.py:303 in wrapper
transformers/models/llava_next/modeling_llava_next.py:362 in get_image_features
transformers/models/llava_next/modeling_llava_next.py:97 in image_size_to_num_patches
```
</details>

### 2. Data dependent operator

- **Location:** `transformers/models/llava_next/modeling_llava_next.py:68`  (hit ×1)

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
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/llava_next/modeling_llava_next.py:610 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/llava_next/modeling_llava_next.py:466 in forward
transformers/utils/output_capturing.py:303 in wrapper
transformers/models/llava_next/modeling_llava_next.py:397 in get_image_features
transformers/models/llava_next/modeling_llava_next.py:292 in pack_image_features
transformers/models/llava_next/modeling_llava_next.py:68 in get_anyres_image_grid_shape
```
</details>

### 3. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/llava_next/modeling_llava_next.py:68`  (hit ×1)

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

  Developer debug context: frame skipped: pack_image_features (/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/llava_next/modeling_llava_next.py line 266)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  LocalGeneratorObjectVariable(enumerate) originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/llava_next/modeling_llava_next.py", line 286
            for image_idx, image_feature in enumerate(image_features):
                                            ~~~~~~~~~^^^^^^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/llava_next/modeling_llava_next.py:610 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/llava_next/modeling_llava_next.py:466 in forward
transformers/utils/output_capturing.py:303 in wrapper
transformers/models/llava_next/modeling_llava_next.py:397 in get_image_features
transformers/models/llava_next/modeling_llava_next.py:292 in pack_image_features
transformers/models/llava_next/modeling_llava_next.py:68 in get_anyres_image_grid_shape
```
</details>

### 4. Data dependent operator

- **Location:** `transformers/models/llava_next/modeling_llava_next.py:129`  (hit ×1)

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
transformers/models/llava_next/modeling_llava_next.py:610 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/llava_next/modeling_llava_next.py:466 in forward
transformers/utils/output_capturing.py:303 in wrapper
transformers/models/llava_next/modeling_llava_next.py:397 in get_image_features
transformers/models/llava_next/modeling_llava_next.py:311 in pack_image_features
transformers/models/llava_next/modeling_llava_next.py:129 in unpad_image
```
</details>

### 5. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/utils/import_utils.py:1975`  (hit ×1)

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
transformers/models/llava_next/modeling_llava_next.py:610 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/llava_next/modeling_llava_next.py:478 in forward
transformers/models/llava_next/modeling_llava_next.py:424 in get_placeholder_mask
transformers/utils/import_utils.py:1975 in torch_compilable_check
```
</details>


## eval_with_cache

### 1. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/llava_next/modeling_llava_next.py:97`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

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
transformers/models/llava_next/modeling_llava_next.py:610 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/llava_next/modeling_llava_next.py:466 in forward
transformers/utils/output_capturing.py:303 in wrapper
transformers/models/llava_next/modeling_llava_next.py:362 in get_image_features
transformers/models/llava_next/modeling_llava_next.py:97 in image_size_to_num_patches
```
</details>

### 2. Data dependent operator

- **Location:** `transformers/models/llava_next/modeling_llava_next.py:68`  (hit ×1)

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
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/llava_next/modeling_llava_next.py:610 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/llava_next/modeling_llava_next.py:466 in forward
transformers/utils/output_capturing.py:303 in wrapper
transformers/models/llava_next/modeling_llava_next.py:397 in get_image_features
transformers/models/llava_next/modeling_llava_next.py:292 in pack_image_features
transformers/models/llava_next/modeling_llava_next.py:68 in get_anyres_image_grid_shape
```
</details>

### 3. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/llava_next/modeling_llava_next.py:68`  (hit ×1)

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

  Developer debug context: frame skipped: pack_image_features (/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/llava_next/modeling_llava_next.py line 266)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  LocalGeneratorObjectVariable(enumerate) originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/llava_next/modeling_llava_next.py", line 286
            for image_idx, image_feature in enumerate(image_features):
                                            ~~~~~~~~~^^^^^^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/llava_next/modeling_llava_next.py:610 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/llava_next/modeling_llava_next.py:466 in forward
transformers/utils/output_capturing.py:303 in wrapper
transformers/models/llava_next/modeling_llava_next.py:397 in get_image_features
transformers/models/llava_next/modeling_llava_next.py:292 in pack_image_features
transformers/models/llava_next/modeling_llava_next.py:68 in get_anyres_image_grid_shape
```
</details>

### 4. Data dependent operator

- **Location:** `transformers/models/llava_next/modeling_llava_next.py:129`  (hit ×1)

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
transformers/models/llava_next/modeling_llava_next.py:610 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/llava_next/modeling_llava_next.py:466 in forward
transformers/utils/output_capturing.py:303 in wrapper
transformers/models/llava_next/modeling_llava_next.py:397 in get_image_features
transformers/models/llava_next/modeling_llava_next.py:311 in pack_image_features
transformers/models/llava_next/modeling_llava_next.py:129 in unpad_image
```
</details>

### 5. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/utils/import_utils.py:1975`  (hit ×1)

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
transformers/models/llava_next/modeling_llava_next.py:610 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/llava_next/modeling_llava_next.py:478 in forward
transformers/models/llava_next/modeling_llava_next.py:424 in get_placeholder_mask
transformers/utils/import_utils.py:1975 in torch_compilable_check
```
</details>


## train_forward

### 1. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/llava_next/modeling_llava_next.py:97`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

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
transformers/models/llava_next/modeling_llava_next.py:610 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/llava_next/modeling_llava_next.py:466 in forward
transformers/utils/output_capturing.py:303 in wrapper
transformers/models/llava_next/modeling_llava_next.py:362 in get_image_features
transformers/models/llava_next/modeling_llava_next.py:97 in image_size_to_num_patches
```
</details>

### 2. Data dependent operator

- **Location:** `transformers/models/llava_next/modeling_llava_next.py:68`  (hit ×1)

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
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/llava_next/modeling_llava_next.py:610 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/llava_next/modeling_llava_next.py:466 in forward
transformers/utils/output_capturing.py:303 in wrapper
transformers/models/llava_next/modeling_llava_next.py:397 in get_image_features
transformers/models/llava_next/modeling_llava_next.py:292 in pack_image_features
transformers/models/llava_next/modeling_llava_next.py:68 in get_anyres_image_grid_shape
```
</details>

### 3. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/llava_next/modeling_llava_next.py:68`  (hit ×1)

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

  Developer debug context: frame skipped: pack_image_features (/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/llava_next/modeling_llava_next.py line 266)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  LocalGeneratorObjectVariable(enumerate) originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/llava_next/modeling_llava_next.py", line 286
            for image_idx, image_feature in enumerate(image_features):
                                            ~~~~~~~~~^^^^^^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/llava_next/modeling_llava_next.py:610 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/llava_next/modeling_llava_next.py:466 in forward
transformers/utils/output_capturing.py:303 in wrapper
transformers/models/llava_next/modeling_llava_next.py:397 in get_image_features
transformers/models/llava_next/modeling_llava_next.py:292 in pack_image_features
transformers/models/llava_next/modeling_llava_next.py:68 in get_anyres_image_grid_shape
```
</details>

### 4. Data dependent operator

- **Location:** `transformers/models/llava_next/modeling_llava_next.py:129`  (hit ×1)

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
transformers/models/llava_next/modeling_llava_next.py:610 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/llava_next/modeling_llava_next.py:466 in forward
transformers/utils/output_capturing.py:303 in wrapper
transformers/models/llava_next/modeling_llava_next.py:397 in get_image_features
transformers/models/llava_next/modeling_llava_next.py:311 in pack_image_features
transformers/models/llava_next/modeling_llava_next.py:129 in unpad_image
```
</details>

### 5. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/utils/import_utils.py:1975`  (hit ×1)

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
transformers/models/llava_next/modeling_llava_next.py:610 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/llava_next/modeling_llava_next.py:478 in forward
transformers/models/llava_next/modeling_llava_next.py:424 in get_placeholder_mask
transformers/utils/import_utils.py:1975 in torch_compilable_check
```
</details>
