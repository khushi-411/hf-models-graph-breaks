# smolvlm — graph breaks

- **HF id:** `HuggingFaceTB/SmolVLM-256M-Instruct`
- **Class:** `AutoModelForImageTextToText`
- **Config overrides:** `{'text_config.num_hidden_layers': 2, 'vision_config.num_hidden_layers': 2}`
- **torch:** `2.16.0a0+git9933f27`  |  **transformers:** `5.22.1`
- **Notes:** Idefics3 architecture.

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ok | 13 | 12 | 4 |
| eval_with_cache | ok | 13 | 12 | 4 |
| train_forward | ok | 13 | 12 | 4 |

## eval_forward

### 1. Dynamic shape operator

- **Location:** `transformers/models/idefics3/modeling_idefics3.py:585`  (hit ×1)

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
transformers/utils/deprecation.py:171 in wrapped_func
transformers/models/idefics3/modeling_idefics3.py:841 in forward
transformers/utils/generic.py:943 in wrapper
transformers/utils/deprecation.py:171 in wrapped_func
transformers/models/idefics3/modeling_idefics3.py:672 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/idefics3/modeling_idefics3.py:585 in get_image_features
```
</details>

### 2. Dynamic shape operator

- **Location:** `transformers/models/idefics3/modeling_idefics3.py:166`  (hit ×1)

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
transformers/utils/deprecation.py:171 in wrapped_func
transformers/models/idefics3/modeling_idefics3.py:841 in forward
transformers/utils/generic.py:943 in wrapper
transformers/utils/deprecation.py:171 in wrapped_func
transformers/models/idefics3/modeling_idefics3.py:672 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/idefics3/modeling_idefics3.py:605 in get_image_features
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/idefics3/modeling_idefics3.py:166 in forward
```
</details>

### 3. Data dependent operator

- **Location:** `transformers/models/idefics3/modeling_idefics3.py:613`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Data dependent operator
  Explanation: Operator `aten._local_scalar_dense.default` has a non-Tensor output whose value is dependent on the data of Tensor inputs.
  Hint: Enable tracing of data-dependent output operators with `torch._dynamo.config.capture_scalar_outputs = True`

  Developer debug context: aten._local_scalar_dense.default

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html

Stack variable source attribution:
  GetAttrVariable(TensorVariable(), split) originated from:
  File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/idefics3/modeling_idefics3.py", line 613
                f.reshape(-1, f.shape[-1]) for f in image_features.split(num_real_images_per_batch.tolist())
                                                    ^^^^^^^^^^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/utils/deprecation.py:171 in wrapped_func
transformers/models/idefics3/modeling_idefics3.py:841 in forward
transformers/utils/generic.py:943 in wrapper
transformers/utils/deprecation.py:171 in wrapped_func
transformers/models/idefics3/modeling_idefics3.py:672 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/idefics3/modeling_idefics3.py:613 in get_image_features
```
</details>

### 4. Unsupported Tensor.item() call with capture_scalar_outputs=False

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
transformers/utils/deprecation.py:171 in wrapped_func
transformers/models/idefics3/modeling_idefics3.py:841 in forward
transformers/utils/generic.py:943 in wrapper
transformers/utils/deprecation.py:171 in wrapped_func
transformers/models/idefics3/modeling_idefics3.py:691 in forward
transformers/models/idefics3/modeling_idefics3.py:556 in inputs_merger
transformers/utils/import_utils.py:1897 in torch_compilable_check
```
</details>


## eval_with_cache

### 1. Dynamic shape operator

- **Location:** `transformers/models/idefics3/modeling_idefics3.py:585`  (hit ×1)

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
transformers/utils/deprecation.py:171 in wrapped_func
transformers/models/idefics3/modeling_idefics3.py:841 in forward
transformers/utils/generic.py:943 in wrapper
transformers/utils/deprecation.py:171 in wrapped_func
transformers/models/idefics3/modeling_idefics3.py:672 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/idefics3/modeling_idefics3.py:585 in get_image_features
```
</details>

### 2. Dynamic shape operator

- **Location:** `transformers/models/idefics3/modeling_idefics3.py:166`  (hit ×1)

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
transformers/utils/deprecation.py:171 in wrapped_func
transformers/models/idefics3/modeling_idefics3.py:841 in forward
transformers/utils/generic.py:943 in wrapper
transformers/utils/deprecation.py:171 in wrapped_func
transformers/models/idefics3/modeling_idefics3.py:672 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/idefics3/modeling_idefics3.py:605 in get_image_features
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/idefics3/modeling_idefics3.py:166 in forward
```
</details>

### 3. Data dependent operator

- **Location:** `transformers/models/idefics3/modeling_idefics3.py:613`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Data dependent operator
  Explanation: Operator `aten._local_scalar_dense.default` has a non-Tensor output whose value is dependent on the data of Tensor inputs.
  Hint: Enable tracing of data-dependent output operators with `torch._dynamo.config.capture_scalar_outputs = True`

  Developer debug context: aten._local_scalar_dense.default

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html

Stack variable source attribution:
  GetAttrVariable(TensorVariable(), split) originated from:
  File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/idefics3/modeling_idefics3.py", line 613
                f.reshape(-1, f.shape[-1]) for f in image_features.split(num_real_images_per_batch.tolist())
                                                    ^^^^^^^^^^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/utils/deprecation.py:171 in wrapped_func
transformers/models/idefics3/modeling_idefics3.py:841 in forward
transformers/utils/generic.py:943 in wrapper
transformers/utils/deprecation.py:171 in wrapped_func
transformers/models/idefics3/modeling_idefics3.py:672 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/idefics3/modeling_idefics3.py:613 in get_image_features
```
</details>

### 4. Unsupported Tensor.item() call with capture_scalar_outputs=False

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
transformers/utils/deprecation.py:171 in wrapped_func
transformers/models/idefics3/modeling_idefics3.py:841 in forward
transformers/utils/generic.py:943 in wrapper
transformers/utils/deprecation.py:171 in wrapped_func
transformers/models/idefics3/modeling_idefics3.py:691 in forward
transformers/models/idefics3/modeling_idefics3.py:556 in inputs_merger
transformers/utils/import_utils.py:1897 in torch_compilable_check
```
</details>


## train_forward

### 1. Dynamic shape operator

- **Location:** `transformers/models/idefics3/modeling_idefics3.py:585`  (hit ×1)

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
transformers/utils/deprecation.py:171 in wrapped_func
transformers/models/idefics3/modeling_idefics3.py:841 in forward
transformers/utils/generic.py:943 in wrapper
transformers/utils/deprecation.py:171 in wrapped_func
transformers/models/idefics3/modeling_idefics3.py:672 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/idefics3/modeling_idefics3.py:585 in get_image_features
```
</details>

### 2. Dynamic shape operator

- **Location:** `transformers/models/idefics3/modeling_idefics3.py:166`  (hit ×1)

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
transformers/utils/deprecation.py:171 in wrapped_func
transformers/models/idefics3/modeling_idefics3.py:841 in forward
transformers/utils/generic.py:943 in wrapper
transformers/utils/deprecation.py:171 in wrapped_func
transformers/models/idefics3/modeling_idefics3.py:672 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/idefics3/modeling_idefics3.py:605 in get_image_features
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/idefics3/modeling_idefics3.py:166 in forward
```
</details>

### 3. Data dependent operator

- **Location:** `transformers/models/idefics3/modeling_idefics3.py:613`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Data dependent operator
  Explanation: Operator `aten._local_scalar_dense.default` has a non-Tensor output whose value is dependent on the data of Tensor inputs.
  Hint: Enable tracing of data-dependent output operators with `torch._dynamo.config.capture_scalar_outputs = True`

  Developer debug context: aten._local_scalar_dense.default

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html

Stack variable source attribution:
  GetAttrVariable(TensorVariable(), split) originated from:
  File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/idefics3/modeling_idefics3.py", line 613
                f.reshape(-1, f.shape[-1]) for f in image_features.split(num_real_images_per_batch.tolist())
                                                    ^^^^^^^^^^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/utils/deprecation.py:171 in wrapped_func
transformers/models/idefics3/modeling_idefics3.py:841 in forward
transformers/utils/generic.py:943 in wrapper
transformers/utils/deprecation.py:171 in wrapped_func
transformers/models/idefics3/modeling_idefics3.py:672 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/idefics3/modeling_idefics3.py:613 in get_image_features
```
</details>

### 4. Unsupported Tensor.item() call with capture_scalar_outputs=False

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
transformers/utils/deprecation.py:171 in wrapped_func
transformers/models/idefics3/modeling_idefics3.py:841 in forward
transformers/utils/generic.py:943 in wrapper
transformers/utils/deprecation.py:171 in wrapped_func
transformers/models/idefics3/modeling_idefics3.py:691 in forward
transformers/models/idefics3/modeling_idefics3.py:556 in inputs_merger
transformers/utils/import_utils.py:1897 in torch_compilable_check
```
</details>
