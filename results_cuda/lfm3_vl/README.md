# lfm3_vl — graph breaks

- **HF id:** `LiquidAI/LFM3-VL-2B`
- **Class:** `AutoModelForImageTextToText`
- **Config overrides:** `{'text_config.num_hidden_layers': 3, 'text_config.layer_types': ['conv', 'full_attention', 'sliding_attention'], 'text_config.sliding_window': 16, 'text_config.hidden_size': 512, 'text_config.intermediate_size': 1024, 'text_config.num_attention_heads': 8, 'text_config.num_key_value_heads': 2, 'text_config.head_dim': 64, 'vision_config.num_hidden_layers': 2, 'dtype': 'float32', 'text_config.dtype': 'float32', 'vision_config.dtype': 'float32'}`
- **torch:** `2.16.0a0+git57f02f8`  |  **transformers:** `5.24.0.dev0` (git `b7fe69cb19`)
- **Notes:** Hybrid conv / full / sliding attention LM with a Qwen2-VL-style vision tower.

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ok | 13 | 12 | 5 |
| eval_with_cache | ok | 13 | 12 | 5 |
| train_forward | ok | 13 | 12 | 5 |

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
transformers/models/lfm3_vl/modeling_lfm3_vl.py:771 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/lfm3_vl/modeling_lfm3_vl.py:634 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/lfm3_vl/modeling_lfm3_vl.py:556 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/lfm3_vl/modeling_lfm3_vl.py:437 in forward
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
transformers/models/lfm3_vl/modeling_lfm3_vl.py:771 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/lfm3_vl/modeling_lfm3_vl.py:634 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/lfm3_vl/modeling_lfm3_vl.py:556 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/lfm3_vl/modeling_lfm3_vl.py:438 in forward
transformers/vision_utils.py:76 in get_vision_attention_seqlens
transformers/vision_utils.py:64 in get_vision_cu_seqlens
```
</details>

### 3. Data dependent operator

- **Location:** `transformers/models/lfm3_vl/modeling_lfm3_vl.py:440`  (hit ×1)

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
transformers/models/lfm3_vl/modeling_lfm3_vl.py:771 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/lfm3_vl/modeling_lfm3_vl.py:634 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/lfm3_vl/modeling_lfm3_vl.py:556 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/lfm3_vl/modeling_lfm3_vl.py:440 in forward
```
</details>

### 4. Data dependent operator

- **Location:** `transformers/models/lfm3_vl/modeling_lfm3_vl.py:558`  (hit ×1)

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
transformers/models/lfm3_vl/modeling_lfm3_vl.py:771 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/lfm3_vl/modeling_lfm3_vl.py:634 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/lfm3_vl/modeling_lfm3_vl.py:558 in get_image_features
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
transformers/models/lfm3_vl/modeling_lfm3_vl.py:771 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/lfm3_vl/modeling_lfm3_vl.py:646 in forward
transformers/models/lfm3_vl/modeling_lfm3_vl.py:589 in get_placeholder_mask
transformers/utils/import_utils.py:1975 in torch_compilable_check
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
transformers/models/lfm3_vl/modeling_lfm3_vl.py:771 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/lfm3_vl/modeling_lfm3_vl.py:634 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/lfm3_vl/modeling_lfm3_vl.py:556 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/lfm3_vl/modeling_lfm3_vl.py:437 in forward
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
transformers/models/lfm3_vl/modeling_lfm3_vl.py:771 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/lfm3_vl/modeling_lfm3_vl.py:634 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/lfm3_vl/modeling_lfm3_vl.py:556 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/lfm3_vl/modeling_lfm3_vl.py:438 in forward
transformers/vision_utils.py:76 in get_vision_attention_seqlens
transformers/vision_utils.py:64 in get_vision_cu_seqlens
```
</details>

### 3. Data dependent operator

- **Location:** `transformers/models/lfm3_vl/modeling_lfm3_vl.py:440`  (hit ×1)

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
transformers/models/lfm3_vl/modeling_lfm3_vl.py:771 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/lfm3_vl/modeling_lfm3_vl.py:634 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/lfm3_vl/modeling_lfm3_vl.py:556 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/lfm3_vl/modeling_lfm3_vl.py:440 in forward
```
</details>

### 4. Data dependent operator

- **Location:** `transformers/models/lfm3_vl/modeling_lfm3_vl.py:558`  (hit ×1)

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
transformers/models/lfm3_vl/modeling_lfm3_vl.py:771 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/lfm3_vl/modeling_lfm3_vl.py:634 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/lfm3_vl/modeling_lfm3_vl.py:558 in get_image_features
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
transformers/models/lfm3_vl/modeling_lfm3_vl.py:771 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/lfm3_vl/modeling_lfm3_vl.py:646 in forward
transformers/models/lfm3_vl/modeling_lfm3_vl.py:589 in get_placeholder_mask
transformers/utils/import_utils.py:1975 in torch_compilable_check
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
transformers/models/lfm3_vl/modeling_lfm3_vl.py:771 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/lfm3_vl/modeling_lfm3_vl.py:634 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/lfm3_vl/modeling_lfm3_vl.py:556 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/lfm3_vl/modeling_lfm3_vl.py:437 in forward
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
transformers/models/lfm3_vl/modeling_lfm3_vl.py:771 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/lfm3_vl/modeling_lfm3_vl.py:634 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/lfm3_vl/modeling_lfm3_vl.py:556 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/lfm3_vl/modeling_lfm3_vl.py:438 in forward
transformers/vision_utils.py:76 in get_vision_attention_seqlens
transformers/vision_utils.py:64 in get_vision_cu_seqlens
```
</details>

### 3. Data dependent operator

- **Location:** `transformers/models/lfm3_vl/modeling_lfm3_vl.py:440`  (hit ×1)

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
transformers/models/lfm3_vl/modeling_lfm3_vl.py:771 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/lfm3_vl/modeling_lfm3_vl.py:634 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/lfm3_vl/modeling_lfm3_vl.py:556 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/lfm3_vl/modeling_lfm3_vl.py:440 in forward
```
</details>

### 4. Data dependent operator

- **Location:** `transformers/models/lfm3_vl/modeling_lfm3_vl.py:558`  (hit ×1)

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
transformers/models/lfm3_vl/modeling_lfm3_vl.py:771 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/lfm3_vl/modeling_lfm3_vl.py:634 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/lfm3_vl/modeling_lfm3_vl.py:558 in get_image_features
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
transformers/models/lfm3_vl/modeling_lfm3_vl.py:771 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/lfm3_vl/modeling_lfm3_vl.py:646 in forward
transformers/models/lfm3_vl/modeling_lfm3_vl.py:589 in get_placeholder_mask
transformers/utils/import_utils.py:1975 in torch_compilable_check
```
</details>
