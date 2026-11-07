# mistral3 — graph breaks

- **HF id:** `unsloth/Mistral-Small-3.1-24B-Instruct-2503`
- **Class:** `AutoModelForImageTextToText`
- **Config overrides:** `{'text_config.num_hidden_layers': 2, 'text_config.hidden_size': 512, 'text_config.intermediate_size': 1024, 'text_config.num_attention_heads': 8, 'text_config.num_key_value_heads': 2, 'vision_config.num_hidden_layers': 2, 'vision_config.image_size': 224}`
- **torch:** `2.16.0a0+git9933f27`  |  **transformers:** `5.22.1`
- **Notes:** Pixtral vision tower.

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ok | 14 | 13 | 4 |
| eval_with_cache | ok | 14 | 13 | 4 |
| train_forward | ok | 14 | 13 | 4 |

## eval_forward

### 1. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/pixtral/modeling_pixtral.py:419`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

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
transformers/models/mistral3/modeling_mistral3.py:408 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/mistral3/modeling_mistral3.py:297 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/mistral3/modeling_mistral3.py:223 in get_image_features
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/pixtral/modeling_pixtral.py:419 in forward
```
</details>

### 2. Data dependent operator

- **Location:** `transformers/models/mistral3/modeling_mistral3.py:84`  (hit ×1)

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
transformers/models/mistral3/modeling_mistral3.py:408 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/mistral3/modeling_mistral3.py:297 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/mistral3/modeling_mistral3.py:237 in get_image_features
transformers/models/mistral3/modeling_mistral3.py:84 in forward
```
</details>

### 3. Data dependent operator

- **Location:** `transformers/models/mistral3/modeling_mistral3.py:240`  (hit ×1)

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
transformers/models/mistral3/modeling_mistral3.py:408 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/mistral3/modeling_mistral3.py:297 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/mistral3/modeling_mistral3.py:240 in get_image_features
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
transformers/models/mistral3/modeling_mistral3.py:408 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/mistral3/modeling_mistral3.py:308 in forward
transformers/models/mistral3/modeling_mistral3.py:263 in get_placeholder_mask
transformers/utils/import_utils.py:1897 in torch_compilable_check
```
</details>


## eval_with_cache

### 1. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/pixtral/modeling_pixtral.py:419`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

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
transformers/models/mistral3/modeling_mistral3.py:408 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/mistral3/modeling_mistral3.py:297 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/mistral3/modeling_mistral3.py:223 in get_image_features
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/pixtral/modeling_pixtral.py:419 in forward
```
</details>

### 2. Data dependent operator

- **Location:** `transformers/models/mistral3/modeling_mistral3.py:84`  (hit ×1)

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
transformers/models/mistral3/modeling_mistral3.py:408 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/mistral3/modeling_mistral3.py:297 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/mistral3/modeling_mistral3.py:237 in get_image_features
transformers/models/mistral3/modeling_mistral3.py:84 in forward
```
</details>

### 3. Data dependent operator

- **Location:** `transformers/models/mistral3/modeling_mistral3.py:240`  (hit ×1)

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
transformers/models/mistral3/modeling_mistral3.py:408 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/mistral3/modeling_mistral3.py:297 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/mistral3/modeling_mistral3.py:240 in get_image_features
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
transformers/models/mistral3/modeling_mistral3.py:408 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/mistral3/modeling_mistral3.py:308 in forward
transformers/models/mistral3/modeling_mistral3.py:263 in get_placeholder_mask
transformers/utils/import_utils.py:1897 in torch_compilable_check
```
</details>


## train_forward

### 1. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/pixtral/modeling_pixtral.py:419`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

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
transformers/models/mistral3/modeling_mistral3.py:408 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/mistral3/modeling_mistral3.py:297 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/mistral3/modeling_mistral3.py:223 in get_image_features
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/pixtral/modeling_pixtral.py:419 in forward
```
</details>

### 2. Data dependent operator

- **Location:** `transformers/models/mistral3/modeling_mistral3.py:84`  (hit ×1)

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
transformers/models/mistral3/modeling_mistral3.py:408 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/mistral3/modeling_mistral3.py:297 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/mistral3/modeling_mistral3.py:237 in get_image_features
transformers/models/mistral3/modeling_mistral3.py:84 in forward
```
</details>

### 3. Data dependent operator

- **Location:** `transformers/models/mistral3/modeling_mistral3.py:240`  (hit ×1)

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
transformers/models/mistral3/modeling_mistral3.py:408 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/mistral3/modeling_mistral3.py:297 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/mistral3/modeling_mistral3.py:240 in get_image_features
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
transformers/models/mistral3/modeling_mistral3.py:408 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/mistral3/modeling_mistral3.py:308 in forward
transformers/models/mistral3/modeling_mistral3.py:263 in get_placeholder_mask
transformers/utils/import_utils.py:1897 in torch_compilable_check
```
</details>
