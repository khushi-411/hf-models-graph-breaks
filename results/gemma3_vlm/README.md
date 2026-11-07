# gemma3_vlm — graph breaks

- **HF id:** `unsloth/gemma-3-4b-it`
- **Class:** `AutoModelForImageTextToText`
- **Config overrides:** `{'text_config.num_hidden_layers': 2, 'text_config.hidden_size': 512, 'text_config.intermediate_size': 1024, 'vision_config.num_hidden_layers': 2, 'vision_config.image_size': 224}`
- **torch:** `2.16.0a0+git9933f27`  |  **transformers:** `5.22.1`

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ok | 4 | 3 | 1 |
| eval_with_cache | ok | 4 | 3 | 1 |
| train_forward | ok | 4 | 3 | 1 |

## eval_forward

### 1. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/utils/import_utils.py:1897`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_FUNCTION_EX: a variadic function call, e.g. f(*args):

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
transformers/models/gemma3/modeling_gemma3.py:997 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/gemma3/modeling_gemma3.py:873 in forward
transformers/models/gemma3/modeling_gemma3.py:803 in get_placeholder_mask
transformers/utils/import_utils.py:1897 in torch_compilable_check
```
</details>


## eval_with_cache

### 1. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/utils/import_utils.py:1897`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_FUNCTION_EX: a variadic function call, e.g. f(*args):

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
transformers/models/gemma3/modeling_gemma3.py:997 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/gemma3/modeling_gemma3.py:873 in forward
transformers/models/gemma3/modeling_gemma3.py:803 in get_placeholder_mask
transformers/utils/import_utils.py:1897 in torch_compilable_check
```
</details>


## train_forward

### 1. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/utils/import_utils.py:1897`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_FUNCTION_EX: a variadic function call, e.g. f(*args):

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
transformers/models/gemma3/modeling_gemma3.py:997 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/gemma3/modeling_gemma3.py:873 in forward
transformers/models/gemma3/modeling_gemma3.py:803 in get_placeholder_mask
transformers/utils/import_utils.py:1897 in torch_compilable_check
```
</details>
