# unlimited_ocr — graph breaks

- **HF id:** `baidu/Unlimited-OCR`
- **Class:** `AutoModelForImageTextToText`
- **Config overrides:** `{'text_config.num_hidden_layers': 2, 'text_config.layer_types': ['reference_sliding_attention', 'reference_sliding_attention'], 'text_config.mlp_layer_types': ['dense', 'sparse'], 'text_config.n_routed_experts': 8, 'text_config.num_experts_per_tok': 2, 'vision_config.sam_config.num_hidden_layers': 2, 'vision_config.sam_config.global_attn_indexes': [1], 'vision_config.encoder_config.num_hidden_layers': 2, 'dtype': 'float32', 'text_config.dtype': 'float32', 'vision_config.dtype': 'float32'}`
- **torch:** `2.16.0a0+git57f02f8`  |  **transformers:** `5.24.0.dev0` (git `046bdb8cc2`)
- **Notes:** Top-100 #59. Config dtype is bfloat16 (from_config keeps it), so float32 is forced. DeepSeek-OCR style: SAM + CLIP vision, DeepseekV2-MoE LM with reference sliding-window attention. Global view only (no local tiles).

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ok | 4 | 3 | 1 |
| eval_with_cache | ok | 4 | 3 | 1 |
| train_forward | ok | 4 | 3 | 1 |

## eval_forward

### 1. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/utils/import_utils.py:1993`  (hit ×1)

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
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/unlimited_ocr/modeling_unlimited_ocr.py:2100 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/unlimited_ocr/modeling_unlimited_ocr.py:1957 in forward
transformers/models/unlimited_ocr/modeling_unlimited_ocr.py:1910 in get_placeholder_mask
transformers/utils/import_utils.py:1993 in torch_compilable_check
```
</details>


## eval_with_cache

### 1. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/utils/import_utils.py:1993`  (hit ×1)

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
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/unlimited_ocr/modeling_unlimited_ocr.py:2100 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/unlimited_ocr/modeling_unlimited_ocr.py:1957 in forward
transformers/models/unlimited_ocr/modeling_unlimited_ocr.py:1910 in get_placeholder_mask
transformers/utils/import_utils.py:1993 in torch_compilable_check
```
</details>


## train_forward

### 1. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/utils/import_utils.py:1993`  (hit ×1)

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
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/unlimited_ocr/modeling_unlimited_ocr.py:2100 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/unlimited_ocr/modeling_unlimited_ocr.py:1957 in forward
transformers/models/unlimited_ocr/modeling_unlimited_ocr.py:1910 in get_placeholder_mask
transformers/utils/import_utils.py:1993 in torch_compilable_check
```
</details>
