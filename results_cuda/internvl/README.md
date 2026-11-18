# internvl — graph breaks

- **HF id:** `OpenGVLab/InternVL3-1B-hf`
- **Class:** `AutoModelForImageTextToText`
- **Config overrides:** `{'text_config.num_hidden_layers': 2, 'vision_config.num_hidden_layers': 2}`
- **torch:** `2.16.0a0+git57f02f8`  |  **transformers:** `5.24.0.dev0` (git `b7fe69cb19`)

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ok | 9 | 8 | 3 |
| eval_with_cache | ok | 9 | 8 | 3 |
| train_forward | ok | 9 | 8 | 3 |

## eval_forward

### 1. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/utils/import_utils.py:1975`  (hit ×1)

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
transformers/models/internvl/modeling_internvl.py:823 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/internvl/modeling_internvl.py:649 in forward
transformers/models/internvl/modeling_internvl.py:606 in get_placeholder_mask
transformers/utils/import_utils.py:1975 in torch_compilable_check
```
</details>

### 2. Data-dependent branching

- **Location:** `transformers/modeling_rope_utils.py:101`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_FUNCTION_EX: a variadic function call, e.g. f(*args):

Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow.

      The branch condition involves a tensor computed as follows:
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/modeling_rope_utils.py", line 89, in dynamic_frequency_update, code: seq_len = torch.max(position_ids) + 1
        max_1 = max(position_ids_1)
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/modeling_rope_utils.py", line 89, in dynamic_frequency_update, code: seq_len = torch.max(position_ids) + 1
        seq_len = add(max_1, 1)
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/modeling_rope_utils.py", line 101, in dynamic_frequency_update, code: if seq_len > max_seq_len_cached:  # growth
        gt = gt(seq_len, 32768)
    
  Hint: For the common pattern `if tensor_cond: x = transform(x)` (e.g. clamping inf/nan values), consider making the code branchless by always applying the transform. Operations like torch.clamp, torch.nan_to_num, and torch.where are typically no-ops on well-behaved inputs and compile without graph breaks.
  Hint: This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround.
  Hint: Use `torch.cond` to express dynamic control flow.

  Developer debug context: attempted to jump with TensorVariable()

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html

<details><summary>User stack</summary>

```
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/internvl/modeling_internvl.py:823 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/internvl/modeling_internvl.py:654 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen2/modeling_qwen2.py:380 in forward
transformers/modeling_rope_utils.py:126 in wrapper
transformers/modeling_rope_utils.py:101 in dynamic_frequency_update
```
</details>

### 3. Data-dependent branching

- **Location:** `transformers/modeling_rope_utils.py:113`  (hit ×1)

```
Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow.

      The branch condition involves a tensor computed as follows:
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/modeling_rope_utils.py", line 113, in torch_dynamo_resume_in_dynamic_frequency_update_at_101, code: if seq_len < self.original_max_seq_len and max_seq_len_cached > self.original_max_seq_len:  # reset
        l_seq_len_: graph input (L_seq_len_)
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/modeling_rope_utils.py", line 113, in torch_dynamo_resume_in_dynamic_frequency_update_at_101, code: if seq_len < self.original_max_seq_len and max_seq_len_cached > self.original_max_seq_len:  # reset
        lt = lt(l_seq_len_, 32768)
    
  Hint: For the common pattern `if tensor_cond: x = transform(x)` (e.g. clamping inf/nan values), consider making the code branchless by always applying the transform. Operations like torch.clamp, torch.nan_to_num, and torch.where are typically no-ops on well-behaved inputs and compile without graph breaks.
  Hint: This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround.
  Hint: Use `torch.cond` to express dynamic control flow.

  Developer debug context: attempted to jump with TensorVariable()

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html

<details><summary>User stack</summary>

```
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/internvl/modeling_internvl.py:823 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/internvl/modeling_internvl.py:654 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen2/modeling_qwen2.py:380 in forward
transformers/modeling_rope_utils.py:126 in wrapper
transformers/modeling_rope_utils.py:113 in dynamic_frequency_update
```
</details>


## eval_with_cache

### 1. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/utils/import_utils.py:1975`  (hit ×1)

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
transformers/models/internvl/modeling_internvl.py:823 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/internvl/modeling_internvl.py:649 in forward
transformers/models/internvl/modeling_internvl.py:606 in get_placeholder_mask
transformers/utils/import_utils.py:1975 in torch_compilable_check
```
</details>

### 2. Data-dependent branching

- **Location:** `transformers/modeling_rope_utils.py:101`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_FUNCTION_EX: a variadic function call, e.g. f(*args):

Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow.

      The branch condition involves a tensor computed as follows:
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/modeling_rope_utils.py", line 89, in dynamic_frequency_update, code: seq_len = torch.max(position_ids) + 1
        max_1 = max(position_ids_1)
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/modeling_rope_utils.py", line 89, in dynamic_frequency_update, code: seq_len = torch.max(position_ids) + 1
        seq_len = add(max_1, 1)
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/modeling_rope_utils.py", line 101, in dynamic_frequency_update, code: if seq_len > max_seq_len_cached:  # growth
        gt = gt(seq_len, 32768)
    
  Hint: For the common pattern `if tensor_cond: x = transform(x)` (e.g. clamping inf/nan values), consider making the code branchless by always applying the transform. Operations like torch.clamp, torch.nan_to_num, and torch.where are typically no-ops on well-behaved inputs and compile without graph breaks.
  Hint: This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround.
  Hint: Use `torch.cond` to express dynamic control flow.

  Developer debug context: attempted to jump with TensorVariable()

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html

<details><summary>User stack</summary>

```
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/internvl/modeling_internvl.py:823 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/internvl/modeling_internvl.py:654 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen2/modeling_qwen2.py:380 in forward
transformers/modeling_rope_utils.py:126 in wrapper
transformers/modeling_rope_utils.py:101 in dynamic_frequency_update
```
</details>

### 3. Data-dependent branching

- **Location:** `transformers/modeling_rope_utils.py:113`  (hit ×1)

```
Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow.

      The branch condition involves a tensor computed as follows:
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/modeling_rope_utils.py", line 113, in torch_dynamo_resume_in_dynamic_frequency_update_at_101, code: if seq_len < self.original_max_seq_len and max_seq_len_cached > self.original_max_seq_len:  # reset
        l_seq_len_: graph input (L_seq_len_)
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/modeling_rope_utils.py", line 113, in torch_dynamo_resume_in_dynamic_frequency_update_at_101, code: if seq_len < self.original_max_seq_len and max_seq_len_cached > self.original_max_seq_len:  # reset
        lt = lt(l_seq_len_, 32768)
    
  Hint: For the common pattern `if tensor_cond: x = transform(x)` (e.g. clamping inf/nan values), consider making the code branchless by always applying the transform. Operations like torch.clamp, torch.nan_to_num, and torch.where are typically no-ops on well-behaved inputs and compile without graph breaks.
  Hint: This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround.
  Hint: Use `torch.cond` to express dynamic control flow.

  Developer debug context: attempted to jump with TensorVariable()

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html

<details><summary>User stack</summary>

```
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/internvl/modeling_internvl.py:823 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/internvl/modeling_internvl.py:654 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen2/modeling_qwen2.py:380 in forward
transformers/modeling_rope_utils.py:126 in wrapper
transformers/modeling_rope_utils.py:113 in dynamic_frequency_update
```
</details>


## train_forward

### 1. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/utils/import_utils.py:1975`  (hit ×1)

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
transformers/models/internvl/modeling_internvl.py:823 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/internvl/modeling_internvl.py:649 in forward
transformers/models/internvl/modeling_internvl.py:606 in get_placeholder_mask
transformers/utils/import_utils.py:1975 in torch_compilable_check
```
</details>

### 2. Data-dependent branching

- **Location:** `transformers/modeling_rope_utils.py:101`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_FUNCTION_EX: a variadic function call, e.g. f(*args):

Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow.

      The branch condition involves a tensor computed as follows:
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/modeling_rope_utils.py", line 89, in dynamic_frequency_update, code: seq_len = torch.max(position_ids) + 1
        max_1 = max(position_ids_1)
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/modeling_rope_utils.py", line 89, in dynamic_frequency_update, code: seq_len = torch.max(position_ids) + 1
        seq_len = add(max_1, 1)
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/modeling_rope_utils.py", line 101, in dynamic_frequency_update, code: if seq_len > max_seq_len_cached:  # growth
        gt = gt(seq_len, 32768)
    
  Hint: For the common pattern `if tensor_cond: x = transform(x)` (e.g. clamping inf/nan values), consider making the code branchless by always applying the transform. Operations like torch.clamp, torch.nan_to_num, and torch.where are typically no-ops on well-behaved inputs and compile without graph breaks.
  Hint: This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround.
  Hint: Use `torch.cond` to express dynamic control flow.

  Developer debug context: attempted to jump with TensorVariable()

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html

<details><summary>User stack</summary>

```
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/internvl/modeling_internvl.py:823 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/internvl/modeling_internvl.py:654 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen2/modeling_qwen2.py:380 in forward
transformers/modeling_rope_utils.py:126 in wrapper
transformers/modeling_rope_utils.py:101 in dynamic_frequency_update
```
</details>

### 3. Data-dependent branching

- **Location:** `transformers/modeling_rope_utils.py:113`  (hit ×1)

```
Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow.

      The branch condition involves a tensor computed as follows:
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/modeling_rope_utils.py", line 113, in torch_dynamo_resume_in_dynamic_frequency_update_at_101, code: if seq_len < self.original_max_seq_len and max_seq_len_cached > self.original_max_seq_len:  # reset
        l_seq_len_: graph input (L_seq_len_)
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/modeling_rope_utils.py", line 113, in torch_dynamo_resume_in_dynamic_frequency_update_at_101, code: if seq_len < self.original_max_seq_len and max_seq_len_cached > self.original_max_seq_len:  # reset
        lt = lt(l_seq_len_, 32768)
    
  Hint: For the common pattern `if tensor_cond: x = transform(x)` (e.g. clamping inf/nan values), consider making the code branchless by always applying the transform. Operations like torch.clamp, torch.nan_to_num, and torch.where are typically no-ops on well-behaved inputs and compile without graph breaks.
  Hint: This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround.
  Hint: Use `torch.cond` to express dynamic control flow.

  Developer debug context: attempted to jump with TensorVariable()

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html

<details><summary>User stack</summary>

```
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/internvl/modeling_internvl.py:823 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/internvl/modeling_internvl.py:654 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/qwen2/modeling_qwen2.py:380 in forward
transformers/modeling_rope_utils.py:126 in wrapper
transformers/modeling_rope_utils.py:113 in dynamic_frequency_update
```
</details>
