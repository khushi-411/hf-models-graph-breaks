# phimoe — graph breaks

- **HF id:** `microsoft/Phi-3.5-MoE-instruct`
- **Class:** `AutoModelForCausalLM`
- **Config overrides:** `{'num_hidden_layers': 2, 'hidden_size': 1024, 'intermediate_size': 1024, 'num_attention_heads': 8, 'num_key_value_heads': 2, 'num_local_experts': 8}`
- **torch:** `2.16.0a0+git57f02f8`  |  **transformers:** `5.24.0.dev0` (git `b7fe69cb19`)
- **Notes:** Phi-3.5-MoE. LongRoPE; router/input jitter noise (0.01) is active in train mode.

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ok | 7 | 6 | 2 |
| eval_with_cache | ok | 7 | 6 | 2 |
| train_forward | ok | 7 | 6 | 2 |

## eval_forward

### 1. Data-dependent branching

- **Location:** `transformers/models/phimoe/modeling_phimoe.py:84`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_FUNCTION_EX: a variadic function call, e.g. f(*args):

Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow.

      The branch condition involves a tensor computed as follows:
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/phimoe/modeling_phimoe.py", line 83, in forward, code: seq_len = torch.max(position_ids) + 1
        max_2 = max(position_ids_1)
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/phimoe/modeling_phimoe.py", line 83, in forward, code: seq_len = torch.max(position_ids) + 1
        seq_len_1 = add(max_2, 1)
    
  Hint: The branch condition uses a scalar integer tensor. Consider rewriting the computation to use plain Python ints (e.g. use int attributes instead of tensor buffers) so the condition becomes a shape guard instead of data-dependent branching.
  Hint: This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround.
  Hint: Use `torch.cond` to express dynamic control flow.

  Developer debug context: attempted to jump with TensorVariable()

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/phimoe/modeling_phimoe.py:798 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/phimoe/modeling_phimoe.py:647 in forward
transformers/modeling_rope_utils.py:129 in wrapper
transformers/models/phimoe/modeling_phimoe.py:84 in forward
```
</details>

### 2. Data-dependent branching

- **Location:** `transformers/models/phimoe/modeling_phimoe.py:87`  (hit ×1)

```
Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow.

      The branch condition involves a tensor computed as follows:
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/phimoe/modeling_phimoe.py", line 87, in torch_dynamo_resume_in_forward_at_84, code: if seq_len > self.config.rope_parameters["original_max_position_embeddings"]
        l_seq_len_: graph input (L_seq_len_)
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/phimoe/modeling_phimoe.py", line 87, in torch_dynamo_resume_in_forward_at_84, code: if seq_len > self.config.rope_parameters["original_max_position_embeddings"]
        gt = gt(l_seq_len_, 4096)
    
  Hint: For the common pattern `if tensor_cond: x = transform(x)` (e.g. clamping inf/nan values), consider making the code branchless by always applying the transform. Operations like torch.clamp, torch.nan_to_num, and torch.where are typically no-ops on well-behaved inputs and compile without graph breaks.
  Hint: This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround.
  Hint: Use `torch.cond` to express dynamic control flow.

  Developer debug context: attempted to jump with TensorVariable()

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/phimoe/modeling_phimoe.py:798 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/phimoe/modeling_phimoe.py:647 in forward
transformers/modeling_rope_utils.py:129 in wrapper
transformers/models/phimoe/modeling_phimoe.py:87 in forward
```
</details>


## eval_with_cache

### 1. Data-dependent branching

- **Location:** `transformers/models/phimoe/modeling_phimoe.py:84`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_FUNCTION_EX: a variadic function call, e.g. f(*args):

Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow.

      The branch condition involves a tensor computed as follows:
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/phimoe/modeling_phimoe.py", line 83, in forward, code: seq_len = torch.max(position_ids) + 1
        max_2 = max(position_ids_1)
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/phimoe/modeling_phimoe.py", line 83, in forward, code: seq_len = torch.max(position_ids) + 1
        seq_len_1 = add(max_2, 1)
    
  Hint: The branch condition uses a scalar integer tensor. Consider rewriting the computation to use plain Python ints (e.g. use int attributes instead of tensor buffers) so the condition becomes a shape guard instead of data-dependent branching.
  Hint: This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround.
  Hint: Use `torch.cond` to express dynamic control flow.

  Developer debug context: attempted to jump with TensorVariable()

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/phimoe/modeling_phimoe.py:798 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/phimoe/modeling_phimoe.py:647 in forward
transformers/modeling_rope_utils.py:129 in wrapper
transformers/models/phimoe/modeling_phimoe.py:84 in forward
```
</details>

### 2. Data-dependent branching

- **Location:** `transformers/models/phimoe/modeling_phimoe.py:87`  (hit ×1)

```
Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow.

      The branch condition involves a tensor computed as follows:
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/phimoe/modeling_phimoe.py", line 87, in torch_dynamo_resume_in_forward_at_84, code: if seq_len > self.config.rope_parameters["original_max_position_embeddings"]
        l_seq_len_: graph input (L_seq_len_)
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/phimoe/modeling_phimoe.py", line 87, in torch_dynamo_resume_in_forward_at_84, code: if seq_len > self.config.rope_parameters["original_max_position_embeddings"]
        gt = gt(l_seq_len_, 4096)
    
  Hint: For the common pattern `if tensor_cond: x = transform(x)` (e.g. clamping inf/nan values), consider making the code branchless by always applying the transform. Operations like torch.clamp, torch.nan_to_num, and torch.where are typically no-ops on well-behaved inputs and compile without graph breaks.
  Hint: This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround.
  Hint: Use `torch.cond` to express dynamic control flow.

  Developer debug context: attempted to jump with TensorVariable()

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/phimoe/modeling_phimoe.py:798 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/phimoe/modeling_phimoe.py:647 in forward
transformers/modeling_rope_utils.py:129 in wrapper
transformers/models/phimoe/modeling_phimoe.py:87 in forward
```
</details>


## train_forward

### 1. Data-dependent branching

- **Location:** `transformers/models/phimoe/modeling_phimoe.py:84`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_FUNCTION_EX: a variadic function call, e.g. f(*args):

Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow.

      The branch condition involves a tensor computed as follows:
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/phimoe/modeling_phimoe.py", line 83, in forward, code: seq_len = torch.max(position_ids) + 1
        max_2 = max(position_ids_1)
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/phimoe/modeling_phimoe.py", line 83, in forward, code: seq_len = torch.max(position_ids) + 1
        seq_len_1 = add(max_2, 1)
    
  Hint: The branch condition uses a scalar integer tensor. Consider rewriting the computation to use plain Python ints (e.g. use int attributes instead of tensor buffers) so the condition becomes a shape guard instead of data-dependent branching.
  Hint: This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround.
  Hint: Use `torch.cond` to express dynamic control flow.

  Developer debug context: attempted to jump with TensorVariable()

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/phimoe/modeling_phimoe.py:798 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/phimoe/modeling_phimoe.py:647 in forward
transformers/modeling_rope_utils.py:129 in wrapper
transformers/models/phimoe/modeling_phimoe.py:84 in forward
```
</details>

### 2. Data-dependent branching

- **Location:** `transformers/models/phimoe/modeling_phimoe.py:87`  (hit ×1)

```
Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow.

      The branch condition involves a tensor computed as follows:
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/phimoe/modeling_phimoe.py", line 87, in torch_dynamo_resume_in_forward_at_84, code: if seq_len > self.config.rope_parameters["original_max_position_embeddings"]
        l_seq_len_: graph input (L_seq_len_)
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/phimoe/modeling_phimoe.py", line 87, in torch_dynamo_resume_in_forward_at_84, code: if seq_len > self.config.rope_parameters["original_max_position_embeddings"]
        gt = gt(l_seq_len_, 4096)
    
  Hint: For the common pattern `if tensor_cond: x = transform(x)` (e.g. clamping inf/nan values), consider making the code branchless by always applying the transform. Operations like torch.clamp, torch.nan_to_num, and torch.where are typically no-ops on well-behaved inputs and compile without graph breaks.
  Hint: This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround.
  Hint: Use `torch.cond` to express dynamic control flow.

  Developer debug context: attempted to jump with TensorVariable()

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/phimoe/modeling_phimoe.py:798 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/phimoe/modeling_phimoe.py:647 in forward
transformers/modeling_rope_utils.py:129 in wrapper
transformers/models/phimoe/modeling_phimoe.py:87 in forward
```
</details>
