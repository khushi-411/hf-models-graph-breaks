# phimoe — graph breaks

- **HF id:** `microsoft/Phi-3.5-MoE-instruct`
- **Class:** `AutoModelForCausalLM`
- **Config overrides:** `{'num_hidden_layers': 2, 'hidden_size': 1024, 'intermediate_size': 1024, 'num_attention_heads': 8, 'num_key_value_heads': 2, 'num_local_experts': 8}`
- **torch:** `2.16.0a0+git6b3a750`  |  **transformers:** `5.22.1`
- **Notes:** Phi-3.5-MoE. LongRoPE; router/input jitter noise (0.01) is active in train mode.

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ok | 7 | 6 | 4 |
| eval_with_cache | ok | 7 | 6 | 4 |
| train_forward | ok | 7 | 6 | 4 |

## eval_forward

### 1. Data-dependent branching

- **Location:** `transformers/models/phimoe/modeling_phimoe.py:84`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_FUNCTION_EX: a variadic function call, e.g. f(*args):

Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow.

      The branch condition involves a tensor computed as follows:
        # File "/home/khushi/anaconda3/envs/torch-dev/lib/python3.14/site-packages/transformers/models/phimoe/modeling_phimoe.py", line 83, in forward, code: seq_len = torch.max(position_ids) + 1
        max_2 = max(position_ids_1)
    
        # File "/home/khushi/anaconda3/envs/torch-dev/lib/python3.14/site-packages/transformers/models/phimoe/modeling_phimoe.py", line 83, in forward, code: seq_len = torch.max(position_ids) + 1
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
transformers/utils/generic.py:943 in wrapper
transformers/models/phimoe/modeling_phimoe.py:792 in forward
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/phimoe/modeling_phimoe.py:645 in forward
transformers/modeling_rope_utils.py:128 in wrapper
transformers/models/phimoe/modeling_phimoe.py:84 in forward
```
</details>

### 2. Reconstruction failure

- **Location:** `transformers/utils/output_capturing.py:272`  (hit ×1)

```
Encountered graph break that we cannot resume from. Compiling up to the previous resumable state, then skipping the rest of the function. Graph break encountered:

Reconstruction failure
  Explanation: Dynamo has no bytecode reconstruction implemented for sourceless variable DictItemsIterator().
  Hint: If Dynamo is attempting to trace a return statement and your code is attempting to return a variable that Dynamo cannot reconstruct, then remove it from the return statement.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.
  Hint: Report an issue to PyTorch if you need reconstrtuction support. Note that objects that don't have reconstruction rules may be fundamentally unreconstructable.

  Developer debug context: DictItemsIterator()

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0092.html

Stack variable source attribution:
  NullVariable originated from:
  File "/home/khushi/anaconda3/envs/torch-dev/lib/python3.14/site-packages/transformers/utils/output_capturing.py", line 272
                collected_outputs = {k.replace("output_", ""): [] for k, v in recordable_keys.items() if v}
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  NullVariable originated from:
  File "/home/khushi/anaconda3/envs/torch-dev/lib/python3.14/site-packages/transformers/utils/output_capturing.py", line 257
                recordable_keys = {
                                  ^
                ...<2 lines>...
                }
                ^
  DictItemsIterator() originated from:
  File "/home/khushi/anaconda3/envs/torch-dev/lib/python3.14/site-packages/transformers/utils/output_capturing.py", line 272
                collected_outputs = {k.replace("output_", ""): [] for k, v in recordable_keys.items() if v}
                                                                              ~~~~~~~~~~~~~~~~~~~~~^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0092.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/phimoe/modeling_phimoe.py:792 in forward
transformers/utils/output_capturing.py:272 in wrapper
```
</details>

### 3. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/utils/output_capturing.py:257`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Reconstruction failure
  Explanation: Dynamo has no bytecode reconstruction implemented for sourceless variable DictIterator().
  Hint: If Dynamo is attempting to trace a return statement and your code is attempting to return a variable that Dynamo cannot reconstruct, then remove it from the return statement.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.
  Hint: Report an issue to PyTorch if you need reconstrtuction support. Note that objects that don't have reconstruction rules may be fundamentally unreconstructable.

  Developer debug context: DictIterator()

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0092.html

Stack variable source attribution:
  NullVariable originated from:
  File "/home/khushi/anaconda3/envs/torch-dev/lib/python3.14/site-packages/transformers/utils/output_capturing.py", line 257
                recordable_keys = {
                                  ^
                ...<2 lines>...
                }
                ^
  DictIterator() originated from:
  File "/home/khushi/anaconda3/envs/torch-dev/lib/python3.14/site-packages/transformers/utils/output_capturing.py", line 259
                    for k in capturable_flags
                             ^^^^^^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0092.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/phimoe/modeling_phimoe.py:792 in forward
transformers/utils/output_capturing.py:257 in wrapper
```
</details>

### 4. Data-dependent branching

- **Location:** `transformers/models/phimoe/modeling_phimoe.py:87`  (hit ×1)

```
Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow.

      The branch condition involves a tensor computed as follows:
        # File "/home/khushi/anaconda3/envs/torch-dev/lib/python3.14/site-packages/transformers/models/phimoe/modeling_phimoe.py", line 87, in torch_dynamo_resume_in_forward_at_84, code: if seq_len > self.config.rope_parameters["original_max_position_embeddings"]
        l_seq_len_: graph input (L_seq_len_)
    
        # File "/home/khushi/anaconda3/envs/torch-dev/lib/python3.14/site-packages/transformers/models/phimoe/modeling_phimoe.py", line 87, in torch_dynamo_resume_in_forward_at_84, code: if seq_len > self.config.rope_parameters["original_max_position_embeddings"]
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
transformers/utils/generic.py:943 in wrapper
transformers/models/phimoe/modeling_phimoe.py:792 in forward
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/phimoe/modeling_phimoe.py:645 in forward
transformers/modeling_rope_utils.py:128 in wrapper
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
        # File "/home/khushi/anaconda3/envs/torch-dev/lib/python3.14/site-packages/transformers/models/phimoe/modeling_phimoe.py", line 83, in forward, code: seq_len = torch.max(position_ids) + 1
        max_2 = max(position_ids_1)
    
        # File "/home/khushi/anaconda3/envs/torch-dev/lib/python3.14/site-packages/transformers/models/phimoe/modeling_phimoe.py", line 83, in forward, code: seq_len = torch.max(position_ids) + 1
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
transformers/utils/generic.py:943 in wrapper
transformers/models/phimoe/modeling_phimoe.py:792 in forward
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/phimoe/modeling_phimoe.py:645 in forward
transformers/modeling_rope_utils.py:128 in wrapper
transformers/models/phimoe/modeling_phimoe.py:84 in forward
```
</details>

### 2. Reconstruction failure

- **Location:** `transformers/utils/output_capturing.py:272`  (hit ×1)

```
Encountered graph break that we cannot resume from. Compiling up to the previous resumable state, then skipping the rest of the function. Graph break encountered:

Reconstruction failure
  Explanation: Dynamo has no bytecode reconstruction implemented for sourceless variable DictItemsIterator().
  Hint: If Dynamo is attempting to trace a return statement and your code is attempting to return a variable that Dynamo cannot reconstruct, then remove it from the return statement.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.
  Hint: Report an issue to PyTorch if you need reconstrtuction support. Note that objects that don't have reconstruction rules may be fundamentally unreconstructable.

  Developer debug context: DictItemsIterator()

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0092.html

Stack variable source attribution:
  NullVariable originated from:
  File "/home/khushi/anaconda3/envs/torch-dev/lib/python3.14/site-packages/transformers/utils/output_capturing.py", line 272
                collected_outputs = {k.replace("output_", ""): [] for k, v in recordable_keys.items() if v}
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  NullVariable originated from:
  File "/home/khushi/anaconda3/envs/torch-dev/lib/python3.14/site-packages/transformers/utils/output_capturing.py", line 257
                recordable_keys = {
                                  ^
                ...<2 lines>...
                }
                ^
  DictItemsIterator() originated from:
  File "/home/khushi/anaconda3/envs/torch-dev/lib/python3.14/site-packages/transformers/utils/output_capturing.py", line 272
                collected_outputs = {k.replace("output_", ""): [] for k, v in recordable_keys.items() if v}
                                                                              ~~~~~~~~~~~~~~~~~~~~~^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0092.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/phimoe/modeling_phimoe.py:792 in forward
transformers/utils/output_capturing.py:272 in wrapper
```
</details>

### 3. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/utils/output_capturing.py:257`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Reconstruction failure
  Explanation: Dynamo has no bytecode reconstruction implemented for sourceless variable DictIterator().
  Hint: If Dynamo is attempting to trace a return statement and your code is attempting to return a variable that Dynamo cannot reconstruct, then remove it from the return statement.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.
  Hint: Report an issue to PyTorch if you need reconstrtuction support. Note that objects that don't have reconstruction rules may be fundamentally unreconstructable.

  Developer debug context: DictIterator()

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0092.html

Stack variable source attribution:
  NullVariable originated from:
  File "/home/khushi/anaconda3/envs/torch-dev/lib/python3.14/site-packages/transformers/utils/output_capturing.py", line 257
                recordable_keys = {
                                  ^
                ...<2 lines>...
                }
                ^
  DictIterator() originated from:
  File "/home/khushi/anaconda3/envs/torch-dev/lib/python3.14/site-packages/transformers/utils/output_capturing.py", line 259
                    for k in capturable_flags
                             ^^^^^^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0092.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/phimoe/modeling_phimoe.py:792 in forward
transformers/utils/output_capturing.py:257 in wrapper
```
</details>

### 4. Data-dependent branching

- **Location:** `transformers/models/phimoe/modeling_phimoe.py:87`  (hit ×1)

```
Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow.

      The branch condition involves a tensor computed as follows:
        # File "/home/khushi/anaconda3/envs/torch-dev/lib/python3.14/site-packages/transformers/models/phimoe/modeling_phimoe.py", line 87, in torch_dynamo_resume_in_forward_at_84, code: if seq_len > self.config.rope_parameters["original_max_position_embeddings"]
        l_seq_len_: graph input (L_seq_len_)
    
        # File "/home/khushi/anaconda3/envs/torch-dev/lib/python3.14/site-packages/transformers/models/phimoe/modeling_phimoe.py", line 87, in torch_dynamo_resume_in_forward_at_84, code: if seq_len > self.config.rope_parameters["original_max_position_embeddings"]
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
transformers/utils/generic.py:943 in wrapper
transformers/models/phimoe/modeling_phimoe.py:792 in forward
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/phimoe/modeling_phimoe.py:645 in forward
transformers/modeling_rope_utils.py:128 in wrapper
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
        # File "/home/khushi/anaconda3/envs/torch-dev/lib/python3.14/site-packages/transformers/models/phimoe/modeling_phimoe.py", line 83, in forward, code: seq_len = torch.max(position_ids) + 1
        max_2 = max(position_ids_1)
    
        # File "/home/khushi/anaconda3/envs/torch-dev/lib/python3.14/site-packages/transformers/models/phimoe/modeling_phimoe.py", line 83, in forward, code: seq_len = torch.max(position_ids) + 1
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
transformers/utils/generic.py:943 in wrapper
transformers/models/phimoe/modeling_phimoe.py:792 in forward
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/phimoe/modeling_phimoe.py:645 in forward
transformers/modeling_rope_utils.py:128 in wrapper
transformers/models/phimoe/modeling_phimoe.py:84 in forward
```
</details>

### 2. Reconstruction failure

- **Location:** `transformers/utils/output_capturing.py:272`  (hit ×1)

```
Encountered graph break that we cannot resume from. Compiling up to the previous resumable state, then skipping the rest of the function. Graph break encountered:

Reconstruction failure
  Explanation: Dynamo has no bytecode reconstruction implemented for sourceless variable DictItemsIterator().
  Hint: If Dynamo is attempting to trace a return statement and your code is attempting to return a variable that Dynamo cannot reconstruct, then remove it from the return statement.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.
  Hint: Report an issue to PyTorch if you need reconstrtuction support. Note that objects that don't have reconstruction rules may be fundamentally unreconstructable.

  Developer debug context: DictItemsIterator()

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0092.html

Stack variable source attribution:
  NullVariable originated from:
  File "/home/khushi/anaconda3/envs/torch-dev/lib/python3.14/site-packages/transformers/utils/output_capturing.py", line 272
                collected_outputs = {k.replace("output_", ""): [] for k, v in recordable_keys.items() if v}
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  NullVariable originated from:
  File "/home/khushi/anaconda3/envs/torch-dev/lib/python3.14/site-packages/transformers/utils/output_capturing.py", line 257
                recordable_keys = {
                                  ^
                ...<2 lines>...
                }
                ^
  DictItemsIterator() originated from:
  File "/home/khushi/anaconda3/envs/torch-dev/lib/python3.14/site-packages/transformers/utils/output_capturing.py", line 272
                collected_outputs = {k.replace("output_", ""): [] for k, v in recordable_keys.items() if v}
                                                                              ~~~~~~~~~~~~~~~~~~~~~^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0092.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/phimoe/modeling_phimoe.py:792 in forward
transformers/utils/output_capturing.py:272 in wrapper
```
</details>

### 3. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/utils/output_capturing.py:257`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Reconstruction failure
  Explanation: Dynamo has no bytecode reconstruction implemented for sourceless variable DictIterator().
  Hint: If Dynamo is attempting to trace a return statement and your code is attempting to return a variable that Dynamo cannot reconstruct, then remove it from the return statement.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.
  Hint: Report an issue to PyTorch if you need reconstrtuction support. Note that objects that don't have reconstruction rules may be fundamentally unreconstructable.

  Developer debug context: DictIterator()

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0092.html

Stack variable source attribution:
  NullVariable originated from:
  File "/home/khushi/anaconda3/envs/torch-dev/lib/python3.14/site-packages/transformers/utils/output_capturing.py", line 257
                recordable_keys = {
                                  ^
                ...<2 lines>...
                }
                ^
  DictIterator() originated from:
  File "/home/khushi/anaconda3/envs/torch-dev/lib/python3.14/site-packages/transformers/utils/output_capturing.py", line 259
                    for k in capturable_flags
                             ^^^^^^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0092.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/phimoe/modeling_phimoe.py:792 in forward
transformers/utils/output_capturing.py:257 in wrapper
```
</details>

### 4. Data-dependent branching

- **Location:** `transformers/models/phimoe/modeling_phimoe.py:87`  (hit ×1)

```
Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow.

      The branch condition involves a tensor computed as follows:
        # File "/home/khushi/anaconda3/envs/torch-dev/lib/python3.14/site-packages/transformers/models/phimoe/modeling_phimoe.py", line 87, in torch_dynamo_resume_in_forward_at_84, code: if seq_len > self.config.rope_parameters["original_max_position_embeddings"]
        l_seq_len_: graph input (L_seq_len_)
    
        # File "/home/khushi/anaconda3/envs/torch-dev/lib/python3.14/site-packages/transformers/models/phimoe/modeling_phimoe.py", line 87, in torch_dynamo_resume_in_forward_at_84, code: if seq_len > self.config.rope_parameters["original_max_position_embeddings"]
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
transformers/utils/generic.py:943 in wrapper
transformers/models/phimoe/modeling_phimoe.py:792 in forward
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/phimoe/modeling_phimoe.py:645 in forward
transformers/modeling_rope_utils.py:128 in wrapper
transformers/models/phimoe/modeling_phimoe.py:87 in forward
```
</details>
