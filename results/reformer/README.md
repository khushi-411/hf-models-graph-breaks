# reformer — graph breaks

- **HF id:** `google/reformer-crime-and-punishment`
- **Class:** `AutoModelForCausalLM`
- **Config overrides:** `{'num_hidden_layers': 2, 'attn_layers': ['local', 'lsh'], 'axial_pos_shape': [4, 8], 'local_attn_chunk_length': 16, 'lsh_attn_chunk_length': 16}`
- **torch:** `2.16.0a0+git9933f27`  |  **transformers:** `5.22.1`
- **Notes:** LSH attention, reversible layers.

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ok | 16 | 15 | 6 |
| train_forward | ok | 16 | 15 | 10 |

## eval_forward

### 1. Data-dependent branching

- **Location:** `transformers/models/reformer/modeling_reformer.py:2023`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_KW: a function call with keyword arguments, e.g. f(x=True):

Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow.

      The branch condition involves a tensor computed as follows:
        # File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/reformer/modeling_reformer.py", line 168, in _get_least_common_mult_chunk_len, code: return np.lcm(config.lsh_attn_chunk_length, config.local_attn_chunk_length)
        least_common_mult_chunk_length = wrapped_lcm(16, 16)
    
        # File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/reformer/modeling_reformer.py", line 2023, in forward, code: input_shape[-1] % least_common_mult_chunk_length != 0
        wrapped_mod = wrapped_mod(32, least_common_mult_chunk_length)
    
        # File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/reformer/modeling_reformer.py", line 2023, in forward, code: input_shape[-1] % least_common_mult_chunk_length != 0
        wrapped_ne = wrapped_ne(wrapped_mod, 0)
    
  Hint: For the common pattern `if tensor_cond: x = transform(x)` (e.g. clamping inf/nan values), consider making the code branchless by always applying the transform. Operations like torch.clamp, torch.nan_to_num, and torch.where are typically no-ops on well-behaved inputs and compile without graph breaks.
  Hint: This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround.
  Hint: Use `torch.cond` to express dynamic control flow.

  Developer debug context: attempted to jump with NumpyNdarrayVariable()

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html

<details><summary>User stack</summary>

```
transformers/models/reformer/modeling_reformer.py:2023 in forward
```
</details>

### 2. Data-dependent branching

- **Location:** `transformers/models/reformer/modeling_reformer.py:2028`  (hit ×1)

```
Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow.

      The branch condition involves a tensor computed as follows:
        # File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/reformer/modeling_reformer.py", line 2028, in torch_dynamo_resume_in_forward_at_2023, code: if must_pad_to_match_chunk_length:
        l_stack0_: graph input (L_stack0_)
    
  Hint: For the common pattern `if tensor_cond: x = transform(x)` (e.g. clamping inf/nan values), consider making the code branchless by always applying the transform. Operations like torch.clamp, torch.nan_to_num, and torch.where are typically no-ops on well-behaved inputs and compile without graph breaks.
  Hint: This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround.
  Hint: Use `torch.cond` to express dynamic control flow.

  Developer debug context: attempted to jump with LazyVariableTracker(realized: NumpyNdarrayVariable())

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html

<details><summary>User stack</summary>

```
transformers/models/reformer/modeling_reformer.py:2222 in forward
transformers/models/reformer/modeling_reformer.py:2028 in forward
```
</details>

### 3. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/models/reformer/modeling_reformer.py:270`  (hit ×1)

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
transformers/models/reformer/modeling_reformer.py:2222 in forward
transformers/models/reformer/modeling_reformer.py:270 in forward
```
</details>

### 4. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/models/reformer/modeling_reformer.py:750`  (hit ×1)

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
transformers/models/reformer/modeling_reformer.py:2222 in forward
transformers/models/reformer/modeling_reformer.py:559 in forward
transformers/models/reformer/modeling_reformer.py:750 in _hash_vectors
```
</details>

### 5. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/reformer/modeling_reformer.py:750`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Unsupported Tensor.item() call with capture_scalar_outputs=False
  Explanation: Dynamo does not support tracing `Tensor.item()` with config.capture_scalar_outputs=False.
  Hint: Set `torch._dynamo.config.capture_scalar_outputs = True` or `export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` to include these operations in the captured graph.

  Developer debug context: call_method TensorVariable() item () {}

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

*** While handling this graph break, another graph break occurred: ***

graph break in loop
  Explanation: torch.compile detected a graph break in a for/while loop. Skipping the frame and falling back to eager, as graph breaks in loops are not supported.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: frame skipped: forward (/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/reformer/modeling_reformer.py line 1660)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  DictValuesIterator() originated from:
  File "/home/khushi/Documents/osc/work/pytorch/torch/nn/modules/container.py", line 416
            return iter(self._modules.values())
                   ~~~~^^^^^^^^^^^^^^^^^^^^^^^^
  NullVariable originated from:
  File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/reformer/modeling_reformer.py", line 1680
            for layer in layers:
                         ^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

<details><summary>User stack</summary>

```
transformers/models/reformer/modeling_reformer.py:2222 in forward
transformers/models/reformer/modeling_reformer.py:2060 in forward
transformers/models/reformer/modeling_reformer.py:559 in forward
transformers/models/reformer/modeling_reformer.py:750 in _hash_vectors
```
</details>

### 6. Data-dependent branching

- **Location:** `transformers/models/reformer/modeling_reformer.py:2073`  (hit ×1)

```
Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow.

      The branch condition involves a tensor computed as follows:
        # File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/reformer/modeling_reformer.py", line 2073, in torch_dynamo_resume_in_forward_at_2060, code: if must_pad_to_match_chunk_length:
        l_must_pad_to_match_chunk_length_: graph input (L_must_pad_to_match_chunk_length_)
    
  Hint: For the common pattern `if tensor_cond: x = transform(x)` (e.g. clamping inf/nan values), consider making the code branchless by always applying the transform. Operations like torch.clamp, torch.nan_to_num, and torch.where are typically no-ops on well-behaved inputs and compile without graph breaks.
  Hint: This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround.
  Hint: Use `torch.cond` to express dynamic control flow.

  Developer debug context: attempted to jump with LazyVariableTracker(realized: NumpyNdarrayVariable())

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html

<details><summary>User stack</summary>

```
transformers/models/reformer/modeling_reformer.py:2222 in forward
transformers/models/reformer/modeling_reformer.py:2073 in forward
```
</details>


## train_forward

### 1. Data-dependent branching

- **Location:** `transformers/models/reformer/modeling_reformer.py:2023`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_KW: a function call with keyword arguments, e.g. f(x=True):

Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow.

      The branch condition involves a tensor computed as follows:
        # File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/reformer/modeling_reformer.py", line 168, in _get_least_common_mult_chunk_len, code: return np.lcm(config.lsh_attn_chunk_length, config.local_attn_chunk_length)
        least_common_mult_chunk_length = wrapped_lcm(16, 16)
    
        # File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/reformer/modeling_reformer.py", line 2023, in forward, code: input_shape[-1] % least_common_mult_chunk_length != 0
        wrapped_mod = wrapped_mod(32, least_common_mult_chunk_length)
    
        # File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/reformer/modeling_reformer.py", line 2023, in forward, code: input_shape[-1] % least_common_mult_chunk_length != 0
        wrapped_ne = wrapped_ne(wrapped_mod, 0)
    
  Hint: For the common pattern `if tensor_cond: x = transform(x)` (e.g. clamping inf/nan values), consider making the code branchless by always applying the transform. Operations like torch.clamp, torch.nan_to_num, and torch.where are typically no-ops on well-behaved inputs and compile without graph breaks.
  Hint: This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround.
  Hint: Use `torch.cond` to express dynamic control flow.

  Developer debug context: attempted to jump with NumpyNdarrayVariable()

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html

<details><summary>User stack</summary>

```
transformers/models/reformer/modeling_reformer.py:2023 in forward
```
</details>

### 2. Data-dependent branching

- **Location:** `transformers/models/reformer/modeling_reformer.py:2028`  (hit ×1)

```
Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow.

      The branch condition involves a tensor computed as follows:
        # File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/reformer/modeling_reformer.py", line 2028, in torch_dynamo_resume_in_forward_at_2023, code: if must_pad_to_match_chunk_length:
        l_stack0_: graph input (L_stack0_)
    
  Hint: For the common pattern `if tensor_cond: x = transform(x)` (e.g. clamping inf/nan values), consider making the code branchless by always applying the transform. Operations like torch.clamp, torch.nan_to_num, and torch.where are typically no-ops on well-behaved inputs and compile without graph breaks.
  Hint: This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround.
  Hint: Use `torch.cond` to express dynamic control flow.

  Developer debug context: attempted to jump with LazyVariableTracker(realized: NumpyNdarrayVariable())

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html

<details><summary>User stack</summary>

```
transformers/models/reformer/modeling_reformer.py:2222 in forward
transformers/models/reformer/modeling_reformer.py:2028 in forward
```
</details>

### 3. Attempted to call function marked as skipped

- **Location:** `transformers/models/reformer/modeling_reformer.py:1511`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_KW: a function call with keyword arguments, e.g. f(x=True):

Attempted to call function marked as skipped
  Higher Order Operator: autograd.Function
  Explanation: Dynamo does not know how to trace the builtin `<unknown module>.Generator.seed.` This function is either a Python builtin (e.g. _warnings.warn) or a third-party C/C++ Python extension (perhaps created with pybind).
  Hint: If it is a Python builtin, please file an issue on GitHub so the PyTorch team can add support for it and see the next case for a workaround.
  Hint: If it is a third-party C/C++ Python extension, please either wrap it into a PyTorch-understood custom operator (see https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html for more details) or, if it is traceable, use `torch.compiler.allow_in_graph`.

  Developer debug context: module: <unknown module>, qualname: Generator.seed, skip reason: cannot determine source file for  (likely a C extension or builtin)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0007.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0007.html

<details><summary>User stack</summary>

```
transformers/models/reformer/modeling_reformer.py:2222 in forward
transformers/models/reformer/modeling_reformer.py:1791 in forward
transformers/models/reformer/modeling_reformer.py:1551 in forward
transformers/models/reformer/modeling_reformer.py:1511 in _init_attention_seed
```
</details>

### 4. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/reformer/modeling_reformer.py:1511`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Attempted to call function marked as skipped
  Explanation: Dynamo does not know how to trace the builtin `<unknown module>.Generator.seed.` This function is either a Python builtin (e.g. _warnings.warn) or a third-party C/C++ Python extension (perhaps created with pybind).
  Hint: If it is a Python builtin, please file an issue on GitHub so the PyTorch team can add support for it and see the next case for a workaround.
  Hint: If it is a third-party C/C++ Python extension, please either wrap it into a PyTorch-understood custom operator (see https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html for more details) or, if it is traceable, use `torch.compiler.allow_in_graph`.

  Developer debug context: module: <unknown module>, qualname: Generator.seed, skip reason: cannot determine source file for  (likely a C extension or builtin)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0007.html

*** While handling this graph break, another graph break occurred: ***

graph break in loop
  Explanation: torch.compile detected a graph break in a for/while loop. Skipping the frame and falling back to eager, as graph breaks in loops are not supported.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: frame skipped: forward (/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/reformer/modeling_reformer.py line 1660)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  DictValuesIterator() originated from:
  File "/home/khushi/Documents/osc/work/pytorch/torch/nn/modules/container.py", line 416
            return iter(self._modules.values())
                   ~~~~^^^^^^^^^^^^^^^^^^^^^^^^
  NullVariable originated from:
  File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/reformer/modeling_reformer.py", line 1680
            for layer in layers:
                         ^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0007.html

<details><summary>User stack</summary>

```
transformers/models/reformer/modeling_reformer.py:2222 in forward
transformers/models/reformer/modeling_reformer.py:2060 in forward
transformers/models/reformer/modeling_reformer.py:1551 in forward
transformers/models/reformer/modeling_reformer.py:1511 in _init_attention_seed
```
</details>

### 5. Attempted to call function marked as skipped

- **Location:** `transformers/models/reformer/modeling_reformer.py:1516`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Attempted to call function marked as skipped
  Explanation: Dynamo developers have intentionally marked that the function `manual_seed` in file `/home/khushi/Documents/osc/work/pytorch/torch/_compile.py` should not be traced.
  Hint: Avoid calling the function `manual_seed`.
  Hint: Apply `@torch._dynamo.dont_skip_tracing` to the function `manual_seed` to force tracing into the function. More graph breaks may occur as a result of attempting to trace into the function.
  Hint: Please file an issue to PyTorch.

  Developer debug context: module: torch.random, qualname: manual_seed, skip reason: file matches MOD_SKIPLIST (/home/khushi/Documents/osc/work/pytorch/torch/_compile)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0007.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0007.html

<details><summary>User stack</summary>

```
transformers/models/reformer/modeling_reformer.py:2222 in forward
transformers/models/reformer/modeling_reformer.py:2060 in forward
transformers/models/reformer/modeling_reformer.py:1684 in forward
transformers/models/reformer/modeling_reformer.py:1551 in forward
transformers/models/reformer/modeling_reformer.py:1516 in _init_attention_seed
```
</details>

### 6. Attempted to call function marked as skipped

- **Location:** `transformers/models/reformer/modeling_reformer.py:1528`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Attempted to call function marked as skipped
  Explanation: Dynamo does not know how to trace the builtin `<unknown module>.Generator.seed.` This function is either a Python builtin (e.g. _warnings.warn) or a third-party C/C++ Python extension (perhaps created with pybind).
  Hint: If it is a Python builtin, please file an issue on GitHub so the PyTorch team can add support for it and see the next case for a workaround.
  Hint: If it is a third-party C/C++ Python extension, please either wrap it into a PyTorch-understood custom operator (see https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html for more details) or, if it is traceable, use `torch.compiler.allow_in_graph`.

  Developer debug context: module: <unknown module>, qualname: Generator.seed, skip reason: cannot determine source file for  (likely a C extension or builtin)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0007.html

Stack variable source attribution:
  WithExitFunctionVariable() originated from:
  File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/reformer/modeling_reformer.py", line 1551
                    self._init_attention_seed()
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0007.html

<details><summary>User stack</summary>

```
transformers/models/reformer/modeling_reformer.py:2222 in forward
transformers/models/reformer/modeling_reformer.py:2060 in forward
transformers/models/reformer/modeling_reformer.py:1684 in forward
transformers/models/reformer/modeling_reformer.py:1575 in forward
transformers/models/reformer/modeling_reformer.py:1528 in _init_feed_forward_seed
```
</details>

### 7. Attempted to call function marked as skipped

- **Location:** `transformers/models/reformer/modeling_reformer.py:1533`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Attempted to call function marked as skipped
  Explanation: Dynamo developers have intentionally marked that the function `manual_seed` in file `/home/khushi/Documents/osc/work/pytorch/torch/_compile.py` should not be traced.
  Hint: Avoid calling the function `manual_seed`.
  Hint: Apply `@torch._dynamo.dont_skip_tracing` to the function `manual_seed` to force tracing into the function. More graph breaks may occur as a result of attempting to trace into the function.
  Hint: Please file an issue to PyTorch.

  Developer debug context: module: torch.random, qualname: manual_seed, skip reason: file matches MOD_SKIPLIST (/home/khushi/Documents/osc/work/pytorch/torch/_compile)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0007.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0007.html

<details><summary>User stack</summary>

```
transformers/models/reformer/modeling_reformer.py:2222 in forward
transformers/models/reformer/modeling_reformer.py:2060 in forward
transformers/models/reformer/modeling_reformer.py:1684 in forward
transformers/models/reformer/modeling_reformer.py:1575 in forward
transformers/models/reformer/modeling_reformer.py:1533 in _init_feed_forward_seed
```
</details>

### 8. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/models/reformer/modeling_reformer.py:750`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_KW: a function call with keyword arguments, e.g. f(x=True):

Unsupported Tensor.item() call with capture_scalar_outputs=False
  Explanation: Dynamo does not support tracing `Tensor.item()` with config.capture_scalar_outputs=False.
  Hint: Set `torch._dynamo.config.capture_scalar_outputs = True` or `export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` to include these operations in the captured graph.

  Developer debug context: call_method TensorVariable() item () {}

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

Stack variable source attribution:
  WithExitFunctionVariable() originated from:
  File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/reformer/modeling_reformer.py", line 1551
                    self._init_attention_seed()
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

<details><summary>User stack</summary>

```
transformers/models/reformer/modeling_reformer.py:2222 in forward
transformers/models/reformer/modeling_reformer.py:2060 in forward
transformers/models/reformer/modeling_reformer.py:1684 in forward
transformers/models/reformer/modeling_reformer.py:559 in forward
transformers/models/reformer/modeling_reformer.py:750 in _hash_vectors
```
</details>

### 9. Data-dependent branching

- **Location:** `transformers/models/reformer/modeling_reformer.py:2073`  (hit ×1)

```
Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow.

      The branch condition involves a tensor computed as follows:
        # File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/reformer/modeling_reformer.py", line 2073, in torch_dynamo_resume_in_forward_at_2060, code: if must_pad_to_match_chunk_length:
        l_must_pad_to_match_chunk_length_: graph input (L_must_pad_to_match_chunk_length_)
    
  Hint: For the common pattern `if tensor_cond: x = transform(x)` (e.g. clamping inf/nan values), consider making the code branchless by always applying the transform. Operations like torch.clamp, torch.nan_to_num, and torch.where are typically no-ops on well-behaved inputs and compile without graph breaks.
  Hint: This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround.
  Hint: Use `torch.cond` to express dynamic control flow.

  Developer debug context: attempted to jump with LazyVariableTracker(realized: NumpyNdarrayVariable())

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html

<details><summary>User stack</summary>

```
transformers/models/reformer/modeling_reformer.py:2222 in forward
transformers/models/reformer/modeling_reformer.py:2073 in forward
```
</details>

### 10. logging.Logger method not supported for non-export cases

- **Location:** `transformers/modeling_utils.py:4685`  (hit ×1)

```
Encountered graph break that we cannot resume from. Compiling up to the previous resumable state, then skipping the rest of the function. Graph break encountered:

logging.Logger method not supported for non-export cases
  Explanation: For non-export cases, logging.Logger methods are only supported if the logger has a source and the method is registered as reorderable.
  Hint: If you do not need this logging side effect, add the exact method being called to `torch._dynamo.config.ignore_logging_functions`. Dynamo will skip the call and return `None`.
  Hint: For example, for `logger.warning_once(...)`, use `torch._dynamo.config.ignore_logging_functions.add(logger.warning_once)`. If `warning_once` is defined on the logger class, add the class method `logging.Logger.warning_once` to ignore this method for all instances of that class.
  Hint: Dynamo does not trace into logging.Logger method bodies, so only the method you call directly (`warning_once`) is checked against the ignore set. Ignoring a method that `warning_once` calls internally has no effect.
  Hint: If you need the log side effect to run, then you can try one of (1) create the logger outside the compiled region and add the method to `torch._dynamo.config.reorderable_logging_functions` (e.g. `torch._dynamo.config.reorderable_logging_functions.add(logger.warning_once)`) so that it runs after the compiled region, as long as it is called without kwargs and its arguments are tensors, constants, or string formatters, (2) `torch._higher_order_ops.print(...)`, (3) wrap the logging call in a custom op (marked as mutable), or (4) preserve the logging contents and move the logging call outside the compiled region.

  Developer debug context: method: <Logger transformers.modeling_utils (WARNING)>.warning_once, args: [ConstantVariable(str: '`loss_type=None` was set in the config but it is unrecognized. Using the default loss: `ForCausalLMLoss`.')], kwargs: {}

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0291.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0291.html

<details><summary>User stack</summary>

```
transformers/models/reformer/modeling_reformer.py:2241 in forward
transformers/modeling_utils.py:4685 in loss_function
```
</details>
