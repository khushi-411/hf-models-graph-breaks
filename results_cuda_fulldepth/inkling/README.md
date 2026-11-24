# inkling — graph breaks

- **HF id:** `thinkingmachines/Inkling`
- **Class:** `AutoModelForCausalLM`
- **Config overrides:** `{'num_hidden_layers': 2, 'layer_types': ['hybrid_sliding', 'hybrid'], 'mlp_layer_types': ['dense', 'sparse'], 'hidden_size': 512, 'num_attention_heads': 8, 'num_key_value_heads': 2, 'head_dim': 64, 'swa_num_attention_heads': 8, 'swa_num_key_value_heads': 2, 'swa_head_dim': 64, 'sliding_window_size': 16, 'rel_extent': 16, 'intermediate_size': 1024, 'moe_intermediate_size': 256, 'n_routed_experts': 8, 'num_experts_per_tok': 2, 'n_shared_experts': 1, 'vocab_size': 32000, 'unpadded_vocab_size': 31000}`
- **torch:** `2.16.0a0+gitf29b6d5`  |  **transformers:** `5.24.0.dev0` (git `046bdb8cc2`)
- **Notes:** Text model of the multimodal Inkling checkpoint. Hybrid sliding/full attention with hidden-state-conditioned relative position bias, log-scaled full attention, short convolutions, MoE with shared-expert sink. Layer 0 sliding + dense MLP, layer 1 full + MoE. MTP layers are not loaded by transformers.

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ok | 16 | 15 | 3 |
| eval_with_cache | ok | 30 | 29 | 3 |
| train_forward | ok | 16 | 15 | 3 |

## eval_forward

### 1. Dynamic shape operator

- **Location:** `transformers/models/inkling/modeling_inkling.py:317`  (hit ×1)

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
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling/modeling_inkling.py:787 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling/modeling_inkling.py:722 in forward
transformers/modeling_layers.py:124 in forward_with_checkpointing
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling/modeling_inkling.py:317 in forward
```
</details>

### 2. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/inkling/modeling_inkling.py:317`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Dynamic shape operator
  Explanation: Operator `aten.nonzero.default`'s output shape depends on input Tensor data.
  Hint: Enable tracing of dynamic shape operators with `torch._dynamo.config.capture_dynamic_output_shape_ops = True`

  Developer debug context: aten.nonzero.default

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

*** While handling this graph break, another graph break occurred: ***

graph break in loop
  Explanation: torch.compile detected a graph break in a for/while loop. Skipping the frame and falling back to eager, as graph breaks in loops are not supported.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: frame skipped: forward (/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/inkling/modeling_inkling.py line 676)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  LocalGeneratorObjectVariable(_enumerate) originated from:
  File "/admin/home/khushi_agrawal/Documents/pytorch/torch/_dynamo/polyfills/builtins.py", line 64
        return _enumerate(start)
               ~~~~~~~~~~^^^^^^^
  NullVariable originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/inkling/modeling_inkling.py", line 720
            for i, decoder_layer in enumerate(self.layers):
                                    ~~~~~~~~~^^^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling/modeling_inkling.py:787 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling/modeling_inkling.py:722 in forward
transformers/modeling_layers.py:124 in forward_with_checkpointing
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling/modeling_inkling.py:317 in forward
```
</details>

### 3. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/inkling/modeling_inkling.py:321`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow.

      The branch condition involves a tensor computed as follows:
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/inkling/modeling_inkling.py", line 319, in torch_dynamo_resume_in_forward_at_317, code: for expert_idx in expert_hit:
        l_stack1_: graph input (L_stack1_)
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/inkling/modeling_inkling.py", line 319, in torch_dynamo_resume_in_forward_at_317, code: for expert_idx in expert_hit:
        expert_idx = getitem(l_stack1_, 0)
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/inkling/modeling_inkling.py", line 320, in torch_dynamo_resume_in_forward_at_317, code: expert_idx = expert_idx[0]
        expert_idx_1 = getitem(expert_idx, 0)
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/inkling/modeling_inkling.py", line 321, in torch_dynamo_resume_in_forward_at_317, code: if expert_idx == self.num_experts:
        eq = eq(expert_idx_1, 8)
    
  Hint: For the common pattern `if tensor_cond: x = transform(x)` (e.g. clamping inf/nan values), consider making the code branchless by always applying the transform. Operations like torch.clamp, torch.nan_to_num, and torch.where are typically no-ops on well-behaved inputs and compile without graph breaks.
  Hint: This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround.
  Hint: Use `torch.cond` to express dynamic control flow.

  Developer debug context: attempted to jump with TensorVariable()

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html

*** While handling this graph break, another graph break occurred: ***

graph break in loop
  Explanation: torch.compile detected a graph break in a for/while loop. Skipping the frame and falling back to eager, as graph breaks in loops are not supported.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: frame skipped: torch_dynamo_resume_in_forward_at_317 (/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/inkling/modeling_inkling.py line 317)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  ListIteratorVariable(length=8, index=1) originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/inkling/modeling_inkling.py", line 319
            for expert_idx in expert_hit:
                              ^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html

<details><summary>User stack</summary>

```
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling/modeling_inkling.py:787 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling/modeling_inkling.py:722 in forward
transformers/modeling_layers.py:124 in forward_with_checkpointing
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling/modeling_inkling.py:584 in forward
transformers/models/inkling/modeling_inkling.py:416 in forward
transformers/models/inkling/modeling_inkling.py:321 in forward
```
</details>


## eval_with_cache

### 1. Dynamic shape operator

- **Location:** `transformers/models/inkling/modeling_inkling.py:317`  (hit ×1)

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
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling/modeling_inkling.py:787 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling/modeling_inkling.py:722 in forward
transformers/modeling_layers.py:124 in forward_with_checkpointing
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling/modeling_inkling.py:317 in forward
```
</details>

### 2. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/inkling/modeling_inkling.py:317`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Dynamic shape operator
  Explanation: Operator `aten.nonzero.default`'s output shape depends on input Tensor data.
  Hint: Enable tracing of dynamic shape operators with `torch._dynamo.config.capture_dynamic_output_shape_ops = True`

  Developer debug context: aten.nonzero.default

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

*** While handling this graph break, another graph break occurred: ***

graph break in loop
  Explanation: torch.compile detected a graph break in a for/while loop. Skipping the frame and falling back to eager, as graph breaks in loops are not supported.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: frame skipped: forward (/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/inkling/modeling_inkling.py line 676)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  LocalGeneratorObjectVariable(_enumerate) originated from:
  File "/admin/home/khushi_agrawal/Documents/pytorch/torch/_dynamo/polyfills/builtins.py", line 64
        return _enumerate(start)
               ~~~~~~~~~~^^^^^^^
  NullVariable originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/inkling/modeling_inkling.py", line 720
            for i, decoder_layer in enumerate(self.layers):
                                    ~~~~~~~~~^^^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling/modeling_inkling.py:787 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling/modeling_inkling.py:722 in forward
transformers/modeling_layers.py:124 in forward_with_checkpointing
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling/modeling_inkling.py:317 in forward
```
</details>

### 3. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/inkling/modeling_inkling.py:321`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow.

      The branch condition involves a tensor computed as follows:
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/inkling/modeling_inkling.py", line 319, in torch_dynamo_resume_in_forward_at_317, code: for expert_idx in expert_hit:
        l_stack1_: graph input (L_stack1_)
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/inkling/modeling_inkling.py", line 319, in torch_dynamo_resume_in_forward_at_317, code: for expert_idx in expert_hit:
        expert_idx = getitem(l_stack1_, 0)
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/inkling/modeling_inkling.py", line 320, in torch_dynamo_resume_in_forward_at_317, code: expert_idx = expert_idx[0]
        expert_idx_1 = getitem(expert_idx, 0)
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/inkling/modeling_inkling.py", line 321, in torch_dynamo_resume_in_forward_at_317, code: if expert_idx == self.num_experts:
        eq = eq(expert_idx_1, 8)
    
  Hint: For the common pattern `if tensor_cond: x = transform(x)` (e.g. clamping inf/nan values), consider making the code branchless by always applying the transform. Operations like torch.clamp, torch.nan_to_num, and torch.where are typically no-ops on well-behaved inputs and compile without graph breaks.
  Hint: This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround.
  Hint: Use `torch.cond` to express dynamic control flow.

  Developer debug context: attempted to jump with TensorVariable()

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html

*** While handling this graph break, another graph break occurred: ***

graph break in loop
  Explanation: torch.compile detected a graph break in a for/while loop. Skipping the frame and falling back to eager, as graph breaks in loops are not supported.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: frame skipped: torch_dynamo_resume_in_forward_at_317 (/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/inkling/modeling_inkling.py line 317)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  ListIteratorVariable(length=8, index=1) originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/inkling/modeling_inkling.py", line 319
            for expert_idx in expert_hit:
                              ^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html

<details><summary>User stack</summary>

```
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling/modeling_inkling.py:787 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling/modeling_inkling.py:722 in forward
transformers/modeling_layers.py:124 in forward_with_checkpointing
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling/modeling_inkling.py:584 in forward
transformers/models/inkling/modeling_inkling.py:416 in forward
transformers/models/inkling/modeling_inkling.py:321 in forward
```
</details>


## train_forward

### 1. Dynamic shape operator

- **Location:** `transformers/models/inkling/modeling_inkling.py:317`  (hit ×1)

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
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling/modeling_inkling.py:787 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling/modeling_inkling.py:722 in forward
transformers/modeling_layers.py:124 in forward_with_checkpointing
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling/modeling_inkling.py:317 in forward
```
</details>

### 2. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/inkling/modeling_inkling.py:317`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Dynamic shape operator
  Explanation: Operator `aten.nonzero.default`'s output shape depends on input Tensor data.
  Hint: Enable tracing of dynamic shape operators with `torch._dynamo.config.capture_dynamic_output_shape_ops = True`

  Developer debug context: aten.nonzero.default

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

*** While handling this graph break, another graph break occurred: ***

graph break in loop
  Explanation: torch.compile detected a graph break in a for/while loop. Skipping the frame and falling back to eager, as graph breaks in loops are not supported.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: frame skipped: forward (/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/inkling/modeling_inkling.py line 676)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  LocalGeneratorObjectVariable(_enumerate) originated from:
  File "/admin/home/khushi_agrawal/Documents/pytorch/torch/_dynamo/polyfills/builtins.py", line 64
        return _enumerate(start)
               ~~~~~~~~~~^^^^^^^
  NullVariable originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/inkling/modeling_inkling.py", line 720
            for i, decoder_layer in enumerate(self.layers):
                                    ~~~~~~~~~^^^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling/modeling_inkling.py:787 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling/modeling_inkling.py:722 in forward
transformers/modeling_layers.py:124 in forward_with_checkpointing
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling/modeling_inkling.py:317 in forward
```
</details>

### 3. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/inkling/modeling_inkling.py:321`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow.

      The branch condition involves a tensor computed as follows:
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/inkling/modeling_inkling.py", line 319, in torch_dynamo_resume_in_forward_at_317, code: for expert_idx in expert_hit:
        l_stack1_: graph input (L_stack1_)
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/inkling/modeling_inkling.py", line 319, in torch_dynamo_resume_in_forward_at_317, code: for expert_idx in expert_hit:
        expert_idx = getitem(l_stack1_, 0)
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/inkling/modeling_inkling.py", line 320, in torch_dynamo_resume_in_forward_at_317, code: expert_idx = expert_idx[0]
        expert_idx_1 = getitem(expert_idx, 0)
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/inkling/modeling_inkling.py", line 321, in torch_dynamo_resume_in_forward_at_317, code: if expert_idx == self.num_experts:
        eq = eq(expert_idx_1, 8)
    
  Hint: For the common pattern `if tensor_cond: x = transform(x)` (e.g. clamping inf/nan values), consider making the code branchless by always applying the transform. Operations like torch.clamp, torch.nan_to_num, and torch.where are typically no-ops on well-behaved inputs and compile without graph breaks.
  Hint: This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround.
  Hint: Use `torch.cond` to express dynamic control flow.

  Developer debug context: attempted to jump with TensorVariable()

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html

*** While handling this graph break, another graph break occurred: ***

graph break in loop
  Explanation: torch.compile detected a graph break in a for/while loop. Skipping the frame and falling back to eager, as graph breaks in loops are not supported.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: frame skipped: torch_dynamo_resume_in_forward_at_317 (/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/inkling/modeling_inkling.py line 317)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  ListIteratorVariable(length=8, index=1) originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/inkling/modeling_inkling.py", line 319
            for expert_idx in expert_hit:
                              ^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html

<details><summary>User stack</summary>

```
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling/modeling_inkling.py:787 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling/modeling_inkling.py:722 in forward
transformers/modeling_layers.py:124 in forward_with_checkpointing
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/inkling/modeling_inkling.py:584 in forward
transformers/models/inkling/modeling_inkling.py:416 in forward
transformers/models/inkling/modeling_inkling.py:321 in forward
```
</details>
