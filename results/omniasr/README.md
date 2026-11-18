# omniasr — graph breaks

- **HF id:** `hf-audio/omniasr-llm-300m-v2`
- **Class:** `AutoModelForSeq2SeqLM`
- **Config overrides:** `{'audio_config.num_hidden_layers': 2, 'text_config.num_hidden_layers': 2, 'text_config.hidden_size': 1024, 'text_config.head_dim': 128}`
- **torch:** `2.16.0a0+git57f02f8`  |  **transformers:** `5.24.0.dev0` (git `b7fe69cb19`)
- **Notes:** Omnilingual ASR, LLM variant (wav2vec2-style encoder + Llama decoder with audio placeholder tokens).

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ok | 5 | 4 | 1 |
| eval_with_cache | ok | 5 | 4 | 1 |
| train_forward | ok | 9 | 8 | 3 |

## eval_forward

### 1. Dynamic shape operator

- **Location:** `transformers/models/omniasr/modeling_omniasr.py:493`  (hit ×1)

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
transformers/models/omniasr/modeling_omniasr.py:597 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/omniasr/modeling_omniasr.py:522 in forward
transformers/models/omniasr/modeling_omniasr.py:493 in get_placeholder_mask
```
</details>


## eval_with_cache

### 1. Dynamic shape operator

- **Location:** `transformers/models/omniasr/modeling_omniasr.py:493`  (hit ×1)

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
transformers/models/omniasr/modeling_omniasr.py:597 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/omniasr/modeling_omniasr.py:522 in forward
transformers/models/omniasr/modeling_omniasr.py:493 in get_placeholder_mask
```
</details>


## train_forward

### 1. Data-dependent branching

- **Location:** `transformers/models/omniasr/modeling_omniasr.py:404`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_FUNCTION_EX: a variadic function call, e.g. f(*args):

Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow.

      The branch condition involves a tensor computed as follows:
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/omniasr/modeling_omniasr.py", line 403, in forward, code: dropout_probability = torch.rand([])
        dropout_probability = rand([])
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/omniasr/modeling_omniasr.py", line 404, in forward, code: if dropout_probability < self.layerdrop:  # skip the layer
        lt = lt(dropout_probability, 0.1)
    
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
transformers/models/omniasr/modeling_omniasr.py:597 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/omniasr/modeling_omniasr.py:517 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/omniasr/modeling_omniasr.py:464 in get_audio_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/omniasr/modeling_omniasr.py:404 in forward
```
</details>

### 2. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/omniasr/modeling_omniasr.py:404`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow.

      The branch condition involves a tensor computed as follows:
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/omniasr/modeling_omniasr.py", line 403, in forward, code: dropout_probability = torch.rand([])
        dropout_probability = rand([])
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/omniasr/modeling_omniasr.py", line 404, in forward, code: if dropout_probability < self.layerdrop:  # skip the layer
        lt = lt(dropout_probability, 0.1)
    
  Hint: For the common pattern `if tensor_cond: x = transform(x)` (e.g. clamping inf/nan values), consider making the code branchless by always applying the transform. Operations like torch.clamp, torch.nan_to_num, and torch.where are typically no-ops on well-behaved inputs and compile without graph breaks.
  Hint: This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround.
  Hint: Use `torch.cond` to express dynamic control flow.

  Developer debug context: attempted to jump with TensorVariable()

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html

*** While handling this graph break, another graph break occurred: ***

graph break in loop
  Explanation: torch.compile detected a graph break in a for/while loop. Skipping the frame and falling back to eager, as graph breaks in loops are not supported.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: frame skipped: forward (/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/omniasr/modeling_omniasr.py line 367)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  DictValuesIterator() originated from:
  File "/admin/home/khushi_agrawal/miniforge3/envs/torch-snap-57f02f8/lib/python3.15/site-packages/torch/nn/modules/container.py", line 416
            return iter(self._modules.values())
                   ~~~~^^^^^^^^^^^^^^^^^^^^^^^^
  NullVariable originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/omniasr/modeling_omniasr.py", line 399
            for encoder_layer in self.layers:
                                 ^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/models/omniasr/modeling_omniasr.py:597 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/omniasr/modeling_omniasr.py:517 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/omniasr/modeling_omniasr.py:464 in get_audio_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/omniasr/modeling_omniasr.py:404 in forward
```
</details>

### 3. Dynamic shape operator

- **Location:** `transformers/models/omniasr/modeling_omniasr.py:493`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_KW: a function call with keyword arguments, e.g. f(x=True):

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
transformers/models/omniasr/modeling_omniasr.py:597 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/omniasr/modeling_omniasr.py:522 in forward
transformers/models/omniasr/modeling_omniasr.py:493 in get_placeholder_mask
```
</details>
