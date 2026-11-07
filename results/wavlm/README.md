# wavlm — graph breaks

- **HF id:** `microsoft/wavlm-base-plus`
- **Class:** `AutoModelForCTC`
- **Config overrides:** `{'num_hidden_layers': 2}`
- **torch:** `2.16.0a0+git9933f27`  |  **transformers:** `5.22.1`

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ok | 1 | 0 | 0 |
| train_forward | ok | 12 | 11 | 9 |

## eval_forward

No graph breaks. ✅

## train_forward

### 1. Data-dependent branching

- **Location:** `transformers/models/wavlm/modeling_wavlm.py:1159`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_FUNCTION_EX: a variadic function call, e.g. f(*args):

Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow.

      The branch condition involves a tensor computed as follows:
        # File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/wavlm/modeling_wavlm.py", line 1159, in forward, code: if labels is not None and labels.max() >= self.config.vocab_size:
        ge = ge(max_1, 32)
    
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
transformers/models/wavlm/modeling_wavlm.py:1159 in forward
```
</details>

### 2. Data dependent operator

- **Location:** `transformers/models/wavlm/modeling_wavlm.py:862`  (hit ×1)

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
transformers/utils/generic.py:943 in wrapper
transformers/models/wavlm/modeling_wavlm.py:1162 in forward
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/wavlm/modeling_wavlm.py:1048 in forward
transformers/models/wavlm/modeling_wavlm.py:999 in _mask_hidden_states
transformers/models/wavlm/modeling_wavlm.py:862 in _compute_mask_indices
```
</details>

### 3. requires_grad_() intermediate leaked as output

- **Location:** `transformers/models/wavlm/modeling_wavlm.py:769`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

requires_grad_() intermediate leaked as output
  Explanation: An intermediate tensor with requires_grad_() called on it (or a tensor derived from it) is returned from the compiled region. Graph breaking here to preserve partial acceleration.
  Hint: Call .detach() before returning if you only need values.
  Hint: Consume the gradient inside the compiled function (call backward() and use .grad), or move requires_grad_() outside torch.compile.

  Developer debug context: call_method TensorVariable() requires_grad_

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb8985.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb8985.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/wavlm/modeling_wavlm.py:1162 in forward
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/wavlm/modeling_wavlm.py:769 in forward
```
</details>

### 4. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/wavlm/modeling_wavlm.py:864`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Attempted to read undefined local variable
  Explanation: Could not find a local variable with name `mask_length`
  Hint: Your code may result in an error when running in eager. Please double check that your code doesn't contain a similar error when actually running eager/uncompiled. You can do this by removing the `torch.compile` call, or by using `torch.compiler.set_stance("force_eager")`. 

  Developer debug context: LOAD_FAST mask_length

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0011.html

Stack variable source attribution:
  CellVariable() originated from:
  File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/wavlm/modeling_wavlm.py", line 864
        def compute_num_masked_span(input_length):
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0011.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/wavlm/modeling_wavlm.py:1162 in forward
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/wavlm/modeling_wavlm.py:1048 in forward
transformers/models/wavlm/modeling_wavlm.py:999 in _mask_hidden_states
transformers/models/wavlm/modeling_wavlm.py:864 in _compute_mask_indices
```
</details>

### 5. Data-dependent branching

- **Location:** `transformers/models/wavlm/modeling_wavlm.py:418`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_FUNCTION_EX: a variadic function call, e.g. f(*args):

Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow.

      The branch condition involves a tensor computed as follows:
        # File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/wavlm/modeling_wavlm.py", line 415, in forward, code: dropout_probability = torch.rand([])
        dropout_probability_1 = rand([])
    
        # File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/wavlm/modeling_wavlm.py", line 417, in forward, code: skip_the_layer = self.training and i > 0 and (dropout_probability < self.config.layerdrop)
        skip_the_layer = lt(dropout_probability_1, 0.05)
    
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
transformers/models/wavlm/modeling_wavlm.py:1162 in forward
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/wavlm/modeling_wavlm.py:418 in forward
```
</details>

### 6. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/wavlm/modeling_wavlm.py:418`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow.

      The branch condition involves a tensor computed as follows:
        # File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/wavlm/modeling_wavlm.py", line 415, in forward, code: dropout_probability = torch.rand([])
        dropout_probability_1 = rand([])
    
        # File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/wavlm/modeling_wavlm.py", line 417, in forward, code: skip_the_layer = self.training and i > 0 and (dropout_probability < self.config.layerdrop)
        skip_the_layer = lt(dropout_probability_1, 0.05)
    
  Hint: For the common pattern `if tensor_cond: x = transform(x)` (e.g. clamping inf/nan values), consider making the code branchless by always applying the transform. Operations like torch.clamp, torch.nan_to_num, and torch.where are typically no-ops on well-behaved inputs and compile without graph breaks.
  Hint: This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround.
  Hint: Use `torch.cond` to express dynamic control flow.

  Developer debug context: attempted to jump with TensorVariable()

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html

*** While handling this graph break, another graph break occurred: ***

graph break in loop
  Explanation: torch.compile detected a graph break in a for/while loop. Skipping the frame and falling back to eager, as graph breaks in loops are not supported.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: frame skipped: forward (/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/wavlm/modeling_wavlm.py line 394)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  LocalGeneratorObjectVariable(enumerate) originated from:
  File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/wavlm/modeling_wavlm.py", line 413
            for i, layer in enumerate(self.layers):
                            ~~~~~~~~~^^^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/wavlm/modeling_wavlm.py:1162 in forward
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/wavlm/modeling_wavlm.py:1052 in forward
transformers/models/wavlm/modeling_wavlm.py:418 in forward
```
</details>

### 7. Dynamic shape operator

- **Location:** `transformers/loss/loss_ctc.py:48`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_FUNCTION_EX: a variadic function call, e.g. f(*args):

Dynamic shape operator
  Explanation: Operator `aten.masked_select.default`'s output shape depends on input Tensor data.
  Hint: Enable tracing of dynamic shape operators with `torch._dynamo.config.capture_dynamic_output_shape_ops = True`

  Developer debug context: aten.masked_select.default

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/wavlm/modeling_wavlm.py:1180 in forward
transformers/loss/loss_ctc.py:48 in ForCTCLoss
```
</details>

### 8. torch.* op returned non-Tensor

- **Location:** `transformers/loss/loss_ctc.py:53`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_KW: a function call with keyword arguments, e.g. f(x=True):

torch.* op returned non-Tensor
  Explanation: torch.* ops that return a non-Tensor cannot be traced into the Dynamo FX graph output


  Developer debug context: example_value type: _GeneratorContextManager; op: call_function; target: <function flags at 0x7bf46618dd20>

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0208.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0208.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/wavlm/modeling_wavlm.py:1180 in forward
transformers/loss/loss_ctc.py:53 in ForCTCLoss
```
</details>

### 9. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `python/contextlib.py:203`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

generator function over a skipped function
  Explanation: Cannot trace a generator whose underlying function is skipped by Dynamo (e.g. defined in a skip-listed module), since its body cannot be symbolically traced.
  Hint: This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround.

  Developer debug context: SkipFunctionVariable()

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7151.html

*** While handling this graph break, another graph break occurred: ***

Graph break under GenericContextWrappingVariable
  Explanation: Attempted to graph break in an active context manager(s) that doesn't support graph breaking.
  Hint: Move the offending context manager(s) to outside the compiled region.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: Active generic context managers: [LazyVariableTracker(realized: GenericContextWrappingVariable(_GeneratorContextManager))]

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0066.html

Stack variable source attribution:
  WithExitFunctionVariable() originated from:
  File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/loss/loss_ctc.py", line 53
        with torch.backends.cudnn.flags(enabled=False):
             ~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7151.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/wavlm/modeling_wavlm.py:1180 in forward
transformers/loss/loss_ctc.py:53 in ForCTCLoss
```
</details>
