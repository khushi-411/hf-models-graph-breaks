# omniasr_ctc — graph breaks

- **HF id:** `hf-audio/omniasr-ctc-300m-v2`
- **Class:** `AutoModelForCTC`
- **Config overrides:** `{'audio_config.num_hidden_layers': 2}`
- **torch:** `2.16.0a0+git9933f27`  |  **transformers:** `5.24.0.dev0` (git `84fe6ceb8a`)
- **Notes:** Omnilingual ASR, CTC head on a wav2vec2-style encoder. Eager bugs found while setting up (not graph breaks): labels without padding_mask raise AttributeError ('NoneType' has no attribute 'sum'); the documented padding_mask shape (batch, 1, samples) fails in conv1d, only (batch, samples) works.

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ok | 1 | 0 | 0 |
| train_forward | ok | 8 | 7 | 5 |

## eval_forward

No graph breaks. ✅

## train_forward

### 1. Data-dependent branching

- **Location:** `transformers/models/omniasr/modeling_omniasr.py:404`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_FUNCTION_EX: a variadic function call, e.g. f(*args):

Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow.

      The branch condition involves a tensor computed as follows:
        # File "/home/khushi/Documents/transformers/src/transformers/models/omniasr/modeling_omniasr.py", line 403, in forward, code: dropout_probability = torch.rand([])
        dropout_probability = rand([])
    
        # File "/home/khushi/Documents/transformers/src/transformers/models/omniasr/modeling_omniasr.py", line 404, in forward, code: if dropout_probability < self.layerdrop:  # skip the layer
        lt_1 = lt(dropout_probability, 0.1)
    
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
transformers/models/omniasr_ctc/modeling_omniasr_ctc.py:169 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/omniasr_ctc/modeling_omniasr_ctc.py:117 in forward
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
        # File "/home/khushi/Documents/transformers/src/transformers/models/omniasr/modeling_omniasr.py", line 403, in forward, code: dropout_probability = torch.rand([])
        dropout_probability = rand([])
    
        # File "/home/khushi/Documents/transformers/src/transformers/models/omniasr/modeling_omniasr.py", line 404, in forward, code: if dropout_probability < self.layerdrop:  # skip the layer
        lt_1 = lt(dropout_probability, 0.1)
    
  Hint: For the common pattern `if tensor_cond: x = transform(x)` (e.g. clamping inf/nan values), consider making the code branchless by always applying the transform. Operations like torch.clamp, torch.nan_to_num, and torch.where are typically no-ops on well-behaved inputs and compile without graph breaks.
  Hint: This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround.
  Hint: Use `torch.cond` to express dynamic control flow.

  Developer debug context: attempted to jump with TensorVariable()

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html

*** While handling this graph break, another graph break occurred: ***

graph break in loop
  Explanation: torch.compile detected a graph break in a for/while loop. Skipping the frame and falling back to eager, as graph breaks in loops are not supported.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: frame skipped: forward (/home/khushi/Documents/transformers/src/transformers/models/omniasr/modeling_omniasr.py line 367)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  DictValuesIterator() originated from:
  File "/home/khushi/Documents/osc/work/pytorch/torch/nn/modules/container.py", line 416
            return iter(self._modules.values())
                   ~~~~^^^^^^^^^^^^^^^^^^^^^^^^
  NullVariable originated from:
  File "/home/khushi/Documents/transformers/src/transformers/models/omniasr/modeling_omniasr.py", line 399
            for encoder_layer in self.layers:
                                 ^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/models/omniasr_ctc/modeling_omniasr_ctc.py:169 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/omniasr_ctc/modeling_omniasr_ctc.py:117 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/omniasr/modeling_omniasr.py:404 in forward
```
</details>

### 3. Dynamic shape operator

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
transformers/utils/generic.py:1036 in wrapper
transformers/models/omniasr_ctc/modeling_omniasr_ctc.py:182 in forward
transformers/loss/loss_ctc.py:48 in ForCTCLoss
```
</details>

### 4. torch.* op returned non-Tensor

- **Location:** `transformers/loss/loss_ctc.py:53`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_KW: a function call with keyword arguments, e.g. f(x=True):

torch.* op returned non-Tensor
  Explanation: torch.* ops that return a non-Tensor cannot be traced into the Dynamo FX graph output


  Developer debug context: example_value type: _GeneratorContextManager; op: call_function; target: <function flags at 0x75f7ccd8dfe0>

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0208.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0208.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/models/omniasr_ctc/modeling_omniasr_ctc.py:182 in forward
transformers/loss/loss_ctc.py:53 in ForCTCLoss
```
</details>

### 5. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

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
  File "/home/khushi/Documents/transformers/src/transformers/loss/loss_ctc.py", line 53
        with torch.backends.cudnn.flags(enabled=False):
             ~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7151.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/models/omniasr_ctc/modeling_omniasr_ctc.py:182 in forward
transformers/loss/loss_ctc.py:53 in ForCTCLoss
```
</details>
