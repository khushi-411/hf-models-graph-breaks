# switch_transformers — graph breaks

- **HF id:** `google/switch-base-8`
- **Class:** `AutoModelForSeq2SeqLM`
- **Config overrides:** `{'num_layers': 2, 'num_decoder_layers': 2, 'num_sparse_encoder_layers': 1, 'num_sparse_decoder_layers': 1, 'encoder_sparse_step': 2, 'decoder_sparse_step': 2}`
- **torch:** `2.16.0a0+git9933f27`  |  **transformers:** `5.22.1`
- **Notes:** eval_with_cache raises "use_cache can only be set to True if ... is used as a decoder" (HF passes use_cache to the encoder).

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ok | 18 | 17 | 4 |
| train_forward | ok | 19 | 18 | 4 |

## eval_forward

### 1. Attempted to use `torch.nn.Parameter()` constructor with Dynamo

- **Location:** `torch/nn/modules/module.py:1005`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_FUNCTION_EX: a variadic function call, e.g. f(*args):

Attempted to use `torch.nn.Parameter()` constructor with Dynamo
  Explanation: Dynamo does not support this
  Hint: Try to construct `torch.nn.Parameter()` outside the compiled region.
  Hint: If this is not possible, turn `graph_break_on_nn_param_ctor` off
  Hint: It may be possible to write Dynamo tracing rules for this code. Please report an issue to PyTorch if you encounter this graph break often and it is causing performance issues.

  Developer debug context: 

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0264.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0264.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/switch_transformers/modeling_switch_transformers.py:998 in forward
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/switch_transformers/modeling_switch_transformers.py:776 in forward
transformers/modeling_layers.py:111 in __call__
transformers/models/switch_transformers/modeling_switch_transformers.py:98 in forward
```
</details>

### 2. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `torch/nn/modules/module.py:1005`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Attempted to use `torch.nn.Parameter()` constructor with Dynamo
  Explanation: Dynamo does not support this
  Hint: Try to construct `torch.nn.Parameter()` outside the compiled region.
  Hint: If this is not possible, turn `graph_break_on_nn_param_ctor` off
  Hint: It may be possible to write Dynamo tracing rules for this code. Please report an issue to PyTorch if you encounter this graph break often and it is causing performance issues.

  Developer debug context: 

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0264.html

*** While handling this graph break, another graph break occurred: ***

graph break in loop
  Explanation: torch.compile detected a graph break in a for/while loop. Skipping the frame and falling back to eager, as graph breaks in loops are not supported.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: frame skipped: forward (/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/switch_transformers/modeling_switch_transformers.py line 704)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  DictValuesIterator() originated from:
  File "/home/khushi/Documents/osc/work/pytorch/torch/nn/modules/container.py", line 416
            return iter(self._modules.values())
                   ~~~~^^^^^^^^^^^^^^^^^^^^^^^^
  NullVariable originated from:
  File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/switch_transformers/modeling_switch_transformers.py", line 775
            for layer_module in self.block:
                                ^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0264.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/switch_transformers/modeling_switch_transformers.py:998 in forward
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/switch_transformers/modeling_switch_transformers.py:776 in forward
transformers/modeling_layers.py:111 in __call__
transformers/models/switch_transformers/modeling_switch_transformers.py:98 in forward
```
</details>

### 3. Dynamic shape operator

- **Location:** `transformers/models/switch_transformers/modeling_switch_transformers.py:173`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Dynamic shape operator
  Explanation: Operator `aten.nonzero.default`'s output shape depends on input Tensor data.
  Hint: Enable tracing of dynamic shape operators with `torch._dynamo.config.capture_dynamic_output_shape_ops = True`

  Developer debug context: aten.nonzero.default

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/switch_transformers/modeling_switch_transformers.py:998 in forward
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/switch_transformers/modeling_switch_transformers.py:776 in forward
transformers/modeling_layers.py:111 in __call__
transformers/models/switch_transformers/modeling_switch_transformers.py:588 in forward
transformers/models/switch_transformers/modeling_switch_transformers.py:227 in forward
transformers/models/switch_transformers/modeling_switch_transformers.py:173 in forward
```
</details>

### 4. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/switch_transformers/modeling_switch_transformers.py:175`  (hit ×1)

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

  Developer debug context: frame skipped: torch_dynamo_resume_in_forward_at_173 (/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/switch_transformers/modeling_switch_transformers.py line 173)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  ListIteratorVariable(length=8, index=1) originated from:
  File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/switch_transformers/modeling_switch_transformers.py", line 174
            for expert_idx in expert_hit:
                              ^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/switch_transformers/modeling_switch_transformers.py:998 in forward
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/switch_transformers/modeling_switch_transformers.py:776 in forward
transformers/modeling_layers.py:111 in __call__
transformers/models/switch_transformers/modeling_switch_transformers.py:588 in forward
transformers/models/switch_transformers/modeling_switch_transformers.py:227 in forward
transformers/models/switch_transformers/modeling_switch_transformers.py:195 in forward
transformers/models/switch_transformers/modeling_switch_transformers.py:175 in forward
```
</details>


## train_forward

### 1. Attempted to use `torch.nn.Parameter()` constructor with Dynamo

- **Location:** `torch/nn/modules/module.py:1005`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_FUNCTION_EX: a variadic function call, e.g. f(*args):

Attempted to use `torch.nn.Parameter()` constructor with Dynamo
  Explanation: Dynamo does not support this
  Hint: Try to construct `torch.nn.Parameter()` outside the compiled region.
  Hint: If this is not possible, turn `graph_break_on_nn_param_ctor` off
  Hint: It may be possible to write Dynamo tracing rules for this code. Please report an issue to PyTorch if you encounter this graph break often and it is causing performance issues.

  Developer debug context: 

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0264.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0264.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/switch_transformers/modeling_switch_transformers.py:998 in forward
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/switch_transformers/modeling_switch_transformers.py:776 in forward
transformers/modeling_layers.py:111 in __call__
transformers/models/switch_transformers/modeling_switch_transformers.py:98 in forward
```
</details>

### 2. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `torch/nn/modules/module.py:1005`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Attempted to use `torch.nn.Parameter()` constructor with Dynamo
  Explanation: Dynamo does not support this
  Hint: Try to construct `torch.nn.Parameter()` outside the compiled region.
  Hint: If this is not possible, turn `graph_break_on_nn_param_ctor` off
  Hint: It may be possible to write Dynamo tracing rules for this code. Please report an issue to PyTorch if you encounter this graph break often and it is causing performance issues.

  Developer debug context: 

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0264.html

*** While handling this graph break, another graph break occurred: ***

graph break in loop
  Explanation: torch.compile detected a graph break in a for/while loop. Skipping the frame and falling back to eager, as graph breaks in loops are not supported.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: frame skipped: forward (/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/switch_transformers/modeling_switch_transformers.py line 704)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  DictValuesIterator() originated from:
  File "/home/khushi/Documents/osc/work/pytorch/torch/nn/modules/container.py", line 416
            return iter(self._modules.values())
                   ~~~~^^^^^^^^^^^^^^^^^^^^^^^^
  NullVariable originated from:
  File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/switch_transformers/modeling_switch_transformers.py", line 775
            for layer_module in self.block:
                                ^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0264.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/switch_transformers/modeling_switch_transformers.py:998 in forward
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/switch_transformers/modeling_switch_transformers.py:776 in forward
transformers/modeling_layers.py:111 in __call__
transformers/models/switch_transformers/modeling_switch_transformers.py:98 in forward
```
</details>

### 3. Dynamic shape operator

- **Location:** `transformers/models/switch_transformers/modeling_switch_transformers.py:173`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Dynamic shape operator
  Explanation: Operator `aten.nonzero.default`'s output shape depends on input Tensor data.
  Hint: Enable tracing of dynamic shape operators with `torch._dynamo.config.capture_dynamic_output_shape_ops = True`

  Developer debug context: aten.nonzero.default

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/switch_transformers/modeling_switch_transformers.py:998 in forward
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/switch_transformers/modeling_switch_transformers.py:776 in forward
transformers/modeling_layers.py:111 in __call__
transformers/models/switch_transformers/modeling_switch_transformers.py:588 in forward
transformers/models/switch_transformers/modeling_switch_transformers.py:227 in forward
transformers/models/switch_transformers/modeling_switch_transformers.py:173 in forward
```
</details>

### 4. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/switch_transformers/modeling_switch_transformers.py:175`  (hit ×1)

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

  Developer debug context: frame skipped: torch_dynamo_resume_in_forward_at_173 (/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/switch_transformers/modeling_switch_transformers.py line 173)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  ListIteratorVariable(length=8, index=1) originated from:
  File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/switch_transformers/modeling_switch_transformers.py", line 174
            for expert_idx in expert_hit:
                              ^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/switch_transformers/modeling_switch_transformers.py:998 in forward
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/switch_transformers/modeling_switch_transformers.py:776 in forward
transformers/modeling_layers.py:111 in __call__
transformers/models/switch_transformers/modeling_switch_transformers.py:588 in forward
transformers/models/switch_transformers/modeling_switch_transformers.py:227 in forward
transformers/models/switch_transformers/modeling_switch_transformers.py:195 in forward
transformers/models/switch_transformers/modeling_switch_transformers.py:175 in forward
```
</details>
