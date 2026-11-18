# mimi — graph breaks

- **HF id:** `kyutai/mimi`
- **Class:** `AutoModel`
- **Config overrides:** `{'num_hidden_layers': 2}`
- **torch:** `2.16.0a0+git57f02f8`  |  **transformers:** `5.24.0.dev0` (git `b7fe69cb19`)
- **Notes:** Mimi codec (conv + transformer encoder/decoder + RVQ), used by Moshi.

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ok | 25 | 24 | 2 |
| train_forward | ok | 25 | 24 | 2 |

## eval_forward

### 1. Data dependent operator

- **Location:** `torch/nn/functional.py:5952`  (hit ×1)

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
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/mimi/modeling_mimi.py:1516 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/mimi/modeling_mimi.py:1413 in encode
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/mimi/modeling_mimi.py:378 in forward
transformers/models/mimi/modeling_mimi.py:327 in _pad1d
```
</details>

### 2. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `torch/nn/functional.py:5952`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Data dependent operator
  Explanation: Operator `aten._local_scalar_dense.default` has a non-Tensor output whose value is dependent on the data of Tensor inputs.
  Hint: Enable tracing of data-dependent output operators with `torch._dynamo.config.capture_scalar_outputs = True`

  Developer debug context: aten._local_scalar_dense.default

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html

*** While handling this graph break, another graph break occurred: ***

graph break in loop
  Explanation: torch.compile detected a graph break in a for/while loop. Skipping the frame and falling back to eager, as graph breaks in loops are not supported.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: frame skipped: forward (/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/mimi/modeling_mimi.py line 525)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  DictValuesIterator() originated from:
  File "/admin/home/khushi_agrawal/miniforge3/envs/torch-snap-57f02f8/lib/python3.15/site-packages/torch/nn/modules/container.py", line 416
            return iter(self._modules.values())
                   ~~~~^^^^^^^^^^^^^^^^^^^^^^^^
  NullVariable originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/mimi/modeling_mimi.py", line 526
            for layer in self.layers:
                         ^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/mimi/modeling_mimi.py:1516 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/mimi/modeling_mimi.py:1413 in encode
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/mimi/modeling_mimi.py:1265 in forward
transformers/models/mimi/modeling_mimi.py:378 in forward
transformers/models/mimi/modeling_mimi.py:327 in _pad1d
```
</details>


## train_forward

### 1. Data dependent operator

- **Location:** `torch/nn/functional.py:5952`  (hit ×1)

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
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/mimi/modeling_mimi.py:1516 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/mimi/modeling_mimi.py:1413 in encode
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/mimi/modeling_mimi.py:378 in forward
transformers/models/mimi/modeling_mimi.py:327 in _pad1d
```
</details>

### 2. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `torch/nn/functional.py:5952`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Data dependent operator
  Explanation: Operator `aten._local_scalar_dense.default` has a non-Tensor output whose value is dependent on the data of Tensor inputs.
  Hint: Enable tracing of data-dependent output operators with `torch._dynamo.config.capture_scalar_outputs = True`

  Developer debug context: aten._local_scalar_dense.default

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html

*** While handling this graph break, another graph break occurred: ***

graph break in loop
  Explanation: torch.compile detected a graph break in a for/while loop. Skipping the frame and falling back to eager, as graph breaks in loops are not supported.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: frame skipped: forward (/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/mimi/modeling_mimi.py line 525)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  DictValuesIterator() originated from:
  File "/admin/home/khushi_agrawal/miniforge3/envs/torch-snap-57f02f8/lib/python3.15/site-packages/torch/nn/modules/container.py", line 416
            return iter(self._modules.values())
                   ~~~~^^^^^^^^^^^^^^^^^^^^^^^^
  NullVariable originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/mimi/modeling_mimi.py", line 526
            for layer in self.layers:
                         ^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/mimi/modeling_mimi.py:1516 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/mimi/modeling_mimi.py:1413 in encode
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/mimi/modeling_mimi.py:1265 in forward
transformers/models/mimi/modeling_mimi.py:378 in forward
transformers/models/mimi/modeling_mimi.py:327 in _pad1d
```
</details>
