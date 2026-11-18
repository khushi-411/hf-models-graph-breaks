# mimi — graph breaks

- **HF id:** `kyutai/mimi`
- **Class:** `AutoModel`
- **Config overrides:** `{'num_hidden_layers': 2}`
- **torch:** `2.16.0a0+git6b3a750`  |  **transformers:** `5.22.1`
- **Notes:** Mimi codec (conv + transformer encoder/decoder + RVQ), used by Moshi.

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ok | 25 | 24 | 4 |
| train_forward | ok | 25 | 24 | 4 |

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
transformers/utils/generic.py:943 in wrapper
transformers/models/mimi/modeling_mimi.py:1515 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/mimi/modeling_mimi.py:1413 in encode
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/mimi/modeling_mimi.py:377 in forward
transformers/models/mimi/modeling_mimi.py:326 in _pad1d
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
transformers/models/mimi/modeling_mimi.py:1515 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/mimi/modeling_mimi.py:1413 in encode
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
transformers/models/mimi/modeling_mimi.py:1515 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/mimi/modeling_mimi.py:1413 in encode
transformers/utils/output_capturing.py:257 in wrapper
```
</details>

### 4. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

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

  Developer debug context: frame skipped: forward (/home/khushi/anaconda3/envs/torch-dev/lib/python3.14/site-packages/transformers/models/mimi/modeling_mimi.py line 524)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  DictValuesIterator() originated from:
  File "/home/khushi/Documents/pytorch/torch/nn/modules/container.py", line 416
            return iter(self._modules.values())
                   ~~~~^^^^^^^^^^^^^^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/mimi/modeling_mimi.py:1515 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/mimi/modeling_mimi.py:1413 in encode
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/mimi/modeling_mimi.py:1262 in forward
transformers/models/mimi/modeling_mimi.py:377 in forward
transformers/models/mimi/modeling_mimi.py:326 in _pad1d
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
transformers/utils/generic.py:943 in wrapper
transformers/models/mimi/modeling_mimi.py:1515 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/mimi/modeling_mimi.py:1413 in encode
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/mimi/modeling_mimi.py:377 in forward
transformers/models/mimi/modeling_mimi.py:326 in _pad1d
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
transformers/models/mimi/modeling_mimi.py:1515 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/mimi/modeling_mimi.py:1413 in encode
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
transformers/models/mimi/modeling_mimi.py:1515 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/mimi/modeling_mimi.py:1413 in encode
transformers/utils/output_capturing.py:257 in wrapper
```
</details>

### 4. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

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

  Developer debug context: frame skipped: forward (/home/khushi/anaconda3/envs/torch-dev/lib/python3.14/site-packages/transformers/models/mimi/modeling_mimi.py line 524)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  DictValuesIterator() originated from:
  File "/home/khushi/Documents/pytorch/torch/nn/modules/container.py", line 416
            return iter(self._modules.values())
                   ~~~~^^^^^^^^^^^^^^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/mimi/modeling_mimi.py:1515 in forward
transformers/utils/generic.py:943 in wrapper
transformers/models/mimi/modeling_mimi.py:1413 in encode
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/mimi/modeling_mimi.py:1262 in forward
transformers/models/mimi/modeling_mimi.py:377 in forward
transformers/models/mimi/modeling_mimi.py:326 in _pad1d
```
</details>
