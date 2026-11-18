# apertus1p5_vlm — graph breaks

- **HF id:** `swiss-ai/Apertus-v1.5-8B`
- **Class:** `AutoModelForImageTextToText`
- **Config overrides:** `{'text_config.num_hidden_layers': 2, 'text_config.hidden_size': 512, 'text_config.intermediate_size': 1024, 'text_config.num_attention_heads': 8, 'text_config.num_key_value_heads': 8}`
- **torch:** `2.16.0a0+git9933f27`  |  **transformers:** `5.24.0.dev0` (git `84fe6ceb8a`)
- **Notes:** Full Apertus 1.5 with its image tokenizer. Built from Apertus1p5Config defaults (gated checkpoint). Audio input not exercised.

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ok | 7 | 6 | 3 |
| eval_with_cache | ok | 7 | 6 | 3 |
| train_forward | ok | 7 | 6 | 3 |

## eval_forward

### 1. Data dependent operator

- **Location:** `transformers/models/apertus1p5/modeling_apertus1p5.py:815`  (hit ×1)

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
transformers/models/apertus1p5/modeling_apertus1p5.py:1056 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/apertus1p5/modeling_apertus1p5.py:954 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/apertus1p5/modeling_apertus1p5.py:828 in get_image_features
transformers/models/apertus1p5/modeling_apertus1p5.py:815 in get_image_tokens
```
</details>

### 2. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/apertus1p5/modeling_apertus1p5.py:829`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Attempted to read undefined local variable
  Explanation: Could not find a local variable with name `self`
  Hint: Your code may result in an error when running in eager. Please double check that your code doesn't contain a similar error when actually running eager/uncompiled. You can do this by removing the `torch.compile` call, or by using `torch.compiler.set_stance("force_eager")`. 

  Developer debug context: LOAD_FAST self

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0011.html

Stack variable source attribution:
  ListVariable(length=0) originated from:
  File "/home/khushi/Documents/transformers/src/transformers/models/apertus1p5/modeling_apertus1p5.py", line 829
            image_features = tuple(self.get_input_embeddings()(tokens) for tokens in image_tokens)
                             ^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0011.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/apertus1p5/modeling_apertus1p5.py:1056 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/apertus1p5/modeling_apertus1p5.py:954 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/apertus1p5/modeling_apertus1p5.py:829 in get_image_features
```
</details>

### 3. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/utils/import_utils.py:1975`  (hit ×1)

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
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/apertus1p5/modeling_apertus1p5.py:1056 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/apertus1p5/modeling_apertus1p5.py:967 in forward
transformers/models/apertus1p5/modeling_apertus1p5.py:903 in get_placeholder_mask
transformers/utils/import_utils.py:1975 in torch_compilable_check
```
</details>


## eval_with_cache

### 1. Data dependent operator

- **Location:** `transformers/models/apertus1p5/modeling_apertus1p5.py:815`  (hit ×1)

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
transformers/models/apertus1p5/modeling_apertus1p5.py:1056 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/apertus1p5/modeling_apertus1p5.py:954 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/apertus1p5/modeling_apertus1p5.py:828 in get_image_features
transformers/models/apertus1p5/modeling_apertus1p5.py:815 in get_image_tokens
```
</details>

### 2. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/apertus1p5/modeling_apertus1p5.py:829`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Attempted to read undefined local variable
  Explanation: Could not find a local variable with name `self`
  Hint: Your code may result in an error when running in eager. Please double check that your code doesn't contain a similar error when actually running eager/uncompiled. You can do this by removing the `torch.compile` call, or by using `torch.compiler.set_stance("force_eager")`. 

  Developer debug context: LOAD_FAST self

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0011.html

Stack variable source attribution:
  ListVariable(length=0) originated from:
  File "/home/khushi/Documents/transformers/src/transformers/models/apertus1p5/modeling_apertus1p5.py", line 829
            image_features = tuple(self.get_input_embeddings()(tokens) for tokens in image_tokens)
                             ^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0011.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/apertus1p5/modeling_apertus1p5.py:1056 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/apertus1p5/modeling_apertus1p5.py:954 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/apertus1p5/modeling_apertus1p5.py:829 in get_image_features
```
</details>

### 3. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/utils/import_utils.py:1975`  (hit ×1)

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
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/apertus1p5/modeling_apertus1p5.py:1056 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/apertus1p5/modeling_apertus1p5.py:967 in forward
transformers/models/apertus1p5/modeling_apertus1p5.py:903 in get_placeholder_mask
transformers/utils/import_utils.py:1975 in torch_compilable_check
```
</details>


## train_forward

### 1. Data dependent operator

- **Location:** `transformers/models/apertus1p5/modeling_apertus1p5.py:815`  (hit ×1)

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
transformers/models/apertus1p5/modeling_apertus1p5.py:1056 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/apertus1p5/modeling_apertus1p5.py:954 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/apertus1p5/modeling_apertus1p5.py:828 in get_image_features
transformers/models/apertus1p5/modeling_apertus1p5.py:815 in get_image_tokens
```
</details>

### 2. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/apertus1p5/modeling_apertus1p5.py:829`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Attempted to read undefined local variable
  Explanation: Could not find a local variable with name `self`
  Hint: Your code may result in an error when running in eager. Please double check that your code doesn't contain a similar error when actually running eager/uncompiled. You can do this by removing the `torch.compile` call, or by using `torch.compiler.set_stance("force_eager")`. 

  Developer debug context: LOAD_FAST self

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0011.html

Stack variable source attribution:
  ListVariable(length=0) originated from:
  File "/home/khushi/Documents/transformers/src/transformers/models/apertus1p5/modeling_apertus1p5.py", line 829
            image_features = tuple(self.get_input_embeddings()(tokens) for tokens in image_tokens)
                             ^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0011.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/apertus1p5/modeling_apertus1p5.py:1056 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/apertus1p5/modeling_apertus1p5.py:954 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/apertus1p5/modeling_apertus1p5.py:829 in get_image_features
```
</details>

### 3. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/utils/import_utils.py:1975`  (hit ×1)

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
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/apertus1p5/modeling_apertus1p5.py:1056 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/apertus1p5/modeling_apertus1p5.py:967 in forward
transformers/models/apertus1p5/modeling_apertus1p5.py:903 in get_placeholder_mask
transformers/utils/import_utils.py:1975 in torch_compilable_check
```
</details>
