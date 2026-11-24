# gemma4 — graph breaks

- **HF id:** `google/gemma-4-31B-it`
- **Class:** `AutoModelForImageTextToText`
- **Config overrides:** `{'text_config.num_hidden_layers': 2, 'text_config.layer_types': ['sliding_attention', 'full_attention'], 'text_config.hidden_size': 512, 'text_config.intermediate_size': 1024, 'text_config.num_attention_heads': 4, 'text_config.num_key_value_heads': 2, 'text_config.num_global_key_value_heads': 1, 'text_config.head_dim': 64, 'text_config.global_head_dim': 128, 'vision_config.num_hidden_layers': 2}`
- **torch:** `2.16.0a0+git57f02f8`  |  **transformers:** `5.24.0.dev0` (git `046bdb8cc2`)
- **Notes:** Top-100 #55. Gemma 4 dense: sliding/global layers (global K=V, partial RoPE), vision tower with 2D patch positions + pooling.

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ok | 8 | 7 | 3 |
| eval_with_cache | ok | 8 | 7 | 3 |
| train_forward | ok | 8 | 7 | 3 |

## eval_forward

### 1. Dynamic shape operator

- **Location:** `transformers/models/gemma4/modeling_gemma4.py:2105`  (hit ×1)

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
transformers/models/gemma4/modeling_gemma4.py:2659 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gemma4/modeling_gemma4.py:2393 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/gemma4/modeling_gemma4.py:2260 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/gemma4/modeling_gemma4.py:2105 in forward
```
</details>

### 2. Data dependent operator

- **Location:** `transformers/models/gemma4/modeling_gemma4.py:2271`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

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
transformers/models/gemma4/modeling_gemma4.py:2659 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gemma4/modeling_gemma4.py:2393 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/gemma4/modeling_gemma4.py:2271 in get_image_features
```
</details>

### 3. Dynamic shape operator

- **Location:** `transformers/models/gemma4/modeling_gemma4.py:2404`  (hit ×1)

```
Encountered graph break when attempting to trace BINARY_SUBSCR: a binary subscript, e.g. x[attr]:

Dynamic shape operator
  Explanation: Operator `aten.nonzero.default`'s output shape depends on input Tensor data.
  Hint: Enable tracing of dynamic shape operators with `torch._dynamo.config.capture_dynamic_output_shape_ops = True`

  Developer debug context: aten.nonzero.default

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

Stack variable source attribution:
  LazyVariableTracker(unrealized: <class 'function'>) originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/gemma4/modeling_gemma4.py", line 2403
                torch_compilable_check(
                ^^^^^^^^^^^^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gemma4/modeling_gemma4.py:2659 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gemma4/modeling_gemma4.py:2404 in forward
```
</details>


## eval_with_cache

### 1. Dynamic shape operator

- **Location:** `transformers/models/gemma4/modeling_gemma4.py:2105`  (hit ×1)

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
transformers/models/gemma4/modeling_gemma4.py:2659 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gemma4/modeling_gemma4.py:2393 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/gemma4/modeling_gemma4.py:2260 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/gemma4/modeling_gemma4.py:2105 in forward
```
</details>

### 2. Data dependent operator

- **Location:** `transformers/models/gemma4/modeling_gemma4.py:2271`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

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
transformers/models/gemma4/modeling_gemma4.py:2659 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gemma4/modeling_gemma4.py:2393 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/gemma4/modeling_gemma4.py:2271 in get_image_features
```
</details>

### 3. Dynamic shape operator

- **Location:** `transformers/models/gemma4/modeling_gemma4.py:2404`  (hit ×1)

```
Encountered graph break when attempting to trace BINARY_SUBSCR: a binary subscript, e.g. x[attr]:

Dynamic shape operator
  Explanation: Operator `aten.nonzero.default`'s output shape depends on input Tensor data.
  Hint: Enable tracing of dynamic shape operators with `torch._dynamo.config.capture_dynamic_output_shape_ops = True`

  Developer debug context: aten.nonzero.default

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

Stack variable source attribution:
  LazyVariableTracker(unrealized: <class 'function'>) originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/gemma4/modeling_gemma4.py", line 2403
                torch_compilable_check(
                ^^^^^^^^^^^^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gemma4/modeling_gemma4.py:2659 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gemma4/modeling_gemma4.py:2404 in forward
```
</details>


## train_forward

### 1. Dynamic shape operator

- **Location:** `transformers/models/gemma4/modeling_gemma4.py:2105`  (hit ×1)

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
transformers/models/gemma4/modeling_gemma4.py:2659 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gemma4/modeling_gemma4.py:2393 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/gemma4/modeling_gemma4.py:2260 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/gemma4/modeling_gemma4.py:2105 in forward
```
</details>

### 2. Data dependent operator

- **Location:** `transformers/models/gemma4/modeling_gemma4.py:2271`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

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
transformers/models/gemma4/modeling_gemma4.py:2659 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gemma4/modeling_gemma4.py:2393 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/gemma4/modeling_gemma4.py:2271 in get_image_features
```
</details>

### 3. Dynamic shape operator

- **Location:** `transformers/models/gemma4/modeling_gemma4.py:2404`  (hit ×1)

```
Encountered graph break when attempting to trace BINARY_SUBSCR: a binary subscript, e.g. x[attr]:

Dynamic shape operator
  Explanation: Operator `aten.nonzero.default`'s output shape depends on input Tensor data.
  Hint: Enable tracing of dynamic shape operators with `torch._dynamo.config.capture_dynamic_output_shape_ops = True`

  Developer debug context: aten.nonzero.default

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

Stack variable source attribution:
  LazyVariableTracker(unrealized: <class 'function'>) originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/gemma4/modeling_gemma4.py", line 2403
                torch_compilable_check(
                ^^^^^^^^^^^^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gemma4/modeling_gemma4.py:2659 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gemma4/modeling_gemma4.py:2404 in forward
```
</details>
