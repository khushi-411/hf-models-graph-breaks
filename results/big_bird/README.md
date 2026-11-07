# big_bird — graph breaks

- **HF id:** `google/bigbird-roberta-base`
- **Class:** `AutoModelForMaskedLM`
- **Config overrides:** `{'num_hidden_layers': 2}`
- **torch:** `2.16.0a0+git9933f27`  |  **transformers:** `5.22.1`
- **Notes:** At SEQ_LEN=32 BigBird falls back from block-sparse to full attention.

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ok | 3 | 2 | 4 |
| train_forward | ok | 3 | 2 | 4 |

## eval_forward

### 1. logging.Logger method not supported for non-export cases

- **Location:** `transformers/models/big_bird/modeling_big_bird.py:1580`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_FUNCTION_EX: a variadic function call, e.g. f(*args):

logging.Logger method not supported for non-export cases
  Explanation: For non-export cases, logging.Logger methods are only supported if the logger has a source and the method is registered as reorderable.
  Hint: If you do not need this logging side effect, add the exact method being called to `torch._dynamo.config.ignore_logging_functions`. Dynamo will skip the call and return `None`.
  Hint: For example, for `logger.warning(...)`, use `torch._dynamo.config.ignore_logging_functions.add(logger.warning)`. If `warning` is defined on the logger class, add the class method `logging.Logger.warning` to ignore this method for all instances of that class.
  Hint: Dynamo does not trace into logging.Logger method bodies, so only the method you call directly (`warning`) is checked against the ignore set. Ignoring a method that `warning` calls internally has no effect.
  Hint: If you need the log side effect to run, then you can try one of (1) create the logger outside the compiled region and add the method to `torch._dynamo.config.reorderable_logging_functions` (e.g. `torch._dynamo.config.reorderable_logging_functions.add(logger.warning)`) so that it runs after the compiled region, as long as it is called without kwargs and its arguments are tensors, constants, or string formatters, (2) `torch._higher_order_ops.print(...)`, (3) wrap the logging call in a custom op (marked as mutable), or (4) preserve the logging contents and move the logging call outside the compiled region.

  Developer debug context: method: <Logger transformers.models.big_bird.modeling_big_bird (WARNING)>.warning, args: [ConstantVariable(str: "Attention type 'block_sparse' is not possible if sequence_length: 32 <= num global tokens: 2 * config.block_size + min. num sliding tokens: 3 * config.block_size + config.num_random_blocks * config.block_size + additional buffer: config.num_random_blocks * config.block_size = 704 with config.block_size = 64, config.num_random_blocks = 3. Changing attention type to 'original_full'...")], kwargs: {}

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0291.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0291.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/big_bird/modeling_big_bird.py:1940 in forward
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/big_bird/modeling_big_bird.py:1580 in forward
```
</details>

### 2. Attempted to use `torch.nn.Parameter()` constructor with Dynamo

- **Location:** `torch/nn/modules/linear.py:108`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

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
transformers/models/big_bird/modeling_big_bird.py:1940 in forward
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/big_bird/modeling_big_bird.py:1591 in forward
transformers/models/big_bird/modeling_big_bird.py:1093 in set_attention_type
transformers/models/big_bird/modeling_big_bird.py:145 in __init__
```
</details>

### 3. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `torch/nn/modules/linear.py:108`  (hit ×1)

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

  Developer debug context: frame skipped: set_attention_type (/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/big_bird/modeling_big_bird.py line 1275)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  LocalGeneratorObjectVariable(enumerate) originated from:
  File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/big_bird/modeling_big_bird.py", line 1284
            for i, layer in enumerate(self.layer):
                            ~~~~~~~~~^^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0264.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/big_bird/modeling_big_bird.py:1940 in forward
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/big_bird/modeling_big_bird.py:1591 in forward
transformers/models/big_bird/modeling_big_bird.py:1093 in set_attention_type
transformers/models/big_bird/modeling_big_bird.py:145 in __init__
```
</details>

### 4. Uninitialized nn.Module

- **Location:** `transformers/models/big_bird/modeling_big_bird.py:134`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_FUNCTION: a call to a regular function, e.g. f(x, y):

Uninitialized nn.Module
  Explanation: Attempted to trace an uninitialized nn.Module of type BigBirdSelfAttention.
  Hint: Your code may result in an error when running in eager. Please double check that your code doesn't contain a similar error when actually running eager/uncompiled. You can do this by removing the `torch.compile` call, or by using `torch.compiler.set_stance("force_eager")`. 
  Hint: Ensure your nn.Module instance has called `super().__init__()`.

  Developer debug context: BigBirdSelfAttention

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0119.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0119.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/big_bird/modeling_big_bird.py:1940 in forward
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/big_bird/modeling_big_bird.py:1591 in forward
transformers/models/big_bird/modeling_big_bird.py:1093 in set_attention_type
transformers/models/big_bird/modeling_big_bird.py:134 in __init__
```
</details>


## train_forward

### 1. logging.Logger method not supported for non-export cases

- **Location:** `transformers/models/big_bird/modeling_big_bird.py:1580`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_FUNCTION_EX: a variadic function call, e.g. f(*args):

logging.Logger method not supported for non-export cases
  Explanation: For non-export cases, logging.Logger methods are only supported if the logger has a source and the method is registered as reorderable.
  Hint: If you do not need this logging side effect, add the exact method being called to `torch._dynamo.config.ignore_logging_functions`. Dynamo will skip the call and return `None`.
  Hint: For example, for `logger.warning(...)`, use `torch._dynamo.config.ignore_logging_functions.add(logger.warning)`. If `warning` is defined on the logger class, add the class method `logging.Logger.warning` to ignore this method for all instances of that class.
  Hint: Dynamo does not trace into logging.Logger method bodies, so only the method you call directly (`warning`) is checked against the ignore set. Ignoring a method that `warning` calls internally has no effect.
  Hint: If you need the log side effect to run, then you can try one of (1) create the logger outside the compiled region and add the method to `torch._dynamo.config.reorderable_logging_functions` (e.g. `torch._dynamo.config.reorderable_logging_functions.add(logger.warning)`) so that it runs after the compiled region, as long as it is called without kwargs and its arguments are tensors, constants, or string formatters, (2) `torch._higher_order_ops.print(...)`, (3) wrap the logging call in a custom op (marked as mutable), or (4) preserve the logging contents and move the logging call outside the compiled region.

  Developer debug context: method: <Logger transformers.models.big_bird.modeling_big_bird (WARNING)>.warning, args: [ConstantVariable(str: "Attention type 'block_sparse' is not possible if sequence_length: 32 <= num global tokens: 2 * config.block_size + min. num sliding tokens: 3 * config.block_size + config.num_random_blocks * config.block_size + additional buffer: config.num_random_blocks * config.block_size = 704 with config.block_size = 64, config.num_random_blocks = 3. Changing attention type to 'original_full'...")], kwargs: {}

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0291.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0291.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/big_bird/modeling_big_bird.py:1940 in forward
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/big_bird/modeling_big_bird.py:1580 in forward
```
</details>

### 2. Attempted to use `torch.nn.Parameter()` constructor with Dynamo

- **Location:** `torch/nn/modules/linear.py:108`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

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
transformers/models/big_bird/modeling_big_bird.py:1940 in forward
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/big_bird/modeling_big_bird.py:1591 in forward
transformers/models/big_bird/modeling_big_bird.py:1093 in set_attention_type
transformers/models/big_bird/modeling_big_bird.py:145 in __init__
```
</details>

### 3. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `torch/nn/modules/linear.py:108`  (hit ×1)

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

  Developer debug context: frame skipped: set_attention_type (/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/big_bird/modeling_big_bird.py line 1275)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  LocalGeneratorObjectVariable(enumerate) originated from:
  File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/big_bird/modeling_big_bird.py", line 1284
            for i, layer in enumerate(self.layer):
                            ~~~~~~~~~^^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0264.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/big_bird/modeling_big_bird.py:1940 in forward
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/big_bird/modeling_big_bird.py:1591 in forward
transformers/models/big_bird/modeling_big_bird.py:1093 in set_attention_type
transformers/models/big_bird/modeling_big_bird.py:145 in __init__
```
</details>

### 4. Uninitialized nn.Module

- **Location:** `transformers/models/big_bird/modeling_big_bird.py:134`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_FUNCTION: a call to a regular function, e.g. f(x, y):

Uninitialized nn.Module
  Explanation: Attempted to trace an uninitialized nn.Module of type BigBirdSelfAttention.
  Hint: Your code may result in an error when running in eager. Please double check that your code doesn't contain a similar error when actually running eager/uncompiled. You can do this by removing the `torch.compile` call, or by using `torch.compiler.set_stance("force_eager")`. 
  Hint: Ensure your nn.Module instance has called `super().__init__()`.

  Developer debug context: BigBirdSelfAttention

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0119.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0119.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/big_bird/modeling_big_bird.py:1940 in forward
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/big_bird/modeling_big_bird.py:1591 in forward
transformers/models/big_bird/modeling_big_bird.py:1093 in set_attention_type
transformers/models/big_bird/modeling_big_bird.py:134 in __init__
```
</details>
