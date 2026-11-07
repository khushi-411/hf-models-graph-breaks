# ctrl — graph breaks

- **HF id:** `Salesforce/ctrl`
- **Class:** `AutoModelForCausalLM`
- **Config overrides:** `{'n_layer': 2, 'vocab_size': 32000}`
- **torch:** `2.16.0a0+git9933f27`  |  **transformers:** `5.22.1`

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ok | 3 | 2 | 1 |
| eval_with_cache | ok | 3 | 2 | 1 |
| train_forward | ok | 4 | 3 | 2 |

## eval_forward

### 1. Unsupported python_type() call

- **Location:** `torch/nn/modules/module.py:587`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_FUNCTION_EX: a variadic function call, e.g. f(*args):

Unsupported python_type() call
  Explanation: This VariableTracker does not implement python_type(), which is required for object protocol operations.
  Hint: This is likely to be a Dynamo bug. Please report an issue to PyTorch.

  Developer debug context: GetAttrVariable(TensorVariable(), persistent) does not implement python_type()

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb4841.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb4841.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/ctrl/modeling_ctrl.py:403 in forward
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/ctrl/modeling_ctrl.py:317 in forward
```
</details>


## eval_with_cache

### 1. Unsupported python_type() call

- **Location:** `torch/nn/modules/module.py:587`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_FUNCTION_EX: a variadic function call, e.g. f(*args):

Unsupported python_type() call
  Explanation: This VariableTracker does not implement python_type(), which is required for object protocol operations.
  Hint: This is likely to be a Dynamo bug. Please report an issue to PyTorch.

  Developer debug context: GetAttrVariable(TensorVariable(), persistent) does not implement python_type()

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb4841.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb4841.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/ctrl/modeling_ctrl.py:403 in forward
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/ctrl/modeling_ctrl.py:317 in forward
```
</details>


## train_forward

### 1. Unsupported python_type() call

- **Location:** `torch/nn/modules/module.py:587`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_FUNCTION_EX: a variadic function call, e.g. f(*args):

Unsupported python_type() call
  Explanation: This VariableTracker does not implement python_type(), which is required for object protocol operations.
  Hint: This is likely to be a Dynamo bug. Please report an issue to PyTorch.

  Developer debug context: GetAttrVariable(TensorVariable(), persistent) does not implement python_type()

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb4841.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb4841.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/ctrl/modeling_ctrl.py:403 in forward
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/ctrl/modeling_ctrl.py:317 in forward
```
</details>

### 2. logging.Logger method not supported for non-export cases

- **Location:** `transformers/modeling_utils.py:4685`  (hit ×1)

```
Encountered graph break that we cannot resume from. Compiling up to the previous resumable state, then skipping the rest of the function. Graph break encountered:

logging.Logger method not supported for non-export cases
  Explanation: For non-export cases, logging.Logger methods are only supported if the logger has a source and the method is registered as reorderable.
  Hint: If you do not need this logging side effect, add the exact method being called to `torch._dynamo.config.ignore_logging_functions`. Dynamo will skip the call and return `None`.
  Hint: For example, for `logger.warning_once(...)`, use `torch._dynamo.config.ignore_logging_functions.add(logger.warning_once)`. If `warning_once` is defined on the logger class, add the class method `logging.Logger.warning_once` to ignore this method for all instances of that class.
  Hint: Dynamo does not trace into logging.Logger method bodies, so only the method you call directly (`warning_once`) is checked against the ignore set. Ignoring a method that `warning_once` calls internally has no effect.
  Hint: If you need the log side effect to run, then you can try one of (1) create the logger outside the compiled region and add the method to `torch._dynamo.config.reorderable_logging_functions` (e.g. `torch._dynamo.config.reorderable_logging_functions.add(logger.warning_once)`) so that it runs after the compiled region, as long as it is called without kwargs and its arguments are tensors, constants, or string formatters, (2) `torch._higher_order_ops.print(...)`, (3) wrap the logging call in a custom op (marked as mutable), or (4) preserve the logging contents and move the logging call outside the compiled region.

  Developer debug context: method: <Logger transformers.modeling_utils (WARNING)>.warning_once, args: [ConstantVariable(str: '`loss_type=None` was set in the config but it is unrecognized. Using the default loss: `ForCausalLMLoss`.')], kwargs: {}

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0291.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0291.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:943 in wrapper
transformers/models/ctrl/modeling_ctrl.py:420 in forward
transformers/modeling_utils.py:4685 in loss_function
```
</details>
