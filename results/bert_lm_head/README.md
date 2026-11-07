# bert_lm_head — graph breaks

- **HF id:** `google-bert/bert-base-uncased`
- **Class:** `AutoModelForCausalLM`
- **Config overrides:** `{'num_hidden_layers': 2, 'is_decoder': True}`
- **torch:** `2.16.0a0+git9933f27`  |  **transformers:** `5.22.1`
- **Notes:** BertLMHeadModel (BERT used as a causal decoder).

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ok | 1 | 0 | 0 |
| eval_with_cache | ok | 1 | 0 | 0 |
| train_forward | ok | 2 | 1 | 1 |

## eval_forward

No graph breaks. ✅

## eval_with_cache

No graph breaks. ✅

## train_forward

### 1. logging.Logger method not supported for non-export cases

- **Location:** `transformers/modeling_utils.py:4685`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_FUNCTION_EX: a variadic function call, e.g. f(*args):

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
transformers/models/bert/modeling_bert.py:895 in forward
transformers/modeling_utils.py:4685 in loss_function
```
</details>
