# fsmt — graph breaks

- **HF id:** `facebook/wmt19-en-de`
- **Class:** `AutoModelForSeq2SeqLM`
- **Config overrides:** `{'encoder_layers': 2, 'decoder_layers': 2}`
- **torch:** `2.16.0a0+git57f02f8`  |  **transformers:** `5.24.0.dev0` (git `b7fe69cb19`)

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ok | 3 | 2 | 1 |
| eval_with_cache | ok | 1 | 0 | 0 |
| train_forward | ok | 5 | 4 | 1 |

## eval_forward

### 1. Data-dependent branching

- **Location:** `transformers/models/fsmt/modeling_fsmt.py:266`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_FUNCTION_EX: a variadic function call, e.g. f(*args):

Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow.
  Hint: For the common pattern `if tensor_cond: x = transform(x)` (e.g. clamping inf/nan values), consider making the code branchless by always applying the transform. Operations like torch.clamp, torch.nan_to_num, and torch.where are typically no-ops on well-behaved inputs and compile without graph breaks.
  Hint: Note: Python `or`/`and` between tensor expressions (e.g. `tensor.any() or other_tensor.any()`) triggers implicit bool conversion. Use `torch.logical_or`/`torch.logical_and` or the `|`/`&` operators instead.
  Hint: This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround.
  Hint: Use `torch.cond` to express dynamic control flow.

  Developer debug context: attempted to jump with TensorVariable()

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html

<details><summary>User stack</summary>

```
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/fsmt/modeling_fsmt.py:1004 in forward
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/fsmt/modeling_fsmt.py:858 in forward
transformers/models/fsmt/modeling_fsmt.py:205 in _prepare_fsmt_decoder_inputs
transformers/models/fsmt/modeling_fsmt.py:266 in make_padding_mask
```
</details>


## eval_with_cache

No graph breaks. ✅

## train_forward

### 1. Data-dependent branching

- **Location:** `transformers/models/fsmt/modeling_fsmt.py:266`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_FUNCTION_EX: a variadic function call, e.g. f(*args):

Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow.
  Hint: For the common pattern `if tensor_cond: x = transform(x)` (e.g. clamping inf/nan values), consider making the code branchless by always applying the transform. Operations like torch.clamp, torch.nan_to_num, and torch.where are typically no-ops on well-behaved inputs and compile without graph breaks.
  Hint: Note: Python `or`/`and` between tensor expressions (e.g. `tensor.any() or other_tensor.any()`) triggers implicit bool conversion. Use `torch.logical_or`/`torch.logical_and` or the `|`/`&` operators instead.
  Hint: This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround.
  Hint: Use `torch.cond` to express dynamic control flow.

  Developer debug context: attempted to jump with TensorVariable()

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html

<details><summary>User stack</summary>

```
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/fsmt/modeling_fsmt.py:1004 in forward
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/fsmt/modeling_fsmt.py:858 in forward
transformers/models/fsmt/modeling_fsmt.py:205 in _prepare_fsmt_decoder_inputs
transformers/models/fsmt/modeling_fsmt.py:266 in make_padding_mask
```
</details>
