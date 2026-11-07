# m2m_100 — graph breaks

- **HF id:** `facebook/nllb-200-distilled-600M`
- **Class:** `AutoModelForSeq2SeqLM`
- **Config overrides:** `{'encoder_layers': 2, 'decoder_layers': 2, 'vocab_size': 32000}`
- **torch:** `2.16.0a0+git9933f27`  |  **transformers:** `5.22.1`
- **Notes:** NLLB checkpoint, M2M100 architecture.

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ok | 1 | 0 | 0 |
| eval_with_cache | ok | 1 | 0 | 0 |
| train_forward | ok | 10 | 9 | 4 |

## eval_forward

No graph breaks. ✅

## eval_with_cache

No graph breaks. ✅

## train_forward

### 1. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/m2m_100/modeling_m2m_100.py:578`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow.

      The branch condition involves a tensor computed as follows:
        # File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/m2m_100/modeling_m2m_100.py", line 577, in forward, code: dropout_probability = torch.rand([])
        dropout_probability = rand([])
    
        # File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/m2m_100/modeling_m2m_100.py", line 578, in forward, code: if dropout_probability < self.layerdrop:
        lt = lt(dropout_probability, 0)
    
  Hint: For the common pattern `if tensor_cond: x = transform(x)` (e.g. clamping inf/nan values), consider making the code branchless by always applying the transform. Operations like torch.clamp, torch.nan_to_num, and torch.where are typically no-ops on well-behaved inputs and compile without graph breaks.
  Hint: This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround.
  Hint: Use `torch.cond` to express dynamic control flow.

  Developer debug context: attempted to jump with TensorVariable()

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html

<details><summary>User stack</summary>

```
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/m2m_100/modeling_m2m_100.py:887 in forward
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/m2m_100/modeling_m2m_100.py:778 in forward
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/m2m_100/modeling_m2m_100.py:578 in forward
```
</details>

### 2. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/m2m_100/modeling_m2m_100.py:578`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow.

      The branch condition involves a tensor computed as follows:
        # File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/m2m_100/modeling_m2m_100.py", line 577, in forward, code: dropout_probability = torch.rand([])
        dropout_probability = rand([])
    
        # File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/m2m_100/modeling_m2m_100.py", line 578, in forward, code: if dropout_probability < self.layerdrop:
        lt = lt(dropout_probability, 0)
    
  Hint: For the common pattern `if tensor_cond: x = transform(x)` (e.g. clamping inf/nan values), consider making the code branchless by always applying the transform. Operations like torch.clamp, torch.nan_to_num, and torch.where are typically no-ops on well-behaved inputs and compile without graph breaks.
  Hint: This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround.
  Hint: Use `torch.cond` to express dynamic control flow.

  Developer debug context: attempted to jump with TensorVariable()

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html

*** While handling this graph break, another graph break occurred: ***

graph break in loop
  Explanation: torch.compile detected a graph break in a for/while loop. Skipping the frame and falling back to eager, as graph breaks in loops are not supported.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: frame skipped: forward (/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/m2m_100/modeling_m2m_100.py line 545)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  LocalGeneratorObjectVariable(enumerate) originated from:
  File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/m2m_100/modeling_m2m_100.py", line 573
            for idx, encoder_layer in enumerate(self.layers):
                                      ~~~~~~~~~^^^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html

<details><summary>User stack</summary>

```
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/m2m_100/modeling_m2m_100.py:887 in forward
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/m2m_100/modeling_m2m_100.py:778 in forward
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/m2m_100/modeling_m2m_100.py:578 in forward
```
</details>

### 3. Data-dependent branching

- **Location:** `transformers/models/m2m_100/modeling_m2m_100.py:696`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_FUNCTION_EX: a variadic function call, e.g. f(*args):

Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow.

      The branch condition involves a tensor computed as follows:
        # File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/m2m_100/modeling_m2m_100.py", line 695, in forward, code: dropout_probability = torch.rand([])
        dropout_probability = rand([])
    
        # File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/m2m_100/modeling_m2m_100.py", line 696, in forward, code: if dropout_probability < self.layerdrop:
        lt = lt(dropout_probability, 0)
    
  Hint: For the common pattern `if tensor_cond: x = transform(x)` (e.g. clamping inf/nan values), consider making the code branchless by always applying the transform. Operations like torch.clamp, torch.nan_to_num, and torch.where are typically no-ops on well-behaved inputs and compile without graph breaks.
  Hint: This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround.
  Hint: Use `torch.cond` to express dynamic control flow.

  Developer debug context: attempted to jump with TensorVariable()

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html

<details><summary>User stack</summary>

```
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/m2m_100/modeling_m2m_100.py:887 in forward
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/m2m_100/modeling_m2m_100.py:792 in forward
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/m2m_100/modeling_m2m_100.py:696 in forward
```
</details>

### 4. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/m2m_100/modeling_m2m_100.py:696`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow.

      The branch condition involves a tensor computed as follows:
        # File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/m2m_100/modeling_m2m_100.py", line 695, in forward, code: dropout_probability = torch.rand([])
        dropout_probability = rand([])
    
        # File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/m2m_100/modeling_m2m_100.py", line 696, in forward, code: if dropout_probability < self.layerdrop:
        lt = lt(dropout_probability, 0)
    
  Hint: For the common pattern `if tensor_cond: x = transform(x)` (e.g. clamping inf/nan values), consider making the code branchless by always applying the transform. Operations like torch.clamp, torch.nan_to_num, and torch.where are typically no-ops on well-behaved inputs and compile without graph breaks.
  Hint: This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround.
  Hint: Use `torch.cond` to express dynamic control flow.

  Developer debug context: attempted to jump with TensorVariable()

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html

*** While handling this graph break, another graph break occurred: ***

graph break in loop
  Explanation: torch.compile detected a graph break in a for/while loop. Skipping the frame and falling back to eager, as graph breaks in loops are not supported.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: frame skipped: forward (/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/m2m_100/modeling_m2m_100.py line 634)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  LocalGeneratorObjectVariable(enumerate) originated from:
  File "/home/khushi/anaconda3/envs/pt2dev/lib/python3.15/site-packages/transformers/models/m2m_100/modeling_m2m_100.py", line 692
            for idx, decoder_layer in enumerate(self.layers):
                                      ~~~~~~~~~^^^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html

<details><summary>User stack</summary>

```
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/m2m_100/modeling_m2m_100.py:887 in forward
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/m2m_100/modeling_m2m_100.py:792 in forward
transformers/utils/output_capturing.py:287 in wrapper
transformers/models/m2m_100/modeling_m2m_100.py:696 in forward
```
</details>
