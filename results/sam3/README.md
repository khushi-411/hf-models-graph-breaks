# sam3 — graph breaks

- **HF id:** `facebook/sam3`
- **Class:** `AutoModel`
- **Config overrides:** `{'vision_config.backbone_config.num_hidden_layers': 2, 'text_config.num_hidden_layers': 2, 'geometry_encoder_config.num_layers': 1, 'detr_encoder_config.num_layers': 1, 'detr_decoder_config.num_layers': 1}`
- **torch:** `2.16.0a0+git57f02f8`  |  **transformers:** `5.24.0.dev0` (git `b7fe69cb19`)
- **Notes:** Segment Anything 3 with a text concept prompt (no boxes). Built from Sam3Config defaults (gated hf_id).

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ok | 10 | 9 | 2 |

## eval_forward

### 1. Data dependent operator

- **Location:** `transformers/models/sam3/modeling_sam3.py:1719`  (hit ×1)

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
transformers/models/sam3/modeling_sam3.py:2440 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/sam3/modeling_sam3.py:1827 in forward
transformers/models/sam3/modeling_sam3.py:1742 in _get_rpb_matrix
transformers/pytorch_utils.py:260 in wrapper
transformers/models/sam3/modeling_sam3.py:1719 in _get_coords
```
</details>

### 2. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/sam3/modeling_sam3.py:1719`  (hit ×1)

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

  Developer debug context: frame skipped: forward (/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/sam3/modeling_sam3.py line 1770)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  DictValuesIterator() originated from:
  File "/admin/home/khushi_agrawal/Documents/pytorch/torch/nn/modules/container.py", line 416
            return iter(self._modules.values())
                   ~~~~^^^^^^^^^^^^^^^^^^^^^^^^
  NullVariable originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/sam3/modeling_sam3.py", line 1817
            for layer in self.layers:
                         ^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/models/sam3/modeling_sam3.py:2440 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/sam3/modeling_sam3.py:1827 in forward
transformers/models/sam3/modeling_sam3.py:1742 in _get_rpb_matrix
transformers/pytorch_utils.py:260 in wrapper
transformers/models/sam3/modeling_sam3.py:1719 in _get_coords
```
</details>
