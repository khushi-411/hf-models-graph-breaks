# hy_v4_vl — graph breaks

- **HF id:** `tencent/Hy4`
- **Class:** `AutoModelForImageTextToText`
- **Config overrides:** `{'text_config.num_hidden_layers': 3, 'text_config.layer_types': ['indexed_attention', 'indexed_attention', 'indexed_attention'], 'text_config.mlp_layer_types': ['dense', 'sparse', 'sparse'], 'text_config.indexer_types': ['full', 'full', 'shared'], 'text_config.hidden_size': 512, 'text_config.intermediate_size': 1024, 'text_config.moe_intermediate_size': 128, 'text_config.num_attention_heads': 8, 'text_config.num_key_value_heads': 8, 'text_config.q_lora_rank': 128, 'text_config.kv_lora_rank': 64, 'text_config.qk_nope_head_dim': 32, 'text_config.qk_rope_head_dim': 32, 'text_config.qk_head_dim': 64, 'text_config.head_dim': 32, 'text_config.v_head_dim': 32, 'text_config.index_topk': 16, 'text_config.index_head_dim': 64, 'text_config.index_n_heads': 4, 'text_config.n_routed_experts': 8, 'text_config.num_experts_per_tok': 2, 'vision_config.num_hidden_layers': 2, 'vision_config.out_hidden_size': 512}`
- **torch:** `2.16.0a0+git57f02f8`  |  **transformers:** `5.24.0.dev0` (git `b7fe69cb19`)
- **Notes:** Built from HYV4VLConfig defaults (no public hy_v4_vl checkpoint). MLA + indexed attention with shared indexers, hyper-connections, MoE.

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ok | 23 | 22 | 13 |
| eval_with_cache | ok | 23 | 22 | 13 |
| train_forward | ok | 23 | 22 | 13 |

## eval_forward

### 1. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:795`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_FUNCTION_EX: a variadic function call, e.g. f(*args):

Unsupported Tensor.item() call with capture_scalar_outputs=False
  Explanation: Dynamo does not support tracing `Tensor.item()` with config.capture_scalar_outputs=False.
  Hint: Set `torch._dynamo.config.capture_scalar_outputs = True` or `export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` to include these operations in the captured graph.

  Developer debug context: call_method TensorVariable() item () {}

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

<details><summary>User stack</summary>

```
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1689 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1546 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1419 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:817 in forward
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:795 in interpolate_pos_encoding
```
</details>

### 2. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:795`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Unsupported Tensor.item() call with capture_scalar_outputs=False
  Explanation: Dynamo does not support tracing `Tensor.item()` with config.capture_scalar_outputs=False.
  Hint: Set `torch._dynamo.config.capture_scalar_outputs = True` or `export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` to include these operations in the captured graph.

  Developer debug context: call_method TensorVariable() item () {}

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

*** While handling this graph break, another graph break occurred: ***

graph break in loop
  Explanation: torch.compile detected a graph break in a for/while loop. Skipping the frame and falling back to eager, as graph breaks in loops are not supported.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: frame skipped: forward (/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/hy_v4_vl/modeling_hy_v4_vl.py line 805)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  ListIteratorVariable(length=2, index=1) originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/hy_v4_vl/modeling_hy_v4_vl.py", line 816
            for t, h, w in grid_thw:
                           ^^^^^^^^
  CallMethodVariable(ListVariable(length=0), append) originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/hy_v4_vl/modeling_hy_v4_vl.py", line 817
                position_embeddings_list.append(self.interpolate_pos_encoding(embeddings, h, w).squeeze(0).repeat(t, 1))
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

<details><summary>User stack</summary>

```
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1689 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1546 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1419 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1043 in forward
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:817 in forward
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:795 in interpolate_pos_encoding
```
</details>

### 3. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:796`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Unsupported Tensor.item() call with capture_scalar_outputs=False
  Explanation: Dynamo does not support tracing `Tensor.item()` with config.capture_scalar_outputs=False.
  Hint: Set `torch._dynamo.config.capture_scalar_outputs = True` or `export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` to include these operations in the captured graph.

  Developer debug context: call_method TensorVariable() item () {}

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

<details><summary>User stack</summary>

```
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1689 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1546 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1419 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1043 in forward
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:817 in forward
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:796 in interpolate_pos_encoding
```
</details>

### 4. Dynamic shape operator

- **Location:** `transformers/vision_utils.py:64`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_KW: a function call with keyword arguments, e.g. f(x=True):

Dynamic shape operator
  Explanation: Operator `aten.repeat_interleave.Tensor`'s output shape depends on input Tensor data.
  Hint: Enable tracing of dynamic shape operators with `torch._dynamo.config.capture_dynamic_output_shape_ops = True`

  Developer debug context: aten.repeat_interleave.Tensor

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

<details><summary>User stack</summary>

```
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1689 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1546 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1419 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1044 in forward
transformers/vision_utils.py:76 in get_vision_attention_seqlens
transformers/vision_utils.py:64 in get_vision_cu_seqlens
```
</details>

### 5. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:927`  (hit ×1)

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

  Developer debug context: frame skipped: torch_dynamo_resume_in_forward_at_1044 (/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/hy_v4_vl/modeling_hy_v4_vl.py line 1044)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  DictValuesIterator() originated from:
  File "/admin/home/khushi_agrawal/miniforge3/envs/torch-snap-57f02f8/lib/python3.15/site-packages/torch/nn/modules/container.py", line 416
            return iter(self._modules.values())
                   ~~~~^^^^^^^^^^^^^^^^^^^^^^^^
  NullVariable originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/hy_v4_vl/modeling_hy_v4_vl.py", line 1046
            for layer in self.layers:
                         ^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html

<details><summary>User stack</summary>

```
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1689 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1546 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1419 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1047 in forward
transformers/modeling_layers.py:122 in forward_with_checkpointing
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:927 in forward
```
</details>

### 6. Data dependent operator

- **Location:** `transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:927`  (hit ×1)

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
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1689 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1546 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1419 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1047 in forward
transformers/modeling_layers.py:122 in forward_with_checkpointing
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:927 in forward
```
</details>

### 7. Data dependent operator

- **Location:** `transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1423`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Data dependent operator
  Explanation: Operator `aten._local_scalar_dense.default` has a non-Tensor output whose value is dependent on the data of Tensor inputs.
  Hint: Enable tracing of data-dependent output operators with `torch._dynamo.config.capture_scalar_outputs = True`

  Developer debug context: aten._local_scalar_dense.default

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html

Stack variable source attribution:
  LazyVariableTracker(unrealized: <class 'function'>) originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/hy_v4_vl/modeling_hy_v4_vl.py", line 1423
            vision_outputs.pooler_output = torch.split(vision_outputs.pooler_output.squeeze(0), split_sizes.tolist())
                                           ^^^^^^^^^^^
  TensorVariable() originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/hy_v4_vl/modeling_hy_v4_vl.py", line 1423
            vision_outputs.pooler_output = torch.split(vision_outputs.pooler_output.squeeze(0), split_sizes.tolist())
                                                       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1689 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1546 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1423 in get_image_features
```
</details>

### 8. Unsupported Tensor.item() call with capture_scalar_outputs=False

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
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1689 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1559 in forward
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1453 in get_placeholder_mask
transformers/utils/import_utils.py:1975 in torch_compilable_check
```
</details>

### 9. Dynamic shape operator

- **Location:** `transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1360`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_KW: a function call with keyword arguments, e.g. f(x=True):

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
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1689 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1574 in forward
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1489 in compute_3d_position_ids
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1360 in get_rope_index
```
</details>

### 10. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1360`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Dynamic shape operator
  Explanation: Operator `aten.nonzero.default`'s output shape depends on input Tensor data.
  Hint: Enable tracing of dynamic shape operators with `torch._dynamo.config.capture_dynamic_output_shape_ops = True`

  Developer debug context: aten.nonzero.default

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

*** While handling this graph break, another graph break occurred: ***

graph break in loop
  Explanation: torch.compile detected a graph break in a for/while loop. Skipping the frame and falling back to eager, as graph breaks in loops are not supported.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: frame skipped: get_rope_index (/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/hy_v4_vl/modeling_hy_v4_vl.py line 1295)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  LocalGeneratorObjectVariable(enumerate) originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/hy_v4_vl/modeling_hy_v4_vl.py", line 1357
            for batch_idx, current_input_ids in enumerate(input_ids):
                                                ~~~~~~~~~^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1689 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1574 in forward
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1489 in compute_3d_position_ids
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1360 in get_rope_index
```
</details>

### 11. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1279`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

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
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1689 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1574 in forward
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1489 in compute_3d_position_ids
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1383 in get_rope_index
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1279 in get_vision_position_ids
```
</details>

### 12. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1280`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Unsupported Tensor.item() call with capture_scalar_outputs=False
  Explanation: Dynamo does not support tracing `Tensor.item()` with config.capture_scalar_outputs=False.
  Hint: Set `torch._dynamo.config.capture_scalar_outputs = True` or `export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` to include these operations in the captured graph.

  Developer debug context: call_method TensorVariable() item () {}

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

Stack variable source attribution:
  ConstantVariable(int: 1) originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/hy_v4_vl/modeling_hy_v4_vl.py", line 1279
                grid_thw[0].item() // temp_merge_size,
                ~~~~~~~~~~~~~~~~~~~^^~~~~~~~~~~~~~~~~
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1689 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1574 in forward
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1489 in compute_3d_position_ids
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1383 in get_rope_index
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1280 in get_vision_position_ids
```
</details>

### 13. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1281`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Unsupported Tensor.item() call with capture_scalar_outputs=False
  Explanation: Dynamo does not support tracing `Tensor.item()` with config.capture_scalar_outputs=False.
  Hint: Set `torch._dynamo.config.capture_scalar_outputs = True` or `export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` to include these operations in the captured graph.

  Developer debug context: call_method TensorVariable() item () {}

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

Stack variable source attribution:
  LazyVariableTracker(unrealized: <class 'int'>) originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/hy_v4_vl/modeling_hy_v4_vl.py", line 1280
                grid_thw[1].item() // spatial_merge_size,
  ConstantVariable(int: 4) originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/hy_v4_vl/modeling_hy_v4_vl.py", line 1280
                grid_thw[1].item() // spatial_merge_size,
                ~~~~~~~~~~~~~~~~~~~^^~~~~~~~~~~~~~~~~~~~
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1689 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1574 in forward
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1489 in compute_3d_position_ids
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1383 in get_rope_index
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1281 in get_vision_position_ids
```
</details>


## eval_with_cache

### 1. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:795`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_FUNCTION_EX: a variadic function call, e.g. f(*args):

Unsupported Tensor.item() call with capture_scalar_outputs=False
  Explanation: Dynamo does not support tracing `Tensor.item()` with config.capture_scalar_outputs=False.
  Hint: Set `torch._dynamo.config.capture_scalar_outputs = True` or `export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` to include these operations in the captured graph.

  Developer debug context: call_method TensorVariable() item () {}

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

<details><summary>User stack</summary>

```
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1689 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1546 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1419 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:817 in forward
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:795 in interpolate_pos_encoding
```
</details>

### 2. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:795`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Unsupported Tensor.item() call with capture_scalar_outputs=False
  Explanation: Dynamo does not support tracing `Tensor.item()` with config.capture_scalar_outputs=False.
  Hint: Set `torch._dynamo.config.capture_scalar_outputs = True` or `export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` to include these operations in the captured graph.

  Developer debug context: call_method TensorVariable() item () {}

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

*** While handling this graph break, another graph break occurred: ***

graph break in loop
  Explanation: torch.compile detected a graph break in a for/while loop. Skipping the frame and falling back to eager, as graph breaks in loops are not supported.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: frame skipped: forward (/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/hy_v4_vl/modeling_hy_v4_vl.py line 805)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  ListIteratorVariable(length=2, index=1) originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/hy_v4_vl/modeling_hy_v4_vl.py", line 816
            for t, h, w in grid_thw:
                           ^^^^^^^^
  CallMethodVariable(ListVariable(length=0), append) originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/hy_v4_vl/modeling_hy_v4_vl.py", line 817
                position_embeddings_list.append(self.interpolate_pos_encoding(embeddings, h, w).squeeze(0).repeat(t, 1))
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

<details><summary>User stack</summary>

```
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1689 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1546 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1419 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1043 in forward
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:817 in forward
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:795 in interpolate_pos_encoding
```
</details>

### 3. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:796`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Unsupported Tensor.item() call with capture_scalar_outputs=False
  Explanation: Dynamo does not support tracing `Tensor.item()` with config.capture_scalar_outputs=False.
  Hint: Set `torch._dynamo.config.capture_scalar_outputs = True` or `export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` to include these operations in the captured graph.

  Developer debug context: call_method TensorVariable() item () {}

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

<details><summary>User stack</summary>

```
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1689 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1546 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1419 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1043 in forward
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:817 in forward
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:796 in interpolate_pos_encoding
```
</details>

### 4. Dynamic shape operator

- **Location:** `transformers/vision_utils.py:64`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_KW: a function call with keyword arguments, e.g. f(x=True):

Dynamic shape operator
  Explanation: Operator `aten.repeat_interleave.Tensor`'s output shape depends on input Tensor data.
  Hint: Enable tracing of dynamic shape operators with `torch._dynamo.config.capture_dynamic_output_shape_ops = True`

  Developer debug context: aten.repeat_interleave.Tensor

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

<details><summary>User stack</summary>

```
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1689 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1546 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1419 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1044 in forward
transformers/vision_utils.py:76 in get_vision_attention_seqlens
transformers/vision_utils.py:64 in get_vision_cu_seqlens
```
</details>

### 5. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:927`  (hit ×1)

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

  Developer debug context: frame skipped: torch_dynamo_resume_in_forward_at_1044 (/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/hy_v4_vl/modeling_hy_v4_vl.py line 1044)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  DictValuesIterator() originated from:
  File "/admin/home/khushi_agrawal/miniforge3/envs/torch-snap-57f02f8/lib/python3.15/site-packages/torch/nn/modules/container.py", line 416
            return iter(self._modules.values())
                   ~~~~^^^^^^^^^^^^^^^^^^^^^^^^
  NullVariable originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/hy_v4_vl/modeling_hy_v4_vl.py", line 1046
            for layer in self.layers:
                         ^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html

<details><summary>User stack</summary>

```
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1689 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1546 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1419 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1047 in forward
transformers/modeling_layers.py:122 in forward_with_checkpointing
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:927 in forward
```
</details>

### 6. Data dependent operator

- **Location:** `transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:927`  (hit ×1)

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
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1689 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1546 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1419 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1047 in forward
transformers/modeling_layers.py:122 in forward_with_checkpointing
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:927 in forward
```
</details>

### 7. Data dependent operator

- **Location:** `transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1423`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Data dependent operator
  Explanation: Operator `aten._local_scalar_dense.default` has a non-Tensor output whose value is dependent on the data of Tensor inputs.
  Hint: Enable tracing of data-dependent output operators with `torch._dynamo.config.capture_scalar_outputs = True`

  Developer debug context: aten._local_scalar_dense.default

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html

Stack variable source attribution:
  LazyVariableTracker(unrealized: <class 'function'>) originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/hy_v4_vl/modeling_hy_v4_vl.py", line 1423
            vision_outputs.pooler_output = torch.split(vision_outputs.pooler_output.squeeze(0), split_sizes.tolist())
                                           ^^^^^^^^^^^
  TensorVariable() originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/hy_v4_vl/modeling_hy_v4_vl.py", line 1423
            vision_outputs.pooler_output = torch.split(vision_outputs.pooler_output.squeeze(0), split_sizes.tolist())
                                                       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1689 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1546 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1423 in get_image_features
```
</details>

### 8. Unsupported Tensor.item() call with capture_scalar_outputs=False

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
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1689 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1559 in forward
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1453 in get_placeholder_mask
transformers/utils/import_utils.py:1975 in torch_compilable_check
```
</details>

### 9. Dynamic shape operator

- **Location:** `transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1360`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_KW: a function call with keyword arguments, e.g. f(x=True):

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
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1689 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1574 in forward
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1489 in compute_3d_position_ids
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1360 in get_rope_index
```
</details>

### 10. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1360`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Dynamic shape operator
  Explanation: Operator `aten.nonzero.default`'s output shape depends on input Tensor data.
  Hint: Enable tracing of dynamic shape operators with `torch._dynamo.config.capture_dynamic_output_shape_ops = True`

  Developer debug context: aten.nonzero.default

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

*** While handling this graph break, another graph break occurred: ***

graph break in loop
  Explanation: torch.compile detected a graph break in a for/while loop. Skipping the frame and falling back to eager, as graph breaks in loops are not supported.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: frame skipped: get_rope_index (/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/hy_v4_vl/modeling_hy_v4_vl.py line 1295)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  LocalGeneratorObjectVariable(enumerate) originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/hy_v4_vl/modeling_hy_v4_vl.py", line 1357
            for batch_idx, current_input_ids in enumerate(input_ids):
                                                ~~~~~~~~~^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1689 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1574 in forward
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1489 in compute_3d_position_ids
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1360 in get_rope_index
```
</details>

### 11. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1279`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

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
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1689 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1574 in forward
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1489 in compute_3d_position_ids
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1383 in get_rope_index
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1279 in get_vision_position_ids
```
</details>

### 12. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1280`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Unsupported Tensor.item() call with capture_scalar_outputs=False
  Explanation: Dynamo does not support tracing `Tensor.item()` with config.capture_scalar_outputs=False.
  Hint: Set `torch._dynamo.config.capture_scalar_outputs = True` or `export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` to include these operations in the captured graph.

  Developer debug context: call_method TensorVariable() item () {}

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

Stack variable source attribution:
  ConstantVariable(int: 1) originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/hy_v4_vl/modeling_hy_v4_vl.py", line 1279
                grid_thw[0].item() // temp_merge_size,
                ~~~~~~~~~~~~~~~~~~~^^~~~~~~~~~~~~~~~~
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1689 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1574 in forward
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1489 in compute_3d_position_ids
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1383 in get_rope_index
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1280 in get_vision_position_ids
```
</details>

### 13. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1281`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Unsupported Tensor.item() call with capture_scalar_outputs=False
  Explanation: Dynamo does not support tracing `Tensor.item()` with config.capture_scalar_outputs=False.
  Hint: Set `torch._dynamo.config.capture_scalar_outputs = True` or `export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` to include these operations in the captured graph.

  Developer debug context: call_method TensorVariable() item () {}

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

Stack variable source attribution:
  LazyVariableTracker(unrealized: <class 'int'>) originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/hy_v4_vl/modeling_hy_v4_vl.py", line 1280
                grid_thw[1].item() // spatial_merge_size,
  ConstantVariable(int: 4) originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/hy_v4_vl/modeling_hy_v4_vl.py", line 1280
                grid_thw[1].item() // spatial_merge_size,
                ~~~~~~~~~~~~~~~~~~~^^~~~~~~~~~~~~~~~~~~~
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1689 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1574 in forward
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1489 in compute_3d_position_ids
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1383 in get_rope_index
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1281 in get_vision_position_ids
```
</details>


## train_forward

### 1. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:795`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_FUNCTION_EX: a variadic function call, e.g. f(*args):

Unsupported Tensor.item() call with capture_scalar_outputs=False
  Explanation: Dynamo does not support tracing `Tensor.item()` with config.capture_scalar_outputs=False.
  Hint: Set `torch._dynamo.config.capture_scalar_outputs = True` or `export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` to include these operations in the captured graph.

  Developer debug context: call_method TensorVariable() item () {}

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

<details><summary>User stack</summary>

```
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1689 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1546 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1419 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:817 in forward
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:795 in interpolate_pos_encoding
```
</details>

### 2. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:795`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Unsupported Tensor.item() call with capture_scalar_outputs=False
  Explanation: Dynamo does not support tracing `Tensor.item()` with config.capture_scalar_outputs=False.
  Hint: Set `torch._dynamo.config.capture_scalar_outputs = True` or `export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` to include these operations in the captured graph.

  Developer debug context: call_method TensorVariable() item () {}

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

*** While handling this graph break, another graph break occurred: ***

graph break in loop
  Explanation: torch.compile detected a graph break in a for/while loop. Skipping the frame and falling back to eager, as graph breaks in loops are not supported.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: frame skipped: forward (/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/hy_v4_vl/modeling_hy_v4_vl.py line 805)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  ListIteratorVariable(length=2, index=1) originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/hy_v4_vl/modeling_hy_v4_vl.py", line 816
            for t, h, w in grid_thw:
                           ^^^^^^^^
  CallMethodVariable(ListVariable(length=0), append) originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/hy_v4_vl/modeling_hy_v4_vl.py", line 817
                position_embeddings_list.append(self.interpolate_pos_encoding(embeddings, h, w).squeeze(0).repeat(t, 1))
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

<details><summary>User stack</summary>

```
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1689 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1546 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1419 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1043 in forward
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:817 in forward
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:795 in interpolate_pos_encoding
```
</details>

### 3. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:796`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Unsupported Tensor.item() call with capture_scalar_outputs=False
  Explanation: Dynamo does not support tracing `Tensor.item()` with config.capture_scalar_outputs=False.
  Hint: Set `torch._dynamo.config.capture_scalar_outputs = True` or `export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` to include these operations in the captured graph.

  Developer debug context: call_method TensorVariable() item () {}

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

<details><summary>User stack</summary>

```
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1689 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1546 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1419 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1043 in forward
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:817 in forward
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:796 in interpolate_pos_encoding
```
</details>

### 4. Dynamic shape operator

- **Location:** `transformers/vision_utils.py:64`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_KW: a function call with keyword arguments, e.g. f(x=True):

Dynamic shape operator
  Explanation: Operator `aten.repeat_interleave.Tensor`'s output shape depends on input Tensor data.
  Hint: Enable tracing of dynamic shape operators with `torch._dynamo.config.capture_dynamic_output_shape_ops = True`

  Developer debug context: aten.repeat_interleave.Tensor

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

<details><summary>User stack</summary>

```
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1689 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1546 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1419 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1044 in forward
transformers/vision_utils.py:76 in get_vision_attention_seqlens
transformers/vision_utils.py:64 in get_vision_cu_seqlens
```
</details>

### 5. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:927`  (hit ×1)

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

  Developer debug context: frame skipped: torch_dynamo_resume_in_forward_at_1044 (/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/hy_v4_vl/modeling_hy_v4_vl.py line 1044)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  DictValuesIterator() originated from:
  File "/admin/home/khushi_agrawal/miniforge3/envs/torch-snap-57f02f8/lib/python3.15/site-packages/torch/nn/modules/container.py", line 416
            return iter(self._modules.values())
                   ~~~~^^^^^^^^^^^^^^^^^^^^^^^^
  NullVariable originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/hy_v4_vl/modeling_hy_v4_vl.py", line 1046
            for layer in self.layers:
                         ^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html

<details><summary>User stack</summary>

```
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1689 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1546 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1419 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1047 in forward
transformers/modeling_layers.py:122 in forward_with_checkpointing
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:927 in forward
```
</details>

### 6. Data dependent operator

- **Location:** `transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:927`  (hit ×1)

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
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1689 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1546 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1419 in get_image_features
transformers/utils/output_capturing.py:361 in wrapper
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1047 in forward
transformers/modeling_layers.py:122 in forward_with_checkpointing
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:927 in forward
```
</details>

### 7. Data dependent operator

- **Location:** `transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1423`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Data dependent operator
  Explanation: Operator `aten._local_scalar_dense.default` has a non-Tensor output whose value is dependent on the data of Tensor inputs.
  Hint: Enable tracing of data-dependent output operators with `torch._dynamo.config.capture_scalar_outputs = True`

  Developer debug context: aten._local_scalar_dense.default

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html

Stack variable source attribution:
  LazyVariableTracker(unrealized: <class 'function'>) originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/hy_v4_vl/modeling_hy_v4_vl.py", line 1423
            vision_outputs.pooler_output = torch.split(vision_outputs.pooler_output.squeeze(0), split_sizes.tolist())
                                           ^^^^^^^^^^^
  TensorVariable() originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/hy_v4_vl/modeling_hy_v4_vl.py", line 1423
            vision_outputs.pooler_output = torch.split(vision_outputs.pooler_output.squeeze(0), split_sizes.tolist())
                                                       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1689 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1546 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1423 in get_image_features
```
</details>

### 8. Unsupported Tensor.item() call with capture_scalar_outputs=False

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
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1689 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1559 in forward
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1453 in get_placeholder_mask
transformers/utils/import_utils.py:1975 in torch_compilable_check
```
</details>

### 9. Dynamic shape operator

- **Location:** `transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1360`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_KW: a function call with keyword arguments, e.g. f(x=True):

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
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1689 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1574 in forward
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1489 in compute_3d_position_ids
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1360 in get_rope_index
```
</details>

### 10. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1360`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Dynamic shape operator
  Explanation: Operator `aten.nonzero.default`'s output shape depends on input Tensor data.
  Hint: Enable tracing of dynamic shape operators with `torch._dynamo.config.capture_dynamic_output_shape_ops = True`

  Developer debug context: aten.nonzero.default

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

*** While handling this graph break, another graph break occurred: ***

graph break in loop
  Explanation: torch.compile detected a graph break in a for/while loop. Skipping the frame and falling back to eager, as graph breaks in loops are not supported.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: frame skipped: get_rope_index (/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/hy_v4_vl/modeling_hy_v4_vl.py line 1295)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  LocalGeneratorObjectVariable(enumerate) originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/hy_v4_vl/modeling_hy_v4_vl.py", line 1357
            for batch_idx, current_input_ids in enumerate(input_ids):
                                                ~~~~~~~~~^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1689 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1574 in forward
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1489 in compute_3d_position_ids
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1360 in get_rope_index
```
</details>

### 11. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1279`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

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
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1689 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1574 in forward
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1489 in compute_3d_position_ids
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1383 in get_rope_index
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1279 in get_vision_position_ids
```
</details>

### 12. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1280`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Unsupported Tensor.item() call with capture_scalar_outputs=False
  Explanation: Dynamo does not support tracing `Tensor.item()` with config.capture_scalar_outputs=False.
  Hint: Set `torch._dynamo.config.capture_scalar_outputs = True` or `export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` to include these operations in the captured graph.

  Developer debug context: call_method TensorVariable() item () {}

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

Stack variable source attribution:
  ConstantVariable(int: 1) originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/hy_v4_vl/modeling_hy_v4_vl.py", line 1279
                grid_thw[0].item() // temp_merge_size,
                ~~~~~~~~~~~~~~~~~~~^^~~~~~~~~~~~~~~~~
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1689 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1574 in forward
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1489 in compute_3d_position_ids
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1383 in get_rope_index
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1280 in get_vision_position_ids
```
</details>

### 13. Unsupported Tensor.item() call with capture_scalar_outputs=False

- **Location:** `transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1281`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Unsupported Tensor.item() call with capture_scalar_outputs=False
  Explanation: Dynamo does not support tracing `Tensor.item()` with config.capture_scalar_outputs=False.
  Hint: Set `torch._dynamo.config.capture_scalar_outputs = True` or `export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` to include these operations in the captured graph.

  Developer debug context: call_method TensorVariable() item () {}

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

Stack variable source attribution:
  LazyVariableTracker(unrealized: <class 'int'>) originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/hy_v4_vl/modeling_hy_v4_vl.py", line 1280
                grid_thw[1].item() // spatial_merge_size,
  ConstantVariable(int: 4) originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/hy_v4_vl/modeling_hy_v4_vl.py", line 1280
                grid_thw[1].item() // spatial_merge_size,
                ~~~~~~~~~~~~~~~~~~~^^~~~~~~~~~~~~~~~~~~~
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1689 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1574 in forward
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1489 in compute_3d_position_ids
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1383 in get_rope_index
transformers/models/hy_v4_vl/modeling_hy_v4_vl.py:1281 in get_vision_position_ids
```
</details>
