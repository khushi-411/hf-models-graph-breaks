# encodec — graph breaks

- **HF id:** `facebook/encodec_24khz`
- **Class:** `AutoModel`
- **Config overrides:** `{'num_lstm_layers': 1}`
- **torch:** `2.16.0a0+git57f02f8`  |  **transformers:** `5.24.0.dev0` (git `b7fe69cb19`)

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ok | 36 | 35 | 3 |
| train_forward | ok | 36 | 35 | 3 |

## eval_forward

### 1. Data-dependent branching

- **Location:** `transformers/models/encodec/modeling_encodec.py:150`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow.

      The branch condition involves a tensor computed as follows:
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/encodec/modeling_encodec.py", line 132, in _get_extra_padding_for_conv1d, code: n_frames = (length - self.kernel_size + self.padding_total) / self.stride + 1
        l_self_modules_encoder_modules_layers_modules_0_buffers_padding_total_: graph input (L_self_modules_encoder_modules_layers_modules_0_buffers_padding_total_)
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/encodec/modeling_encodec.py", line 132, in _get_extra_padding_for_conv1d, code: n_frames = (length - self.kernel_size + self.padding_total) / self.stride + 1
        l_self_modules_encoder_modules_layers_modules_0_buffers_stride_: graph input (L_self_modules_encoder_modules_layers_modules_0_buffers_stride_)
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/encodec/modeling_encodec.py", line 132, in _get_extra_padding_for_conv1d, code: n_frames = (length - self.kernel_size + self.padding_total) / self.stride + 1
        l_self_modules_encoder_modules_layers_modules_0_buffers_kernel_size_: graph input (L_self_modules_encoder_modules_layers_modules_0_buffers_kernel_size_)
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/encodec/modeling_encodec.py", line 133, in _get_extra_padding_for_conv1d, code: n_frames = torch.ceil(n_frames).to(torch.int64) - 1
        n_frames_1 = sub(to, 1)
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/encodec/modeling_encodec.py", line 134, in _get_extra_padding_for_conv1d, code: ideal_length = n_frames * self.stride + self.kernel_size - self.padding_total
        mul_1 = mul(n_frames_1, l_self_modules_encoder_modules_layers_modules_0_buffers_stride_)
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/encodec/modeling_encodec.py", line 134, in _get_extra_padding_for_conv1d, code: ideal_length = n_frames * self.stride + self.kernel_size - self.padding_total
        add_2 = add(mul_1, l_self_modules_encoder_modules_layers_modules_0_buffers_kernel_size_)
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/encodec/modeling_encodec.py", line 134, in _get_extra_padding_for_conv1d, code: ideal_length = n_frames * self.stride + self.kernel_size - self.padding_total
        ideal_length = sub(add_2, l_self_modules_encoder_modules_layers_modules_0_buffers_padding_total_)
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/encodec/modeling_encodec.py", line 136, in _get_extra_padding_for_conv1d, code: return ideal_length - length
        extra_padding = sub(ideal_length, 24000)
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/encodec/modeling_encodec.py", line 148, in _pad1d, code: max_pad = max(padding_left, padding_right)
        max_pad = maximum(l_self_modules_encoder_modules_layers_modules_0_buffers_padding_total_, extra_padding)
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/encodec/modeling_encodec.py", line 150, in _pad1d, code: if length <= max_pad:
        ge = ge(max_pad, 24000)
    
  Hint: For the common pattern `if tensor_cond: x = transform(x)` (e.g. clamping inf/nan values), consider making the code branchless by always applying the transform. Operations like torch.clamp, torch.nan_to_num, and torch.where are typically no-ops on well-behaved inputs and compile without graph breaks.
  Hint: This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround.
  Hint: Use `torch.cond` to express dynamic control flow.

  Developer debug context: attempted to jump with TensorVariable()

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html

<details><summary>User stack</summary>

```
transformers/models/encodec/modeling_encodec.py:793 in forward
transformers/models/encodec/modeling_encodec.py:590 in encode
transformers/models/encodec/modeling_encodec.py:519 in _encode_frame
transformers/models/encodec/modeling_encodec.py:162 in forward
transformers/models/encodec/modeling_encodec.py:150 in _pad1d
```
</details>

### 2. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/encodec/modeling_encodec.py:150`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow.

      The branch condition involves a tensor computed as follows:
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/encodec/modeling_encodec.py", line 132, in _get_extra_padding_for_conv1d, code: n_frames = (length - self.kernel_size + self.padding_total) / self.stride + 1
        l_self_modules_encoder_modules_layers_modules_0_buffers_padding_total_: graph input (L_self_modules_encoder_modules_layers_modules_0_buffers_padding_total_)
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/encodec/modeling_encodec.py", line 132, in _get_extra_padding_for_conv1d, code: n_frames = (length - self.kernel_size + self.padding_total) / self.stride + 1
        l_self_modules_encoder_modules_layers_modules_0_buffers_stride_: graph input (L_self_modules_encoder_modules_layers_modules_0_buffers_stride_)
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/encodec/modeling_encodec.py", line 132, in _get_extra_padding_for_conv1d, code: n_frames = (length - self.kernel_size + self.padding_total) / self.stride + 1
        l_self_modules_encoder_modules_layers_modules_0_buffers_kernel_size_: graph input (L_self_modules_encoder_modules_layers_modules_0_buffers_kernel_size_)
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/encodec/modeling_encodec.py", line 133, in _get_extra_padding_for_conv1d, code: n_frames = torch.ceil(n_frames).to(torch.int64) - 1
        n_frames_1 = sub(to, 1)
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/encodec/modeling_encodec.py", line 134, in _get_extra_padding_for_conv1d, code: ideal_length = n_frames * self.stride + self.kernel_size - self.padding_total
        mul_1 = mul(n_frames_1, l_self_modules_encoder_modules_layers_modules_0_buffers_stride_)
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/encodec/modeling_encodec.py", line 134, in _get_extra_padding_for_conv1d, code: ideal_length = n_frames * self.stride + self.kernel_size - self.padding_total
        add_2 = add(mul_1, l_self_modules_encoder_modules_layers_modules_0_buffers_kernel_size_)
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/encodec/modeling_encodec.py", line 134, in _get_extra_padding_for_conv1d, code: ideal_length = n_frames * self.stride + self.kernel_size - self.padding_total
        ideal_length = sub(add_2, l_self_modules_encoder_modules_layers_modules_0_buffers_padding_total_)
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/encodec/modeling_encodec.py", line 136, in _get_extra_padding_for_conv1d, code: return ideal_length - length
        extra_padding = sub(ideal_length, 24000)
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/encodec/modeling_encodec.py", line 148, in _pad1d, code: max_pad = max(padding_left, padding_right)
        max_pad = maximum(l_self_modules_encoder_modules_layers_modules_0_buffers_padding_total_, extra_padding)
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/encodec/modeling_encodec.py", line 150, in _pad1d, code: if length <= max_pad:
        ge = ge(max_pad, 24000)
    
  Hint: For the common pattern `if tensor_cond: x = transform(x)` (e.g. clamping inf/nan values), consider making the code branchless by always applying the transform. Operations like torch.clamp, torch.nan_to_num, and torch.where are typically no-ops on well-behaved inputs and compile without graph breaks.
  Hint: This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround.
  Hint: Use `torch.cond` to express dynamic control flow.

  Developer debug context: attempted to jump with TensorVariable()

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html

*** While handling this graph break, another graph break occurred: ***

graph break in loop
  Explanation: torch.compile detected a graph break in a for/while loop. Skipping the frame and falling back to eager, as graph breaks in loops are not supported.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: frame skipped: encode (/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/encodec/modeling_encodec.py line 524)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  RangeIteratorVariable() originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/encodec/modeling_encodec.py", line 587
            for offset in range(0, input_length, stride):
                          ~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html

<details><summary>User stack</summary>

```
transformers/models/encodec/modeling_encodec.py:793 in forward
transformers/models/encodec/modeling_encodec.py:590 in encode
transformers/models/encodec/modeling_encodec.py:519 in _encode_frame
transformers/models/encodec/modeling_encodec.py:162 in forward
transformers/models/encodec/modeling_encodec.py:150 in _pad1d
```
</details>

### 3. Unsupported function call (delayed)

- **Location:** `transformers/models/encodec/modeling_encodec.py:247`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Unsupported function call (delayed)
  Explanation: Dynamo determined that a graph break should occur when calling `L['self']._modules['lstm']`. Reason: Dynamo does not support RNN, GRU, or LSTM. Set torch._dynamo.config.allow_rnn=True to enable experimental support for RNN, GRU, and LSTM in Dynamo


  Developer debug context: source: DictGetItemSource(base=AttrSource(base=UnspecializedNNModuleSource(base=LocalSource(local_name='self', is_input=True, dynamism=None, is_derefed_cell_contents=False)), member='_modules'), index='lstm')

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0148.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0148.html

<details><summary>User stack</summary>

```
transformers/models/encodec/modeling_encodec.py:793 in forward
transformers/models/encodec/modeling_encodec.py:590 in encode
transformers/models/encodec/modeling_encodec.py:519 in _encode_frame
transformers/models/encodec/modeling_encodec.py:312 in forward
transformers/models/encodec/modeling_encodec.py:247 in forward
```
</details>


## train_forward

### 1. Data-dependent branching

- **Location:** `transformers/models/encodec/modeling_encodec.py:150`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow.

      The branch condition involves a tensor computed as follows:
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/encodec/modeling_encodec.py", line 132, in _get_extra_padding_for_conv1d, code: n_frames = (length - self.kernel_size + self.padding_total) / self.stride + 1
        l_self_modules_encoder_modules_layers_modules_0_buffers_padding_total_: graph input (L_self_modules_encoder_modules_layers_modules_0_buffers_padding_total_)
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/encodec/modeling_encodec.py", line 132, in _get_extra_padding_for_conv1d, code: n_frames = (length - self.kernel_size + self.padding_total) / self.stride + 1
        l_self_modules_encoder_modules_layers_modules_0_buffers_stride_: graph input (L_self_modules_encoder_modules_layers_modules_0_buffers_stride_)
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/encodec/modeling_encodec.py", line 132, in _get_extra_padding_for_conv1d, code: n_frames = (length - self.kernel_size + self.padding_total) / self.stride + 1
        l_self_modules_encoder_modules_layers_modules_0_buffers_kernel_size_: graph input (L_self_modules_encoder_modules_layers_modules_0_buffers_kernel_size_)
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/encodec/modeling_encodec.py", line 133, in _get_extra_padding_for_conv1d, code: n_frames = torch.ceil(n_frames).to(torch.int64) - 1
        n_frames_1 = sub(to, 1)
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/encodec/modeling_encodec.py", line 134, in _get_extra_padding_for_conv1d, code: ideal_length = n_frames * self.stride + self.kernel_size - self.padding_total
        mul_1 = mul(n_frames_1, l_self_modules_encoder_modules_layers_modules_0_buffers_stride_)
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/encodec/modeling_encodec.py", line 134, in _get_extra_padding_for_conv1d, code: ideal_length = n_frames * self.stride + self.kernel_size - self.padding_total
        add_2 = add(mul_1, l_self_modules_encoder_modules_layers_modules_0_buffers_kernel_size_)
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/encodec/modeling_encodec.py", line 134, in _get_extra_padding_for_conv1d, code: ideal_length = n_frames * self.stride + self.kernel_size - self.padding_total
        ideal_length = sub(add_2, l_self_modules_encoder_modules_layers_modules_0_buffers_padding_total_)
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/encodec/modeling_encodec.py", line 136, in _get_extra_padding_for_conv1d, code: return ideal_length - length
        extra_padding = sub(ideal_length, 24000)
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/encodec/modeling_encodec.py", line 148, in _pad1d, code: max_pad = max(padding_left, padding_right)
        max_pad = maximum(l_self_modules_encoder_modules_layers_modules_0_buffers_padding_total_, extra_padding)
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/encodec/modeling_encodec.py", line 150, in _pad1d, code: if length <= max_pad:
        ge = ge(max_pad, 24000)
    
  Hint: For the common pattern `if tensor_cond: x = transform(x)` (e.g. clamping inf/nan values), consider making the code branchless by always applying the transform. Operations like torch.clamp, torch.nan_to_num, and torch.where are typically no-ops on well-behaved inputs and compile without graph breaks.
  Hint: This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround.
  Hint: Use `torch.cond` to express dynamic control flow.

  Developer debug context: attempted to jump with TensorVariable()

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html

<details><summary>User stack</summary>

```
transformers/models/encodec/modeling_encodec.py:793 in forward
transformers/models/encodec/modeling_encodec.py:590 in encode
transformers/models/encodec/modeling_encodec.py:519 in _encode_frame
transformers/models/encodec/modeling_encodec.py:162 in forward
transformers/models/encodec/modeling_encodec.py:150 in _pad1d
```
</details>

### 2. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `transformers/models/encodec/modeling_encodec.py:150`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow.

      The branch condition involves a tensor computed as follows:
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/encodec/modeling_encodec.py", line 132, in _get_extra_padding_for_conv1d, code: n_frames = (length - self.kernel_size + self.padding_total) / self.stride + 1
        l_self_modules_encoder_modules_layers_modules_0_buffers_padding_total_: graph input (L_self_modules_encoder_modules_layers_modules_0_buffers_padding_total_)
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/encodec/modeling_encodec.py", line 132, in _get_extra_padding_for_conv1d, code: n_frames = (length - self.kernel_size + self.padding_total) / self.stride + 1
        l_self_modules_encoder_modules_layers_modules_0_buffers_stride_: graph input (L_self_modules_encoder_modules_layers_modules_0_buffers_stride_)
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/encodec/modeling_encodec.py", line 132, in _get_extra_padding_for_conv1d, code: n_frames = (length - self.kernel_size + self.padding_total) / self.stride + 1
        l_self_modules_encoder_modules_layers_modules_0_buffers_kernel_size_: graph input (L_self_modules_encoder_modules_layers_modules_0_buffers_kernel_size_)
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/encodec/modeling_encodec.py", line 133, in _get_extra_padding_for_conv1d, code: n_frames = torch.ceil(n_frames).to(torch.int64) - 1
        n_frames_1 = sub(to, 1)
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/encodec/modeling_encodec.py", line 134, in _get_extra_padding_for_conv1d, code: ideal_length = n_frames * self.stride + self.kernel_size - self.padding_total
        mul_1 = mul(n_frames_1, l_self_modules_encoder_modules_layers_modules_0_buffers_stride_)
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/encodec/modeling_encodec.py", line 134, in _get_extra_padding_for_conv1d, code: ideal_length = n_frames * self.stride + self.kernel_size - self.padding_total
        add_2 = add(mul_1, l_self_modules_encoder_modules_layers_modules_0_buffers_kernel_size_)
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/encodec/modeling_encodec.py", line 134, in _get_extra_padding_for_conv1d, code: ideal_length = n_frames * self.stride + self.kernel_size - self.padding_total
        ideal_length = sub(add_2, l_self_modules_encoder_modules_layers_modules_0_buffers_padding_total_)
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/encodec/modeling_encodec.py", line 136, in _get_extra_padding_for_conv1d, code: return ideal_length - length
        extra_padding = sub(ideal_length, 24000)
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/encodec/modeling_encodec.py", line 148, in _pad1d, code: max_pad = max(padding_left, padding_right)
        max_pad = maximum(l_self_modules_encoder_modules_layers_modules_0_buffers_padding_total_, extra_padding)
    
        # File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/encodec/modeling_encodec.py", line 150, in _pad1d, code: if length <= max_pad:
        ge = ge(max_pad, 24000)
    
  Hint: For the common pattern `if tensor_cond: x = transform(x)` (e.g. clamping inf/nan values), consider making the code branchless by always applying the transform. Operations like torch.clamp, torch.nan_to_num, and torch.where are typically no-ops on well-behaved inputs and compile without graph breaks.
  Hint: This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround.
  Hint: Use `torch.cond` to express dynamic control flow.

  Developer debug context: attempted to jump with TensorVariable()

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html

*** While handling this graph break, another graph break occurred: ***

graph break in loop
  Explanation: torch.compile detected a graph break in a for/while loop. Skipping the frame and falling back to eager, as graph breaks in loops are not supported.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: frame skipped: encode (/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/encodec/modeling_encodec.py line 524)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  RangeIteratorVariable() originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/encodec/modeling_encodec.py", line 587
            for offset in range(0, input_length, stride):
                          ~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html

<details><summary>User stack</summary>

```
transformers/models/encodec/modeling_encodec.py:793 in forward
transformers/models/encodec/modeling_encodec.py:590 in encode
transformers/models/encodec/modeling_encodec.py:519 in _encode_frame
transformers/models/encodec/modeling_encodec.py:162 in forward
transformers/models/encodec/modeling_encodec.py:150 in _pad1d
```
</details>

### 3. Unsupported function call (delayed)

- **Location:** `transformers/models/encodec/modeling_encodec.py:247`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Unsupported function call (delayed)
  Explanation: Dynamo determined that a graph break should occur when calling `L['self']._modules['lstm']`. Reason: Dynamo does not support RNN, GRU, or LSTM. Set torch._dynamo.config.allow_rnn=True to enable experimental support for RNN, GRU, and LSTM in Dynamo


  Developer debug context: source: DictGetItemSource(base=AttrSource(base=UnspecializedNNModuleSource(base=LocalSource(local_name='self', is_input=True, dynamism=None, is_derefed_cell_contents=False)), member='_modules'), index='lstm')

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0148.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0148.html

<details><summary>User stack</summary>

```
transformers/models/encodec/modeling_encodec.py:793 in forward
transformers/models/encodec/modeling_encodec.py:590 in encode
transformers/models/encodec/modeling_encodec.py:519 in _encode_frame
transformers/models/encodec/modeling_encodec.py:312 in forward
transformers/models/encodec/modeling_encodec.py:247 in forward
```
</details>
