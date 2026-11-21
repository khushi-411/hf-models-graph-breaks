# zonos3_mini — graph breaks

- **HF id:** `Zyphra/ZONOS3-Mini-hf`
- **Class:** `AutoModel`
- **Config overrides:** `{'lm.n_layers': 4}`
- **torch:** `2.16.0a0+git57f02f8`  |  **transformers:** `5.24.0.dev0` (git `b7fe69cb19`)
- **Notes:** TTS LM step (batch size 1, forward is @torch.no_grad, so no train scenario). Remote code.

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ok | 18 | 17 | 4 |

## eval_forward

### 1. Dynamic shape operator

- **Location:** `/admin/home/khushi_agrawal/.cache/huggingface/modules/transformers_modules/Zyphra/ZONOS3_hyphen_Mini_hyphen_hf/e39d30df560125d9c434acd5cb0e83e4c2d64e0d/modeling_zonos3.py:321`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_KW: a function call with keyword arguments, e.g. f(x=True):

Dynamic shape operator
  Explanation: Operator `aten._unique2.default`'s output shape depends on input Tensor data.
  Hint: Enable tracing of dynamic shape operators with `torch._dynamo.config.capture_dynamic_output_shape_ops = True`

  Developer debug context: aten._unique2.default

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

<details><summary>User stack</summary>

```
```
</details>

### 2. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `/admin/home/khushi_agrawal/.cache/huggingface/modules/transformers_modules/Zyphra/ZONOS3_hyphen_Mini_hyphen_hf/e39d30df560125d9c434acd5cb0e83e4c2d64e0d/modeling_zonos3.py:321`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Dynamic shape operator
  Explanation: Operator `aten._unique2.default`'s output shape depends on input Tensor data.
  Hint: Enable tracing of dynamic shape operators with `torch._dynamo.config.capture_dynamic_output_shape_ops = True`

  Developer debug context: aten._unique2.default

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

*** While handling this graph break, another graph break occurred: ***

graph break in loop
  Explanation: torch.compile detected a graph break in a for/while loop. Skipping the frame and falling back to eager, as graph breaks in loops are not supported.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: frame skipped: forward (/admin/home/khushi_agrawal/.cache/huggingface/modules/transformers_modules/Zyphra/ZONOS3_hyphen_Mini_hyphen_hf/e39d30df560125d9c434acd5cb0e83e4c2d64e0d/modeling_zonos3.py line 349)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  RangeIteratorVariable() originated from:
  File "/admin/home/khushi_agrawal/.cache/huggingface/modules/transformers_modules/Zyphra/ZONOS3_hyphen_Mini_hyphen_hf/e39d30df560125d9c434acd5cb0e83e4c2d64e0d/modeling_zonos3.py", line 365
            for layer in range(c["n_layers"]):
                         ~~~~~^^^^^^^^^^^^^^^
  CallMethodVariable(ListVariable(length=4), append) originated from:
  File "/admin/home/khushi_agrawal/.cache/huggingface/modules/transformers_modules/Zyphra/ZONOS3_hyphen_Mini_hyphen_hf/e39d30df560125d9c434acd5cb0e83e4c2d64e0d/modeling_zonos3.py", line 374
                sources.append(self._feed_forward(_rms(mixed, block.ffn_norm.weight, c["norm_eps"]), layer))
                ^^^^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

<details><summary>User stack</summary>

```
```
</details>

### 3. Data dependent operator

- **Location:** `/admin/home/khushi_agrawal/.cache/huggingface/modules/transformers_modules/Zyphra/ZONOS3_hyphen_Mini_hyphen_hf/e39d30df560125d9c434acd5cb0e83e4c2d64e0d/modeling_zonos3.py:321`  (hit ×1)

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
```
</details>

### 4. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `/admin/home/khushi_agrawal/.cache/huggingface/modules/transformers_modules/Zyphra/ZONOS3_hyphen_Mini_hyphen_hf/e39d30df560125d9c434acd5cb0e83e4c2d64e0d/modeling_zonos3.py:322`  (hit ×1)

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

  Developer debug context: frame skipped: torch_dynamo_resume_in__feed_forward_at_321 (/admin/home/khushi_agrawal/.cache/huggingface/modules/transformers_modules/Zyphra/ZONOS3_hyphen_Mini_hyphen_hf/e39d30df560125d9c434acd5cb0e83e4c2d64e0d/modeling_zonos3.py line 321)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  ListIteratorVariable(length=18, index=1) originated from:
  File "/admin/home/khushi_agrawal/.cache/huggingface/modules/transformers_modules/Zyphra/ZONOS3_hyphen_Mini_hyphen_hf/e39d30df560125d9c434acd5cb0e83e4c2d64e0d/modeling_zonos3.py", line 321
            for expert in torch.unique(choices).tolist():
                          ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

<details><summary>User stack</summary>

```
```
</details>
