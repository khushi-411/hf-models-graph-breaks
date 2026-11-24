# clap — graph breaks

- **HF id:** `laion/clap-htsat-fused`
- **Class:** `AutoModel`
- **Config overrides:** `{'text_config.num_hidden_layers': 2, 'audio_config.depths': [1, 1, 1, 1]}`
- **torch:** `2.16.0a0+git57f02f8`  |  **transformers:** `5.24.0.dev0` (git `046bdb8cc2`)
- **Notes:** Top-100 #35. Audio-text dual encoder (HTS-AT Swin audio tower with feature fusion + RoBERTa text). eval: one clip marked is_longer; train: all (fusion BatchNorm needs >1 sample in train mode).

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ok | 3 | 2 | 1 |
| train_forward | ok | 3 | 2 | 1 |

## eval_forward

### 1. Dynamic shape operator

- **Location:** `transformers/models/clap/modeling_clap.py:819`  (hit ×1)

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
transformers/models/clap/modeling_clap.py:1322 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/clap/modeling_clap.py:819 in forward
```
</details>


## train_forward

### 1. Dynamic shape operator

- **Location:** `transformers/models/clap/modeling_clap.py:819`  (hit ×1)

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
transformers/models/clap/modeling_clap.py:1322 in forward
transformers/utils/generic.py:1036 in wrapper
transformers/models/clap/modeling_clap.py:819 in forward
```
</details>
