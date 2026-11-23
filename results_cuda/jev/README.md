# jev — graph breaks

- **HF id:** `autotrust/JEV-9B`
- **Class:** `AutoModelForDecision`
- **Config overrides:** `{'hidden_size': 512, 'text_config.num_hidden_layers': 4, 'text_config.layer_types': ['linear_attention', 'linear_attention', 'linear_attention', 'full_attention'], 'text_config.hidden_size': 512, 'text_config.intermediate_size': 1024, 'text_config.num_attention_heads': 4, 'text_config.num_key_value_heads': 2, 'text_config.head_dim': 64, 'text_config.linear_num_key_heads': 4, 'text_config.linear_num_value_heads': 8, 'text_config.linear_key_head_dim': 32, 'text_config.linear_value_head_dim': 32, 'text_config.vocab_size': 32000}`
- **torch:** `2.16.0a0+git57f02f8`  |  **transformers:** `5.24.0.dev0` (git `046bdb8cc2`)
- **Notes:** JEV decision head (scores answer options at `<|mark|>` positions) on a Qwen3.5 text backbone. One prompt per call (batch size 1).

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ok | 1 | 0 | 1 |
| train_forward | ok | 1 | 0 | 1 |

## eval_forward

### 1. Dynamic shape operator

- **Location:** `transformers/models/jev/modeling_jev.py:64`  (hit ×1)

```
Encountered graph break when attempting to trace BINARY_SUBSCR: a binary subscript, e.g. x[attr]:

Dynamic shape operator
  Explanation: Operator `aten.nonzero.default`'s output shape depends on input Tensor data.
  Hint: Enable tracing of dynamic shape operators with `torch._dynamo.config.capture_dynamic_output_shape_ops = True`

  Developer debug context: aten.nonzero.default

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

<details><summary>User stack</summary>

```
transformers/models/jev/modeling_jev.py:64 in forward
```
</details>


## train_forward

### 1. Dynamic shape operator

- **Location:** `transformers/models/jev/modeling_jev.py:64`  (hit ×1)

```
Encountered graph break when attempting to trace BINARY_SUBSCR: a binary subscript, e.g. x[attr]:

Dynamic shape operator
  Explanation: Operator `aten.nonzero.default`'s output shape depends on input Tensor data.
  Hint: Enable tracing of dynamic shape operators with `torch._dynamo.config.capture_dynamic_output_shape_ops = True`

  Developer debug context: aten.nonzero.default

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html

<details><summary>User stack</summary>

```
transformers/models/jev/modeling_jev.py:64 in forward
```
</details>
