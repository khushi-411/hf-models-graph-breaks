# jamba — graph breaks

- **HF id:** `ai21labs/Jamba-tiny-dev`
- **Class:** `AutoModelForCausalLM`
- **Config overrides:** `{'num_hidden_layers': 4, 'attn_layer_period': 2, 'attn_layer_offset': 1, 'expert_layer_period': 2, 'expert_layer_offset': 1}`
- **torch:** `2.16.0a0+git9933f27`  |  **transformers:** `5.22.1`
- **Notes:** Hybrid Mamba + attention + MoE. 4 layers so every layer type appears.

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ok | 1 | 0 | 0 |
| eval_with_cache | ok | 1 | 0 | 0 |
| train_forward | ok | 1 | 0 | 0 |

## eval_forward

No graph breaks. ✅

## eval_with_cache

No graph breaks. ✅

## train_forward

No graph breaks. ✅
