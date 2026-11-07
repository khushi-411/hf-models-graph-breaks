# lfm2 — graph breaks

- **HF id:** `LiquidAI/LFM2-350M`
- **Class:** `AutoModelForCausalLM`
- **Config overrides:** `{'num_hidden_layers': 4, 'layer_types': ['conv', 'full_attention', 'conv', 'full_attention'], 'vocab_size': 32000}`
- **torch:** `2.16.0a0+git9933f27`  |  **transformers:** `5.22.1`
- **Notes:** Short convolutions + attention hybrid.

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
