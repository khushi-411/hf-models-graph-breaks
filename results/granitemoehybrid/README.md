# granitemoehybrid — graph breaks

- **HF id:** `ibm-granite/granite-4.0-h-tiny`
- **Class:** `AutoModelForCausalLM`
- **Config overrides:** `{'num_hidden_layers': 4, 'layer_types': ['linear_attention', 'linear_attention', 'linear_attention', 'full_attention'], 'num_local_experts': 8, 'num_experts_per_tok': 2, 'vocab_size': 32000}`
- **torch:** `2.16.0a0+git9933f27`  |  **transformers:** `5.22.1`
- **Notes:** Granite 4.0 (Mamba2 + attention + MoE).

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
