# cohere2 — graph breaks

- **HF id:** `CohereLabs/c4ai-command-r7b-12-2024`
- **Class:** `AutoModelForCausalLM`
- **Config overrides:** `{'num_hidden_layers': 2, 'layer_types': ['sliding_attention', 'full_attention'], 'sliding_window': 16, 'hidden_size': 512, 'intermediate_size': 1024, 'num_attention_heads': 8, 'num_key_value_heads': 8, 'head_dim': 64, 'vocab_size': 32000}`
- **torch:** `2.16.0a0+git6b3a750`  |  **transformers:** `5.22.1`
- **Notes:** Command R7B / Command A architecture. Built from Cohere2Config defaults (gated checkpoint).

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
