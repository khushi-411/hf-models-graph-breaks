# qwen3_moe — graph breaks

- **HF id:** `Qwen/Qwen3-30B-A3B`
- **Class:** `AutoModelForCausalLM`
- **Config overrides:** `{'num_hidden_layers': 2, 'num_experts': 8, 'num_experts_per_tok': 2}`
- **torch:** `2.16.0a0+git9933f27`  |  **transformers:** `5.22.1`
- **Notes:** Mixture of experts.

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
