# gpt_neo — graph breaks

- **HF id:** `EleutherAI/gpt-neo-125m`
- **Class:** `AutoModelForCausalLM`
- **Config overrides:** `{'num_layers': 2, 'attention_types': [[['global', 'local'], 1]], 'attention_layers': ['global', 'local']}`
- **torch:** `2.16.0a0+git9933f27`  |  **transformers:** `5.22.1`

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
