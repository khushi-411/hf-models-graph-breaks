# phi — graph breaks

- **HF id:** `microsoft/phi-2`
- **Class:** `AutoModelForCausalLM`
- **Config overrides:** `{'num_hidden_layers': 2, 'hidden_size': 1280, 'intermediate_size': 5120, 'num_attention_heads': 16, 'num_key_value_heads': 16}`
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
