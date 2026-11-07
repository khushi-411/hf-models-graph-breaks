# xlnet — graph breaks

- **HF id:** `xlnet/xlnet-base-cased`
- **Class:** `AutoModelForCausalLM`
- **Config overrides:** `{'n_layer': 2}`
- **torch:** `2.16.0a0+git9933f27`  |  **transformers:** `5.22.1`

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ok | 1 | 0 | 0 |
| train_forward | ok | 1 | 0 | 0 |

## eval_forward

No graph breaks. ✅

## train_forward

No graph breaks. ✅
