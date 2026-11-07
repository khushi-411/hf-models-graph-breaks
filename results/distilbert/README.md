# distilbert — graph breaks

- **HF id:** `distilbert/distilbert-base-uncased`
- **Class:** `AutoModelForMaskedLM`
- **Config overrides:** `{'n_layers': 2}`
- **torch:** `2.16.0a0+git9933f27`  |  **transformers:** `5.22.1`

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ok | 1 | 0 | 0 |
| train_forward | ok | 1 | 0 | 0 |

## eval_forward

No graph breaks. ✅

## train_forward

No graph breaks. ✅
