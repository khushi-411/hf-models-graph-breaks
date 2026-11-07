# mpnet — graph breaks

- **HF id:** `microsoft/mpnet-base`
- **Class:** `AutoModelForMaskedLM`
- **Config overrides:** `{'num_hidden_layers': 2}`
- **torch:** `2.16.0a0+git9933f27`  |  **transformers:** `5.22.1`
- **Notes:** Architecture of all-mpnet-base-v2 (sentence-transformers).

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ok | 1 | 0 | 0 |
| train_forward | ok | 1 | 0 | 0 |

## eval_forward

No graph breaks. ✅

## train_forward

No graph breaks. ✅
