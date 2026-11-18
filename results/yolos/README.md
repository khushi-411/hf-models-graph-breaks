# yolos — graph breaks

- **HF id:** `hustvl/yolos-small`
- **Class:** `AutoModelForObjectDetection`
- **Config overrides:** `{'num_hidden_layers': 2}`
- **torch:** `2.16.0a0+git57f02f8`  |  **transformers:** `5.24.0.dev0` (git `b7fe69cb19`)
- **Notes:** train_forward needs box labels.

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ok | 1 | 0 | 0 |

## eval_forward

No graph breaks. ✅
