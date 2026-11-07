# convnextv2 — graph breaks

- **HF id:** `facebook/convnextv2-tiny-1k-224`
- **Class:** `AutoModelForImageClassification`
- **Config overrides:** `{'depths': [1, 1, 1, 1]}`
- **torch:** `2.16.0a0+git9933f27`  |  **transformers:** `5.22.1`

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ok | 1 | 0 | 0 |
| train_forward | ok | 1 | 0 | 0 |

## eval_forward

No graph breaks. ✅

## train_forward

No graph breaks. ✅
