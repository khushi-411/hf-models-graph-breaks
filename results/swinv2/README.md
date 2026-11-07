# swinv2 — graph breaks

- **HF id:** `microsoft/swinv2-tiny-patch4-window8-256`
- **Class:** `AutoModelForImageClassification`
- **Config overrides:** `{'depths': [1, 1, 2, 1]}`
- **torch:** `2.16.0a0+git9933f27`  |  **transformers:** `5.22.1`

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ok | 1 | 0 | 0 |
| train_forward | ok | 1 | 0 | 0 |

## eval_forward

No graph breaks. ✅

## train_forward

No graph breaks. ✅
