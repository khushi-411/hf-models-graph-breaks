# swin — graph breaks

- **HF id:** `microsoft/swin-tiny-patch4-window7-224`
- **Class:** `AutoModelForImageClassification`
- **Config overrides:** `{}`
- **torch:** `2.16.0a0+git9933f27`  |  **transformers:** `5.22.1`
- **Notes:** Shifted-window attention.

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ok | 1 | 0 | 0 |
| train_forward | ok | 1 | 0 | 0 |

## eval_forward

No graph breaks. ✅

## train_forward

No graph breaks. ✅
