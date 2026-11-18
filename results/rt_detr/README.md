# rt_detr — graph breaks

- **HF id:** `PekingU/rtdetr_r50vd`
- **Class:** `AutoModelForObjectDetection`
- **Config overrides:** `{'decoder_layers': 2}`
- **torch:** `2.16.0a0+git57f02f8`  |  **transformers:** `5.24.0.dev0` (git `b7fe69cb19`)
- **Notes:** train_forward needs box labels.

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ok | 1 | 0 | 0 |

## eval_forward

No graph breaks. ✅
