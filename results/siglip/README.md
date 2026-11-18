# siglip — graph breaks

- **HF id:** `google/siglip-base-patch16-224`
- **Class:** `AutoModel`
- **Config overrides:** `{'text_config.num_hidden_layers': 2, 'vision_config.num_hidden_layers': 2}`
- **torch:** `2.16.0a0+git57f02f8`  |  **transformers:** `5.24.0.dev0` (git `b7fe69cb19`)

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ok | 1 | 0 | 0 |
| train_forward | ok | 1 | 0 | 0 |

## eval_forward

No graph breaks. ✅

## train_forward

No graph breaks. ✅
