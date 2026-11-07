# clip — graph breaks

- **HF id:** `openai/clip-vit-base-patch32`
- **Class:** `AutoModel`
- **Config overrides:** `{'text_config.num_hidden_layers': 2, 'vision_config.num_hidden_layers': 2}`
- **torch:** `2.16.0a0+git9933f27`  |  **transformers:** `5.22.1`

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ok | 1 | 0 | 0 |
| train_forward | ok | 1 | 0 | 0 |

## eval_forward

No graph breaks. ✅

## train_forward

No graph breaks. ✅
