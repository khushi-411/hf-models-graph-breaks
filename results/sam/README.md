# sam — graph breaks

- **HF id:** `facebook/sam-vit-base`
- **Class:** `AutoModel`
- **Config overrides:** `{'vision_config.num_hidden_layers': 2, 'vision_config.global_attn_indexes': [1]}`
- **torch:** `2.16.0a0+git9933f27`  |  **transformers:** `5.22.1`
- **Notes:** No prompts (image-only forward).

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ok | 1 | 0 | 0 |

## eval_forward

No graph breaks. ✅
