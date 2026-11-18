# dinov3_vit — graph breaks

- **HF id:** `facebook/dinov3-vits16-pretrain-lvd1689m`
- **Class:** `AutoModel`
- **Config overrides:** `{'num_hidden_layers': 2}`
- **torch:** `2.16.0a0+git6b3a750`  |  **transformers:** `5.22.1`
- **Notes:** DINOv3 ViT-S/16 backbone (no head, so train_forward has no loss). Built from DINOv3ViTConfig defaults (gated checkpoint).

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ok | 1 | 0 | 0 |
| train_forward | ok | 1 | 0 | 0 |

## eval_forward

No graph breaks. ✅

## train_forward

No graph breaks. ✅
