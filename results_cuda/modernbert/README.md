# modernbert — graph breaks

- **HF id:** `answerdotai/ModernBERT-base`
- **Class:** `AutoModelForMaskedLM`
- **Config overrides:** `{'num_hidden_layers': 2}`
- **torch:** `2.16.0a0+git57f02f8`  |  **transformers:** `5.24.0.dev0` (git `b7fe69cb19`)
- **Notes:** Alternating global/local attention; 2 layers cover both. Also clean with a padded attention mask (checked manually, not a tracked scenario).

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ok | 1 | 0 | 0 |
| train_forward | ok | 1 | 0 | 0 |

## eval_forward

No graph breaks. ✅

## train_forward

No graph breaks. ✅
