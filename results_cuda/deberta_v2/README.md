# deberta_v2 — graph breaks

- **HF id:** `microsoft/deberta-v3-base`
- **Class:** `AutoModelForMaskedLM`
- **Config overrides:** `{'num_hidden_layers': 2}`
- **torch:** `2.16.0a0+git57f02f8`  |  **transformers:** `5.24.0.dev0` (git `b7fe69cb19`)
- **Notes:** Covers DeBERTa-v3 (v3 checkpoints use the DebertaV2 architecture). Disentangled relative-position attention. For DeBERTa v1 see the `deberta` entry.

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ok | 1 | 0 | 0 |
| train_forward | ok | 1 | 0 | 0 |

## eval_forward

No graph breaks. ✅

## train_forward

No graph breaks. ✅
