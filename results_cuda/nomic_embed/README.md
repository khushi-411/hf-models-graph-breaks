# nomic_embed — graph breaks

- **HF id:** `nomic-ai/nomic-embed-text-v1.5`
- **Class:** `AutoModel`
- **Config overrides:** `{'n_layer': 2}`
- **torch:** `2.16.0a0+git57f02f8`  |  **transformers:** `5.24.0.dev0` (git `046bdb8cc2`)
- **Notes:** Top-100 #21. NomicBertModel (rotary BERT, SwiGLU), native in transformers.

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ok | 1 | 0 | 0 |
| train_forward | ok | 1 | 0 | 0 |

## eval_forward

No graph breaks. ✅

## train_forward

No graph breaks. ✅
