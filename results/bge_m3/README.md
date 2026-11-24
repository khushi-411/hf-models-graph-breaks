# bge_m3 — graph breaks

- **HF id:** `BAAI/bge-m3`
- **Class:** `AutoModel`
- **Config overrides:** `{'num_hidden_layers': 2, 'vocab_size': 32000}`
- **torch:** `2.16.0a0+git57f02f8`  |  **transformers:** `5.24.0.dev0` (git `046bdb8cc2`)
- **Notes:** Top-100 #6. Embedding model (XLMRobertaModel, 8192 positions).

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ok | 1 | 0 | 0 |
| train_forward | ok | 1 | 0 | 0 |

## eval_forward

No graph breaks. ✅

## train_forward

No graph breaks. ✅
