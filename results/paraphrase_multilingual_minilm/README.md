# paraphrase_multilingual_minilm — graph breaks

- **HF id:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Class:** `AutoModel`
- **Config overrides:** `{'num_hidden_layers': 2, 'vocab_size': 32000}`
- **torch:** `2.16.0a0+git57f02f8`  |  **transformers:** `5.24.0.dev0` (git `046bdb8cc2`)
- **Notes:** Top-100 #10. Embedding model (BertModel).

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ok | 1 | 0 | 0 |
| train_forward | ok | 1 | 0 | 0 |

## eval_forward

No graph breaks. ✅

## train_forward

No graph breaks. ✅
