# ms_marco_minilm — graph breaks

- **HF id:** `cross-encoder/ms-marco-MiniLM-L6-v2`
- **Class:** `AutoModelForSequenceClassification`
- **Config overrides:** `{'num_hidden_layers': 2}`
- **torch:** `2.16.0a0+git57f02f8`  |  **transformers:** `5.24.0.dev0` (git `046bdb8cc2`)
- **Notes:** Top-100 #3. Cross-encoder reranker (BertForSequenceClassification, num_labels=1: regression loss on float scores).

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ok | 1 | 0 | 0 |
| train_forward | ok | 1 | 0 | 0 |

## eval_forward

No graph breaks. ✅

## train_forward

No graph breaks. ✅
