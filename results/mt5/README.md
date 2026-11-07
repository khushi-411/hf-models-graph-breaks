# mt5 — graph breaks

- **HF id:** `google/mt5-small`
- **Class:** `AutoModelForSeq2SeqLM`
- **Config overrides:** `{'num_layers': 2, 'num_decoder_layers': 2, 'vocab_size': 32000}`
- **torch:** `2.16.0a0+git9933f27`  |  **transformers:** `5.22.1`

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ok | 1 | 0 | 0 |
| eval_with_cache | ok | 1 | 0 | 0 |
| train_forward | ok | 1 | 0 | 0 |

## eval_forward

No graph breaks. ✅

## eval_with_cache

No graph breaks. ✅

## train_forward

No graph breaks. ✅
