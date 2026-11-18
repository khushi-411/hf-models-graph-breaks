# t5gemma — graph breaks

- **HF id:** `google/t5gemma-2b-2b-ul2`
- **Class:** `AutoModelForSeq2SeqLM`
- **Config overrides:** `{'encoder.num_hidden_layers': 2, 'encoder.layer_types': ['sliding_attention', 'full_attention'], 'encoder.sliding_window': 16, 'encoder.hidden_size': 512, 'encoder.intermediate_size': 1024, 'encoder.head_dim': 64, 'encoder.vocab_size': 32000, 'decoder.num_hidden_layers': 2, 'decoder.layer_types': ['sliding_attention', 'full_attention'], 'decoder.sliding_window': 16, 'decoder.hidden_size': 512, 'decoder.intermediate_size': 1024, 'decoder.head_dim': 64, 'decoder.vocab_size': 32000, 'decoder.cross_attention_hidden_size': 512, 'vocab_size': 32000}`
- **torch:** `2.16.0a0+git57f02f8`  |  **transformers:** `5.24.0.dev0` (git `b7fe69cb19`)
- **Notes:** Gemma-2-based encoder-decoder. Built from T5GemmaConfig defaults (gated checkpoint).

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
