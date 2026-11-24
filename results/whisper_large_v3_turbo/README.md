# whisper_large_v3_turbo — graph breaks

- **HF id:** `openai/whisper-large-v3-turbo`
- **Class:** `AutoModelForSpeechSeq2Seq`
- **Config overrides:** `{'encoder_layers': 2, 'decoder_layers': 2, 'd_model': 512, 'encoder_attention_heads': 8, 'decoder_attention_heads': 8, 'encoder_ffn_dim': 1024, 'decoder_ffn_dim': 1024, 'dtype': 'float32'}`
- **torch:** `2.16.0a0+git57f02f8`  |  **transformers:** `5.24.0.dev0` (git `046bdb8cc2`)
- **Notes:** Top-100 #84. 128 mel bins, 4-layer decoder. Config dtype is float16 (from_config keeps it), so float32 is forced.

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
