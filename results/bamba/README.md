# bamba — graph breaks

- **HF id:** `ibm-ai-platform/Bamba-9B`
- **Class:** `AutoModelForCausalLM`
- **Config overrides:** `{'num_hidden_layers': 2, 'attn_layer_indices': [1], 'hidden_size': 1024, 'intermediate_size': 2048, 'num_attention_heads': 16, 'num_key_value_heads': 8, 'mamba_n_heads': 32, 'mamba_d_head': 64, 'vocab_size': 32000}`
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
