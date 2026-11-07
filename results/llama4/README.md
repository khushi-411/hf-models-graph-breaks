# llama4 — graph breaks

- **HF id:** `unsloth/Llama-4-Scout-17B-16E-Instruct`
- **Class:** `AutoModelForCausalLM`
- **Config overrides:** `{'num_hidden_layers': 2, 'hidden_size': 1024, 'num_attention_heads': 16, 'num_key_value_heads': 4, 'head_dim': 64, 'intermediate_size': 1024, 'intermediate_size_mlp': 2048, 'num_local_experts': 4, 'vocab_size': 32000, 'pad_token_id': None}`
- **torch:** `2.16.0a0+git9933f27`  |  **transformers:** `5.22.1`
- **Notes:** Text tower only. MoE every layer, chunked attention, NoPE layers.

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
