# gemma3n — graph breaks

- **HF id:** `unsloth/gemma-3n-E2B-it`
- **Class:** `AutoModelForCausalLM`
- **Config overrides:** `{'num_hidden_layers': 4, 'num_kv_shared_layers': 2, 'intermediate_size': [2048, 2048, 2048, 2048], 'layer_types': ['sliding_attention', 'full_attention', 'sliding_attention', 'full_attention'], 'activation_sparsity_pattern': [0.95, 0.95, 0.0, 0.0], 'vocab_size': 32000, 'vocab_size_per_layer_input': 32000}`
- **torch:** `2.16.0a0+git9933f27`  |  **transformers:** `5.22.1`
- **Notes:** Text tower only. AltUp, Laurel, per-layer embeddings, KV sharing, activation sparsity.

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
