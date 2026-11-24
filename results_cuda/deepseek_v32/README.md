# deepseek_v32 — graph breaks

- **HF id:** `deepseek-ai/DeepSeek-V3.2`
- **Class:** `AutoModelForCausalLM`
- **Config overrides:** `{'num_hidden_layers': 2, 'first_k_dense_replace': 1, 'layer_types': ['indexed_attention', 'indexed_attention'], 'mlp_layer_types': ['dense', 'sparse'], 'n_routed_experts': 8, 'num_experts_per_tok': 2, 'n_group': 4, 'topk_group': 2, 'hidden_size': 1024, 'num_attention_heads': 16, 'num_key_value_heads': 16, 'intermediate_size': 1024, 'moe_intermediate_size': 256, 'index_n_heads': 4, 'vocab_size': 32000, 'quantization_config': None}`
- **torch:** `2.16.0a0+git57f02f8`  |  **transformers:** `5.24.0.dev0` (git `046bdb8cc2`)
- **Notes:** Top-100 #22. MLA + DeepSeek sparse attention (lightning indexer, top-k). Layer 0 dense, layer 1 MoE (mlp_layer_types, not first_k_dense_replace, decides).

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
