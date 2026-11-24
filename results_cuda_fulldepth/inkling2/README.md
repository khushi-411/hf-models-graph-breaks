# inkling2 — graph breaks

- **HF id:** `thinkingmachines/Inkling-2-Small-MXFP8`
- **Class:** `AutoModelForCausalLM`
- **Config overrides:** `{'num_hidden_layers': 7, 'layer_types': ['hybrid_sliding', 'hybrid_sliding', 'hybrid_sliding', 'hybrid_sliding', 'hybrid_sliding', 'hybrid', 'hybrid_sliding'], 'hidden_size': 512, 'num_attention_heads': 8, 'num_key_value_heads': 2, 'head_dim': 64, 'swa_num_attention_heads': 8, 'swa_num_key_value_heads': 2, 'swa_head_dim': 64, 'sliding_window_size': 16, 'rel_extent': 16, 'moe_intermediate_size': 256, 'n_routed_experts': 8, 'num_experts_per_tok': 2, 'vocab_size': 32000, 'unpadded_vocab_size': 31000, 'quantization_config': None, 'dtype': 'float32'}`
- **torch:** `2.16.0a0+gitf29b6d5`  |  **transformers:** `5.24.0.dev0` (git `046bdb8cc2`)
- **Notes:** Text model of Inkling-2 (different architecture from `inkling`). Sigmoid-routed MoE, attention residual blocks, logit softcapping.

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
