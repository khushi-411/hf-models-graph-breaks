# inkling — graph breaks

- **HF id:** `thinkingmachines/Inkling`
- **Class:** `AutoModelForCausalLM`
- **Config overrides:** `{'num_hidden_layers': 2, 'layer_types': ['hybrid_sliding', 'hybrid'], 'mlp_layer_types': ['dense', 'sparse'], 'hidden_size': 512, 'num_attention_heads': 8, 'num_key_value_heads': 2, 'head_dim': 64, 'swa_num_attention_heads': 8, 'swa_num_key_value_heads': 2, 'swa_head_dim': 64, 'sliding_window_size': 16, 'rel_extent': 16, 'intermediate_size': 1024, 'moe_intermediate_size': 256, 'n_routed_experts': 8, 'num_experts_per_tok': 2, 'n_shared_experts': 1, 'vocab_size': 32000, 'unpadded_vocab_size': 31000}`
- **torch:** `2.16.0a0+git9933f27`  |  **transformers:** `5.22.1`
- **Notes:** Text model of the multimodal Inkling checkpoint. Hybrid sliding/full attention with hidden-state-conditioned relative position bias, log-scaled full attention, short convolutions, MoE with shared-expert sink. Layer 0 sliding + dense MLP, layer 1 full + MoE. MTP layers are not loaded by transformers.

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
