# dots3_note — graph breaks

- **HF id:** `dots-studio/dots3-note-prev`
- **Class:** `AutoModelForCausalLM`
- **Config overrides:** `{'num_hidden_layers': 3, 'layer_types': ['indexed_attention', 'indexed_attention', 'sliding_attention'], 'mlp_layer_types': ['dense', 'sparse', 'sparse'], 'hidden_size': 512, 'intermediate_size': 1024, 'moe_intermediate_size': 128, 'shared_experts_intermediate_size': 128, 'num_attention_heads': 8, 'num_key_value_heads': 8, 'head_dim': 128, 'q_lora_rank': 128, 'kv_lora_rank': 64, 'qk_rope_head_dim': 32, 'qk_nope_head_dim': 32, 'v_head_dim': 32, 'index_topk': 16, 'index_head_dim': 64, 'index_n_heads': 4, 'n_routed_experts': 8, 'num_experts_per_tok': 2, 'sliding_window': 16, 'vocab_size': 32000}`
- **torch:** `2.16.0a0+git57f02f8`  |  **transformers:** `5.24.0.dev0` (git `b7fe69cb19`)
- **Notes:** Text model of dots3-note. MLA + indexed (top-k sparse) attention, sliding attention, sigmoid MoE.

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
