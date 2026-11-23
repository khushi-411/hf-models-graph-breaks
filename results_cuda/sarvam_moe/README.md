# sarvam_moe — graph breaks

- **HF id:** `sarvamai/sarvam-30b`
- **Class:** `AutoModelForCausalLM`
- **Config overrides:** `{'num_hidden_layers': 2, 'first_k_dense_replace': 1, 'hidden_size': 512, 'intermediate_size': 1024, 'moe_intermediate_size': 128, 'moe_shared_expert_intermediate_size': 128, 'num_attention_heads': 8, 'num_key_value_heads': 2, 'head_dim': 64, 'n_routed_experts': 8, 'num_experts_per_tok': 2, 'vocab_size': 32000}`
- **torch:** `2.16.0a0+git57f02f8`  |  **transformers:** `5.24.0.dev0` (git `046bdb8cc2`)
- **Notes:** Sarvam 30B (config-only, runs Glm4MoeForCausalLM). Sigmoid-scored MoE, QK norm.

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
