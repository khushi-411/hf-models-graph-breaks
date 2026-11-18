# ernie4_5_moe — graph breaks

- **HF id:** `baidu/ERNIE-4.5-21B-A3B-PT`
- **Class:** `AutoModelForCausalLM`
- **Config overrides:** `{'num_hidden_layers': 2, 'hidden_size': 512, 'intermediate_size': 1024, 'moe_intermediate_size': 128, 'num_attention_heads': 8, 'num_key_value_heads': 2, 'moe_num_experts': 8, 'moe_k': 2, 'moe_layer_end_index': 1, 'vocab_size': 32000}`
- **torch:** `2.16.0a0+git57f02f8`  |  **transformers:** `5.24.0.dev0` (git `b7fe69cb19`)
- **Notes:** Layer 0 dense, layer 1 MoE (moe_layer_start_index 1).

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
