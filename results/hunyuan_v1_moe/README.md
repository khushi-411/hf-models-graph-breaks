# hunyuan_v1_moe — graph breaks

- **HF id:** `tencent/Hunyuan-A13B-Instruct`
- **Class:** `AutoModelForCausalLM`
- **Config overrides:** `{'num_hidden_layers': 2, 'hidden_size': 512, 'num_attention_heads': 8, 'num_key_value_heads': 2, 'head_dim': 64, 'intermediate_size': 512, 'num_experts': 8, 'moe_topk': [2, 2], 'moe_intermediate_size': [256, 256], 'num_shared_expert': [1, 1], 'vocab_size': 128167}`
- **torch:** `2.16.0a0+git6b3a750`  |  **transformers:** `5.22.1`
- **Notes:** Hunyuan-A13B. Its checkpoint config uses dynamic rope (rope_type dynamic).

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
