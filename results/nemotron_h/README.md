# nemotron_h — graph breaks

- **HF id:** `nvidia/NVIDIA-Nemotron-Nano-9B-v2`
- **Class:** `AutoModelForCausalLM`
- **Config overrides:** `{'num_hidden_layers': 3, 'layers_block_type': ['linear_attention', 'full_attention', 'mlp'], 'use_mamba_kernels': False, 'hidden_size': 512, 'intermediate_size': 1024, 'num_attention_heads': 8, 'num_key_value_heads': 2, 'head_dim': 64, 'mamba_num_heads': 16, 'mamba_head_dim': 64, 'n_groups': 2, 'ssm_state_size': 16, 'chunk_size': 16, 'vocab_size': 32000}`
- **torch:** `2.16.0a0+git6b3a750`  |  **transformers:** `5.22.1`
- **Notes:** Hybrid Mamba2 / attention (Nemotron Nano 9B v2).

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
