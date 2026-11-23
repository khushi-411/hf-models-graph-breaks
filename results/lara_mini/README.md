# lara_mini — graph breaks

- **HF id:** `translated/Lara-4-Mini-Base`
- **Class:** `AutoModelForCausalLM`
- **Config overrides:** `{'num_hidden_layers': 4, 'layer_types': ['linear_attention', 'sliding_attention', 'linear_attention', 'full_attention'], 'sliding_window': 16, 'hidden_size': 512, 'num_attention_heads': 8, 'num_key_value_heads': 2, 'head_dim': 64, 'linear_num_key_heads': 4, 'linear_num_value_heads': 8, 'linear_key_head_dim': 32, 'linear_value_head_dim': 32, 'moe_intermediate_size': 128, 'shared_expert_intermediate_size': 128, 'num_experts': 8, 'num_experts_per_tok': 2, 'vocab_size': 32000}`
- **torch:** `2.16.0a0+git57f02f8`  |  **transformers:** `5.24.0.dev0` (git `046bdb8cc2`)
- **Notes:** Lara 4 Mini. Hybrid gated-delta-net / sliding / full attention with MoE.

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ERROR | - | - | - |
| eval_with_cache | ERROR | - | - | - |
| train_forward | ERROR | - | - | - |

## eval_forward

**Error while running:**

```
RuntimeError: Calling torch.triangular_solve on a CPU tensor requires compiling PyTorch with BLAS. Please use PyTorch built with BLAS support.
```

## eval_with_cache

**Error while running:**

```
RuntimeError: Calling torch.triangular_solve on a CPU tensor requires compiling PyTorch with BLAS. Please use PyTorch built with BLAS support.
```

## train_forward

**Error while running:**

```
RuntimeError: Calling torch.triangular_solve on a CPU tensor requires compiling PyTorch with BLAS. Please use PyTorch built with BLAS support.
```
