# qwen3_5_4b — graph breaks

- **HF id:** `Qwen/Qwen3.5-4B`
- **Class:** `AutoModelForImageTextToText`
- **Config overrides:** `{'text_config.num_hidden_layers': 4, 'text_config.layer_types': ['linear_attention', 'linear_attention', 'linear_attention', 'full_attention'], 'text_config.hidden_size': 512, 'text_config.intermediate_size': 1024, 'text_config.num_attention_heads': 4, 'text_config.num_key_value_heads': 2, 'text_config.head_dim': 64, 'text_config.linear_num_key_heads': 4, 'text_config.linear_num_value_heads': 8, 'text_config.linear_key_head_dim': 32, 'text_config.linear_value_head_dim': 32, 'vision_config.depth': 2, 'vision_config.out_hidden_size': 512}`
- **torch:** `2.16.0a0+git57f02f8`  |  **transformers:** `5.24.0.dev0` (git `046bdb8cc2`)
- **Notes:** Top-100 #98. Qwen3.5 (same code as qwen3_8). CUDA only (gated delta net needs triangular_solve).

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
