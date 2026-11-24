# deepseek_v32_fp8 — graph breaks

- **HF id:** `deepseek-ai/DeepSeek-V3.2`
- **Class:** `AutoModelForCausalLM`
- **Config overrides:** `{'num_hidden_layers': 2, 'first_k_dense_replace': 1, 'layer_types': ['indexed_attention', 'indexed_attention'], 'mlp_layer_types': ['dense', 'sparse'], 'n_routed_experts': 8, 'num_experts_per_tok': 2, 'n_group': 4, 'topk_group': 2, 'hidden_size': 1024, 'num_attention_heads': 16, 'num_key_value_heads': 16, 'intermediate_size': 1024, 'moe_intermediate_size': 256, 'index_n_heads': 4, 'vocab_size': 32000}`
- **torch:** `2.16.0a0+git57f02f8`  |  **transformers:** `5.24.0.dev0` (git `046bdb8cc2`)
- **Notes:** Top-100 #22 as shipped: fine-grained FP8 with UE8M0 scales (deepseek_v32 runs it unquantized).

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ERROR | - | - | - |
| eval_with_cache | ERROR | - | - | - |
| train_forward | ERROR | - | - | - |

## eval_forward

**Error while running:**

```
TorchRuntimeError: RuntimeError when making fake tensor call
  Explanation: Dynamo failed to run FX node with fake tensors: call_function <function _load_deepgemm_kernel at 0x7ee5df6637f0>(*(), **{}): got ImportError('Failed to load `kernels-community/deep-gemm` — check that a build matches the current torch/CUDA.')
  Hint: Your code may result in an error when running in eager. Please double check that your code doesn't contain a similar error when actually running eager/uncompiled. You can do this by removing the `torch.compile` call, or by using `torch.compiler.set_stance("force_eager")`. 

  Developer debug context: 

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb4315.html

from user code:
   File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/integrations/deepgemm.py", line 601, in deepgemm_fp8_fp4_linear
    deepgemm = load_deepgemm_kernel()
               ~~~~~~~~~~~~~~~~~~~~^^
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/integrations/deepgemm.py", line 303, in load_deepgemm_kernel
    _load_deepgemm_kernel()
    ~~~~~~~~~~~~~~~~~~~~~^^

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"

```

## eval_with_cache

**Error while running:**

```
TorchRuntimeError: RuntimeError when making fake tensor call
  Explanation: Dynamo failed to run FX node with fake tensors: call_function <function _load_deepgemm_kernel at 0x7ee5df6637f0>(*(), **{}): got ImportError('Failed to load `kernels-community/deep-gemm` — check that a build matches the current torch/CUDA.')
  Hint: Your code may result in an error when running in eager. Please double check that your code doesn't contain a similar error when actually running eager/uncompiled. You can do this by removing the `torch.compile` call, or by using `torch.compiler.set_stance("force_eager")`. 

  Developer debug context: 

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb4315.html

from user code:
   File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/integrations/deepgemm.py", line 601, in deepgemm_fp8_fp4_linear
    deepgemm = load_deepgemm_kernel()
               ~~~~~~~~~~~~~~~~~~~~^^
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/integrations/deepgemm.py", line 303, in load_deepgemm_kernel
    _load_deepgemm_kernel()
    ~~~~~~~~~~~~~~~~~~~~~^^

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"

```

## train_forward

**Error while running:**

```
TorchRuntimeError: RuntimeError when making fake tensor call
  Explanation: Dynamo failed to run FX node with fake tensors: call_function <function _load_deepgemm_kernel at 0x7ee5df6637f0>(*(), **{}): got ImportError('Failed to load `kernels-community/deep-gemm` — check that a build matches the current torch/CUDA.')
  Hint: Your code may result in an error when running in eager. Please double check that your code doesn't contain a similar error when actually running eager/uncompiled. You can do this by removing the `torch.compile` call, or by using `torch.compiler.set_stance("force_eager")`. 

  Developer debug context: 

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb4315.html

from user code:
   File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/integrations/deepgemm.py", line 601, in deepgemm_fp8_fp4_linear
    deepgemm = load_deepgemm_kernel()
               ~~~~~~~~~~~~~~~~~~~~^^
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/integrations/deepgemm.py", line 303, in load_deepgemm_kernel
    _load_deepgemm_kernel()
    ~~~~~~~~~~~~~~~~~~~~~^^

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"

```
