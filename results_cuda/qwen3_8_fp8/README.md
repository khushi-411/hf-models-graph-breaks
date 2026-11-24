# qwen3_8_fp8 — graph breaks

- **HF id:** `Qwen/Qwen3.8-27B-FP8`
- **Class:** `AutoModelForImageTextToText`
- **Config overrides:** `{'text_config.num_hidden_layers': 4, 'text_config.layer_types': ['linear_attention', 'linear_attention', 'linear_attention', 'full_attention'], 'text_config.hidden_size': 512, 'text_config.intermediate_size': 1024, 'text_config.num_attention_heads': 4, 'text_config.num_key_value_heads': 2, 'text_config.head_dim': 64, 'text_config.linear_num_key_heads': 4, 'text_config.linear_num_value_heads': 8, 'text_config.linear_key_head_dim': 32, 'text_config.linear_value_head_dim': 32, 'vision_config.depth': 2, 'vision_config.out_hidden_size': 512}`
- **torch:** `2.16.0a0+git57f02f8`  |  **transformers:** `5.24.0.dev0` (git `046bdb8cc2`)
- **Notes:** Top-100 #42. Fine-grained FP8 (e4m3, 128x128 blocks, dynamic activation scales) of qwen3_8.

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ERROR | - | - | - |
| eval_with_cache | ERROR | - | - | - |
| train_forward | ERROR | - | - | - |

## eval_forward

**Error while running:**

```
TorchRuntimeError: RuntimeError when making fake tensor call
  Explanation: Dynamo failed to run FX node with fake tensors: call_function <function _load_deepgemm_kernel at 0x7076679064b0>(*(), **{}): got ImportError('Failed to load `kernels-community/deep-gemm` — check that a build matches the current torch/CUDA.')
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
  Explanation: Dynamo failed to run FX node with fake tensors: call_function <function _load_deepgemm_kernel at 0x7076679064b0>(*(), **{}): got ImportError('Failed to load `kernels-community/deep-gemm` — check that a build matches the current torch/CUDA.')
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
  Explanation: Dynamo failed to run FX node with fake tensors: call_function <function _load_deepgemm_kernel at 0x7076679064b0>(*(), **{}): got ImportError('Failed to load `kernels-community/deep-gemm` — check that a build matches the current torch/CUDA.')
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
