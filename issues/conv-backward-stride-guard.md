<!-- Draft for: pytorch/pytorch. Once filed, put the URL in known_issues.yaml and set status: open -->

# [dynamic shapes] Inductor `convolution_backward` lowering specializes symbolic sizes (`guard_int_seq` on strides), so train mode recompiles on every new sequence length

### System info

- torch: `2.16.0a0+git6b3a750` (source build)
- transformers: `5.22.1`
- Python 3.14, Linux, CPU

### Description

`convolution_backward_lowering` in `torch/_inductor/kernel/conv.py` (added in #178945) picks
the stride order for the input gradient like this:

```python
guard = V.graph.sizevars.guard_int_seq
stride_order = ir.get_stride_order(guard(layout_dx.stride))
```

`guard_int` on a symbolic stride installs `Eq(s, <hint>)`. For a `(B, C, L)` conv input, the
stride is `L`, so the sequence length gets specialized. With `torch.compile(dynamic=True)`,
every new sequence length recompiles. This happens in train mode only, because the forward conv
lowering doesn't do it.

It isn't specific to one conv type. With every spatial dim dynamic, training on CPU compiles
3 graphs for 3 sizes in all of these cases: Conv1d (plain, depthwise, stride 5, dilated), Conv2d
(plain, ViT-style patch), Conv3d, ConvTranspose1d and ConvTranspose2d. Eval compiles 1 graph in
every case. On NVIDIA CUDA there's no recompile, because the lowering returns
`aten_convolution_backward_fallback` early when
`device_type == "cuda" and not torch.version.hip` (unless it's the channels-last 2D path).
That means CPU and ROCm, plus any other non-CUDA device that takes the Triton backward conv
path, are affected. ROCm hasn't been tested.

It's the same pattern as #137093: a lowering calls `guard_*` on a `sympy.Expr` where only the
*order* of the strides is needed. `lowering.py` already does the non-specializing version,
`ir.get_stride_order(stride, V.graph.sizevars.shape_env)`.

Hugging Face models hit this whenever a `Conv1d` runs over the sequence:
- Mamba-style causal depthwise conv: mamba, falcon_mamba, jamba, bamba, zamba2, falcon_h1,
  nemotron_h, lfm2, granitemoehybrid
- wav2vec2-family feature encoder: wav2vec2, wavlm, hubert

Recompiled graphs on the 2nd and 3rd input size, `train_forward`, `dynamic=True`. The inputs
are (batch, seq) = (2, 32), (3, 40), (4, 56); for audio, the waveform is 16000 / 17600 / 19200
samples. The "simulated fix" column replaces only this `guard_int_seq` call with unguarded size
hints:

| model | recompiled graphs today | with simulated fix | remaining cause |
|---|---|---|---|
| wav2vec2 | 5 | **0** | |
| wavlm | 5 | **0** | |
| bamba | 2 | **0** | |
| zamba2 | 2 | **0** | |
| falcon_h1 | 2 | **0** | |
| lfm2 | 2 | **0** | |
| granitemoehybrid | 2 | **0** | |
| nemotron_h | 2 | 1 | HF: `if pad_size > 0` |
| hubert | 5 | 2 | Inductor CPU `need_use_acc_helper` guard (`... <= 4096`) |
| mamba, falcon_mamba, jamba | 2 | 2 | HF: `associative_scan` guards `(seq // 8) % 2` |

With the fix, no Inductor specialization is left in any of these 12 models. The `ir.py:clamp`
specialization that also shows up in the Mamba family goes away with it, so it was only a
knock-on effect. The simulated fix only checks recompiles; it doesn't compare outputs with eager.

### Reproduction

```python
import torch
from torch._dynamo.testing import CompileCounterWithBackend

torch._inductor.config.force_disable_caches = True
conv = torch.nn.Conv1d(8, 8, kernel_size=4, groups=8, padding=3)  # Mamba-style causal conv

def fn(x):
    return conv(x)[..., : x.shape[-1]]

cnt = CompileCounterWithBackend("inductor")
compiled = torch.compile(fn, backend=cnt, dynamic=True)
for n in (32, 40, 56):
    compiled(torch.randn(2, 8, n, requires_grad=True)).sum().backward()
print(cnt.frame_count)  # 3 (eval / no grad: 1)
# [__recompiles] tensor 'x' size mismatch at index 2. expected 32, actual 40
```

Full script: [repros/conv_backward_stride_guard.py](repros/conv_backward_stride_guard.py).

### Expected behavior

One graph for all sequence lengths, as in eval mode.

### Possible fix

Compute the stride order symbolically instead of guarding:

```python
stride_order = ir.get_stride_order(layout_dx.stride, V.graph.sizevars.shape_env)
```

`conv_layout` (forward) intentionally keeps `guard_int_seq` on stride, padding and dilation,
because the output *shape* depends on them. Here only the relative order of the gradient's
strides is needed, so specializing isn't necessary.
