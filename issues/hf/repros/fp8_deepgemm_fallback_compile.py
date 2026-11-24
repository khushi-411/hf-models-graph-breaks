"""The FP8-linear DeepGEMM -> Triton fallback in transformers/integrations/{deepgemm,finegrained_fp8}.py, without HF.

`_load_deepgemm_kernel` is `@torch._dynamo.allow_in_graph` (an opaque fx node that fills a module global) and
`fp8_linear` catches `ImportError` around it to fall back to Triton. Under torch.compile the node is run on fake
tensors while tracing, so the ImportError reaches user code as `TorchRuntimeError` and the fallback never runs.
"""
import torch

_KERNEL = None


@torch._dynamo.allow_in_graph
def _load_kernel() -> None:  # same shape as transformers' _load_deepgemm_kernel
    global _KERNEL
    if _KERNEL is not None:
        return
    raise ImportError("Failed to load `kernels-community/deep-gemm` — check that a build matches the current torch/CUDA.")


def load_kernel():
    _load_kernel()
    return _KERNEL


def deepgemm_linear(x, w):
    kernel = load_kernel()
    return kernel(x, w)


def fp8_linear(x, w):
    try:
        return deepgemm_linear(x, w)
    except (ImportError, NotImplementedError, ValueError):
        pass  # transformers logs "DeepGEMM declined this call, falling back to Triton"
    return x @ w  # Triton fallback


x, w = torch.randn(4, 8), torch.randn(8, 8)
print("eager   :", "ok" if torch.allclose(fp8_linear(x, w), x @ w) else "mismatch")
try:
    out = torch.compile(fp8_linear, fullgraph=False)(x, w)
    print("compiled:", "ok" if torch.allclose(out, x @ w) else "mismatch")
except Exception as e:
    print("compiled:", type(e).__name__, str(e).splitlines()[0][:150])
