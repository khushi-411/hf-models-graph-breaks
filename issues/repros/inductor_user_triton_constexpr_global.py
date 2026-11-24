"""User Triton kernel built from source at runtime (as kernels-community/gpt-oss-triton-kernels `specialize()` does).

The kernel's function globals contain a `constexpr_function` helper, but its `__module__` is a synthetic module that
does not. Eager launch works; torch.compile fails while Inductor compiles the kernel:
    NameError: '<helper> is not defined'
gpt-oss MXFP4 hits this in `_finalize_matmul` (`USE_FUSED_MIXED_PREC_ACC: tl.constexpr = cuda_capability_geq(10, 0)`).
"""
import inspect
import sys
import types

import torch
import triton
import triton.language as tl

constexpr_function = getattr(triton, "constexpr_function", None) or tl.constexpr_function


@constexpr_function
def use_double(x):  # stand-in for cuda_capability_geq(10, 0)
    return x >= 0


SRC = '''
def scale_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    DOUBLE: tl.constexpr = use_double(1)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    if DOUBLE:
        x = x * 2
    tl.store(out_ptr + offs, x, mask=mask)
'''


def define_kernel(src, module, **extra_globals):
    """Trimmed copy of gpt-oss-triton-kernels `specialize.define_kernel`."""
    def _empty_fn():
        pass
    gdict = dict(_empty_fn.__globals__) | extra_globals
    f = types.FunctionType(_empty_fn.__code__, gdict)
    f.__module__ = module.__name__
    stored = []
    gdict["stored"] = stored
    exec(src + "\nstored.append(scale_kernel)\n", gdict)
    f.__signature__ = inspect.signature(stored[0])
    f.__name__ = "scale_kernel"
    kernel = triton.JITFunction(f)
    kernel._unsafe_update_src(src)
    return kernel


module = types.ModuleType("specialized_kernels")  # like matmul_ogs_<activation>_<epilogue>
sys.modules[module.__name__] = module
module.scale_kernel = define_kernel(SRC, module, tl=tl, triton=triton, use_double=use_double)


def f(x):
    out = torch.empty_like(x)
    module.scale_kernel[(triton.cdiv(x.numel(), 128),)](x, out, x.numel(), BLOCK=128)
    return out


x = torch.randn(1000, device="cuda")
print("eager   :", torch.allclose(f(x), x * 2))
try:
    print("compiled:", torch.allclose(torch.compile(f)(x), x * 2))
except Exception as e:
    msg = str(e)
    print("compiled:", type(e).__name__, next((l.strip() for l in msg.splitlines() if "NameError" in l or "not defined" in l), msg.splitlines()[0])[:200])
