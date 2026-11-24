"""Opt-in local patches for testing proposed PyTorch fixes before they land.

GB_PATCH=identity   torch.utils._sympy.functions.Identity folds numbers (fix proposed in
                    issues/inductor-merge-loops-minmax-not-comparable.md), so runs can see past that crash.

Use with GB_RESULTS_SUFFIX (e.g. `_identityfix`) so patched results don't overwrite the baseline ones.
"""
import os

_PATCHES = {p.strip() for p in os.environ.get("GB_PATCH", "").split(",") if p.strip()}

if "identity" in _PATCHES:
    import sympy
    from torch.utils._sympy.functions import Identity

    @classmethod
    def _identity_eval(cls, arg):
        if arg.is_Number:
            return arg

    Identity.eval = _identity_eval
    sympy.core.cache.clear_cache()
