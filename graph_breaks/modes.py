"""Compile modes beyond the default `torch.compile`: what PyTorch's own benchmark runner also checks.

`check.py` (Dynamo graph breaks), `dynamic.py` (recompiles / Inductor specializations) and `accuracy.py`
(default Inductor vs eager vs fp64) cover one forward with the default compile. Real use goes through other
paths, each of which hits its own bugs. Every mode here compares against eager on the same weights and
inputs, and records errors (with the innermost Inductor frame), graph breaks, recompiles and cudagraph skips.

Modes:
  generate_dynamic   model.generate() with a DynamicCache and `torch.compile(forward, dynamic=True)`:
                     prefill + 8 decode steps, at 2 prompt sizes. Tokens and logits vs eager generate,
                     graphs compiled per call (2nd call should compile 0), recompile reasons, specializations.
  generate_static    model.generate(cache_implementation="static") with the forward compiled as HF recommends
                     (mode="reduce-overhead" on CUDA). Breaks, recompiles across decode steps, cudagraph skips.
  export_nonstrict   torch.export (strict=False) with Dim.AUTO on batch / sequence dims; then the exported
  export_strict      program is run on a 2nd input size and compared with eager. Reports specialized dims.
  aoti               AOTInductor: aoti_compile_and_package on the non-strict export, run on both sizes.
  cudagraphs         (CUDA) mode="reduce-overhead", eval and train, 3 calls so the graph is replayed.
  max_autotune       mode="max-autotune-no-cudagraphs", eval and train.
  freezing           inductor freezing=True (inference), eval.
  autocast           torch.autocast(bfloat16), eval and train, compiled vs eager under the same autocast.
  train_step         3 AdamW steps: compiled model + compiled optimizer.step vs eager; losses and parameters.
  compiled_autograd  forward + backward with compiled autograd on; gradients vs eager.
  perf               eager vs compiled latency (median of 20 after warmup), first-call compile time, peak
                     memory (CUDA), for eval and fwd+bwd. A model that compiles cleanly can still be slower.

    results_modes[_cuda]/<name>/<mode>.json
    MODES.md   (python -m graph_breaks.modes --summarize)

Examples:
    python -m graph_breaks.modes --model inkling2 --mode export_nonstrict
    python -m graph_breaks.modes --model inkling2 --device cuda            # every mode
    python -m graph_breaks.modes --summarize
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import datetime as dt
import json
import logging
import os
import statistics
import sys
import tempfile
import time
import traceback
from pathlib import Path

import torch
import torch._dynamo as dynamo
import torch._inductor.config as inductor_config
from torch._dynamo.utils import counters

from graph_breaks import accuracy, check, dynamic

ROOT = check.ROOT
SUMMARY_FILE = ROOT / "MODES.md"
MODES = ["generate_dynamic", "generate_static", "export_nonstrict", "export_strict", "aoti", "cudagraphs",
         "max_autotune", "freezing", "autocast", "fp16", "train_step", "compiled_autograd", "perf"]
CUDA_ONLY = {"cudagraphs"}
GENERATE_MODES = {"generate_dynamic", "generate_static"}
NEW_TOKENS = 8
GEN_SIZES = [(2, 32), (3, 40)]           # (batch, prompt length) per generate call
EXPORT_SIZES = [(2, 32, 16000), (3, 40, 17600)]

_LOGGERS = {
    "breaks": check.GRAPH_BREAK_LOGGER,
    "recompiles": dynamic.RECOMPILE_LOGGER,
    "cudagraphs": "torch._inductor.cudagraph_utils.__cudagraphs",
    "cudagraphs_sched": "torch._inductor.scheduler.__cudagraphs",
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def capture_logs():
    """Collect graph-break, recompile and cudagraph-skip messages for one check."""
    records: dict[str, list[str]] = {k: [] for k in _LOGGERS}
    torch._logging.set_logs(graph_breaks=True, recompiles=True, cudagraphs=True)
    handlers = []
    for key, name in _LOGGERS.items():
        class _H(logging.Handler):
            def emit(self, record, _k=key):
                records[_k].append(record.getMessage())
        logger, h = logging.getLogger(name), _H()
        handlers.append((logger, h, logger.propagate))
        logger.addHandler(h)
        logger.propagate = False
    try:
        yield records
    finally:
        for logger, h, old in handlers:
            logger.removeHandler(h)
            logger.propagate = old
        torch._logging.set_logs()


EXPERTS_IMPL = os.environ.get("GB_EXPERTS_IMPL")  # e.g. "eager": MoE without torch._grouped_mm


_FP32_MASTER = False  # set by the fp16 mode: fp16 AMP keeps fp32 master weights (GradScaler can't unscale bf16 grads)


def _build(spec, device):
    model = accuracy._build(spec, device)
    if _FP32_MASTER and next(model.parameters()).dtype in (torch.bfloat16, torch.float16):
        model = model.float()
    if EXPERTS_IMPL and hasattr(model, "set_experts_implementation"):
        model.set_experts_implementation(EXPERTS_IMPL)
    return model


def _unique_graphs() -> int:
    return int(counters["stats"]["unique_graphs"])


def _log_summary(logs: dict) -> dict:
    skips = list(dict.fromkeys(m.strip() for m in logs["cudagraphs"] + logs["cudagraphs_sched"]
                               if "skipping cudagraphs" in m.lower()))
    return {
        "graph_breaks": check.summarize_breaks(logs["breaks"], []),
        "recompile_reasons": list(dict.fromkeys(r.strip() for r in logs["recompiles"]))[:20],
        "cudagraph_skips": skips[:20],
    }


def _error(e: BaseException) -> dict:
    tb = traceback.extract_tb(e.__traceback__)
    ind = [f for f in tb if check._short_path(f.filename).startswith("torch/_inductor/")
           and not check._short_path(f.filename).startswith(dynamic._INFRA)]
    return {"status": "error", "error": f"{type(e).__name__}: {str(e)[:2000]}",
            "error_inductor_site": f"{dynamic._site(ind[-1])}:{ind[-1].lineno}" if ind else None,
            "traceback": traceback.format_exc()[-6000:]}


def _inputs(spec, model, scenario, device, batch=check.BATCH, seq=check.SEQ_LEN, audio=check.AUDIO_SAMPLES):
    with dynamic.input_sizes(batch, seq, audio):
        return check.make_inputs(spec, model, scenario, device)


def _dtype(model):
    return next(model.parameters()).dtype


def _sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _fresh():
    dynamo.reset()
    counters.clear()
    inductor_config.fallback_random = True


def _status(checks: list[dict]) -> str:
    if any(c.get("status") == "error" for c in checks):
        return "error"
    if any(c.get("accurate") is False for c in checks):
        return "mismatch"
    if any(c.get("recompiled_graphs") or c.get("graph_breaks") or c.get("cudagraph_skips")
           or c.get("specialized_dims") for c in checks):
        return "issues"
    return "ok"


def _reference(model, inputs: dict, train: bool):
    """Higher-precision eager run (fp64, else fp32), without autocast, as in accuracy.py: a check fails only when
    compiled is clearly further from it than eager is, so bf16 / autocast rounding noise doesn't count."""
    for name, dt_ in (("fp64", torch.float64), ("fp32", torch.float32)):
        cast = lambda x: x.to(dt_) if isinstance(x, torch.Tensor) and x.is_floating_point() else x
        try:
            return accuracy._run(copy.deepcopy(model).to(dt_), {k: cast(v) for k, v in inputs.items()}, train), name
        except Exception:
            continue
    return None, None


def _compiled_vs_eager(spec, device, scenario, compile_fn, ctx=contextlib.nullcontext, dtype_for_tol=None,
                       calls: int = 1, name: str | None = None) -> dict:
    """Eager vs `compile_fn(model)` on one scenario. `calls` > 1 reruns the compiled model (cudagraph replay)."""
    _fresh()
    out = {"name": name or scenario, "scenario": scenario}
    try:
        model = _build(spec, device)
        inputs = _inputs(spec, model, scenario, device)
        cmodel = copy.deepcopy(model)
        train = scenario == "train_forward"
        ref, out["reference"] = _reference(model, inputs, train)
        with ctx():
            eager = accuracy._run(model, inputs, train)
        with capture_logs() as logs, ctx():
            compiled_model = compile_fn(cmodel)
            for _ in range(calls):
                if calls > 1:
                    torch.compiler.cudagraph_mark_step_begin()
                res = {k: v.clone() for k, v in accuracy._run(compiled_model, inputs, train).items()}
        cmp = accuracy.compare(eager, res, ref, dtype_for_tol or _dtype(model))
        out.update(status="ok", graphs=_unique_graphs(), accurate=not cmp["failed"], failed=cmp["failed"][:20],
                   worst_rmse=max((t.get("rmse_compiled_vs_eager", 0.0) for t in cmp["tensors"]), default=0.0),
                   checked=cmp["checked"], tol=cmp["tol"], **_log_summary(logs))
    except Exception as e:
        out.update(_error(e))
    return out


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def _gen_inputs(spec, model, device, batch, seq):
    inputs = _inputs(spec, model, "eval_with_cache", device, batch, seq)
    inputs.pop("use_cache", None)
    inputs.pop("labels", None)
    return inputs


def _generate(model, inputs, **kw):
    cfg = model.generation_config
    pad = cfg.pad_token_id if cfg.pad_token_id is not None else 0
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=NEW_TOKENS, min_new_tokens=NEW_TOKENS, do_sample=False,
                             pad_token_id=pad, output_logits=True, return_dict_in_generate=True, **kw)
    return out.sequences, torch.stack(out.logits, 1)


def mode_generate(spec, device, static: bool) -> list[dict]:
    _fresh()
    checks = []
    try:
        model = _build(spec, device).eval()
        cmodel = copy.deepcopy(model)
        kw = {"cache_implementation": "static"} if static else {}
        if static:
            compile_kw = {"mode": "reduce-overhead"} if device.startswith("cuda") else {}
            cmodel.forward = torch.compile(cmodel.forward, **compile_kw)
        else:
            cmodel.forward = torch.compile(cmodel.forward, dynamic=True)
        with dynamic.record_shape_env_events() as events, capture_logs() as logs:
            for b, s in GEN_SIZES:
                c = {"name": f"batch={b} prompt={s}", "batch": b, "prompt": s}
                checks.append(c)
                try:
                    inputs = _gen_inputs(spec, model, device, b, s)
                    seq_e, logit_e = _generate(model, inputs, **kw)
                    before = _unique_graphs()
                    seq_c, logit_c = _generate(cmodel, inputs, **kw)
                    cmp = accuracy.compare({"logits": logit_e}, {"logits": logit_c}, None, _dtype(model))
                    c.update(status="ok", new_graphs=_unique_graphs() - before,
                             tokens_equal=bool(torch.equal(seq_e, seq_c)), accurate=not cmp["failed"],
                             worst_rmse=cmp["tensors"][0].get("rmse_compiled_vs_eager"))
                except Exception as e:
                    c.update(_error(e))
                    break
        summary = _log_summary(logs)
        grouped = dynamic._group(events)
        # Dynamic: the 2nd prompt size should compile nothing new. Static: each prompt size compiles a prefill and
        # a decode graph by design; anything beyond that (e.g. a recompile per decode step) is the problem.
        later = (sum(max(0, c.get("new_graphs", 0) - 2) for c in checks) if static
                 else sum(c.get("new_graphs", 0) for c in checks[1:]))
        checks.append({"name": "summary", "status": "ok", "graphs_first_call": checks[0].get("new_graphs"),
                       "recompiled_graphs": later,
                       "inductor_specialization_sites": sorted({e["site"] for e in grouped
                                                                if e["phase"] == "inductor" and e["kind"] == "specialize"}),
                       "specializations": [e for e in grouped if e["kind"] == "specialize"][:20], **summary})
    except Exception as e:
        checks.append({"name": "setup", **_error(e)})
    return checks


def _dynamic_shapes(inputs: dict):
    from torch.export import Dim
    shapes = {}
    for k, v in inputs.items():
        if not isinstance(v, torch.Tensor):
            shapes[k] = None
        elif k == "image_grid_thw" or v.ndim == 1:
            shapes[k] = {0: Dim.AUTO}
        elif v.ndim == 2 and not v.is_floating_point():  # ids / masks / labels: batch and sequence
            shapes[k] = {0: Dim.AUTO, 1: Dim.AUTO}
        else:
            shapes[k] = {0: Dim.AUTO}
    return shapes


def _specialized_dims(ep, inputs: dict, shapes: dict) -> list[str]:
    """Dims we asked to be dynamic that export turned into constants."""
    vals = {n.name: n.meta.get("val") for n in ep.graph.nodes if n.op == "placeholder" and n.name in inputs}
    out = []
    for k, dims in shapes.items():
        v = vals.get(k)
        if not dims or v is None or not hasattr(v, "shape"):
            continue
        for d in dims:
            if isinstance(v.shape[d], int):
                out.append(f"{k}.size({d}) = {v.shape[d]}")
    return out


def _flat_out(out):
    return {k: v.detach() for k, v in accuracy._tensors(out).items()}


def mode_export(spec, device, strict: bool, aoti: bool = False) -> list[dict]:
    _fresh()
    checks = []
    try:
        model = _build(spec, device).eval()
        b, s, a = EXPORT_SIZES[0]
        inputs = _inputs(spec, model, "eval_forward", device, b, s, a)
        shapes = _dynamic_shapes(inputs)
        c = {"name": "export", "strict": strict}
        checks.append(c)
        try:
            with torch.no_grad():
                ep = torch.export.export(model, (), inputs, dynamic_shapes=shapes, strict=strict)
            c.update(status="ok", specialized_dims=_specialized_dims(ep, inputs, shapes),
                     range_constraints={str(k): str(v) for k, v in list(ep.range_constraints.items())[:20]})
        except Exception as e:
            c.update(_error(e))
            return checks
        runner = ep.module()
        if aoti:
            c = {"name": "aoti_compile"}
            checks.append(c)
            try:
                pkg = Path(tempfile.mkdtemp()) / f"{spec['name']}.pt2"
                with capture_logs():
                    path = torch._inductor.aoti_compile_and_package(ep, package_path=str(pkg))
                runner = torch._inductor.aoti_load_package(path)
                c.update(status="ok")
            except Exception as e:
                c.update(_error(e))
                return checks
        for b, s, a in EXPORT_SIZES:
            c = {"name": f"run batch={b} seq={s}"}
            checks.append(c)
            try:
                inp = _inputs(spec, model, "eval_forward", device, b, s, a)
                with torch.no_grad():
                    ref = _flat_out(model(**inp))
                    got = _flat_out(runner(**inp))
                if len(got) != len(ref):  # AOTI / unflattened outputs can lose ModelOutput names
                    got = dict(zip(ref, got.values()))
                hi, c["reference"] = _reference(model, inp, False)
                cmp = accuracy.compare(ref, got, hi, _dtype(model))
                c.update(status="ok", accurate=not cmp["failed"], failed=cmp["failed"][:20])
            except Exception as e:
                c.update(_error(e))
    except Exception as e:
        checks.append({"name": "setup", **_error(e)})
    return checks


def mode_compile_variant(spec, device, compile_fn, scenarios, ctx=contextlib.nullcontext, calls=1, tol_dtype=None):
    return [_compiled_vs_eager(spec, device, sc, compile_fn, ctx=ctx, calls=calls, dtype_for_tol=tol_dtype)
            for sc in scenarios if sc in spec.get("scenarios", check.ALL_SCENARIOS)]


def mode_train_step(spec, device, steps: int = 3) -> list[dict]:
    _fresh()
    c = {"name": f"{steps} AdamW steps"}
    try:
        model = _build(spec, device).train()
        inputs = _inputs(spec, model, "train_forward", device)
        cmodel = copy.deepcopy(model)
        opt_e = torch.optim.AdamW(model.parameters(), lr=1e-3)
        opt_c = torch.optim.AdamW(cmodel.parameters(), lr=1e-3)
        fwd = torch.compile(cmodel)
        step = torch.compile(opt_c.step)
        losses = []
        with capture_logs() as logs:
            for _ in range(steps):
                le = dynamic._loss(model(**inputs)); le.backward(); opt_e.step(); opt_e.zero_grad(set_to_none=True)
                lc = dynamic._loss(fwd(**inputs)); lc.backward(); step(); opt_c.zero_grad(set_to_none=True)
                losses.append([le.item(), lc.item()])
        pe = {n: p.detach() for n, p in model.named_parameters()}
        pc = {n: p.detach() for n, p in cmodel.named_parameters()}
        cmp = accuracy.compare(pe, pc, None, _dtype(model))
        c.update(status="ok", losses_eager_compiled=losses, accurate=not cmp["failed"], failed=cmp["failed"][:20],
                 graphs=_unique_graphs(), **_log_summary(logs))
    except Exception as e:
        c.update(_error(e))
    return [c]


def mode_fp16_scaler_steps(spec, device, steps: int = 3) -> list[dict]:
    """fp16 mixed-precision training as people run it: autocast(float16) + GradScaler + AdamW, compiled vs eager."""
    _fresh()
    c = {"name": f"{steps} fp16 autocast + GradScaler steps"}
    try:
        model = _build(spec, device).train()
        inputs = _inputs(spec, model, "train_forward", device)
        cmodel = copy.deepcopy(model)
        dev = device.split(":")[0]
        opt_e = torch.optim.AdamW(model.parameters(), lr=1e-3)
        opt_c = torch.optim.AdamW(cmodel.parameters(), lr=1e-3)
        sc_e, sc_c = torch.amp.GradScaler(dev), torch.amp.GradScaler(dev)
        fwd = torch.compile(cmodel)
        losses, scales = [], []
        with capture_logs() as logs:
            for _ in range(steps):
                for m, opt, sc, out in ((model, opt_e, sc_e, None), (fwd, opt_c, sc_c, None)):
                    with torch.autocast(dev, dtype=torch.float16):
                        loss = dynamic._loss(m(**inputs))
                    sc.scale(loss).backward()
                    sc.step(opt)
                    sc.update()
                    opt.zero_grad(set_to_none=True)
                    losses.append(loss.item())
                scales.append([sc_e.get_scale(), sc_c.get_scale()])
        pe = {n: p.detach() for n, p in model.named_parameters()}
        pc = {n: p.detach() for n, p in cmodel.named_parameters()}
        cmp = accuracy.compare(pe, pc, None, torch.float16)
        nonfinite = [l for l in losses if l != l or abs(l) == float("inf")]
        c.update(status="ok", losses_eager_compiled=[losses[i:i + 2] for i in range(0, len(losses), 2)],
                 grad_scales_eager_compiled=scales, nonfinite_losses=len(nonfinite),
                 accurate=not cmp["failed"] and not nonfinite, failed=cmp["failed"][:20],
                 graphs=_unique_graphs(), **_log_summary(logs))
    except Exception as e:
        c.update(_error(e))
    return [c]


def _bench(fn, device, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    _sync(device)
    times = []
    for _ in range(iters):
        t = time.perf_counter(); fn(); _sync(device); times.append(time.perf_counter() - t)
    return statistics.median(times)


def mode_perf(spec, device) -> list[dict]:
    checks = []
    for sc in ("eval_forward", "train_forward"):
        _fresh()
        c = {"name": sc}
        checks.append(c)
        try:
            model = _build(spec, device).train(sc == "train_forward")
            inputs = _inputs(spec, model, sc, device)
            train = sc == "train_forward"

            def make(m):
                def run():
                    if train:
                        dynamic._loss(m(**inputs)).backward()
                        getattr(m, "_orig_mod", m).zero_grad(set_to_none=True)
                    else:
                        with torch.no_grad():
                            m(**inputs)
                return run

            eager_t = _bench(make(model), device)
            compiled = torch.compile(copy.deepcopy(model))
            if device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            t = time.perf_counter(); make(compiled)(); _sync(device); compile_s = time.perf_counter() - t
            comp_t = _bench(make(compiled), device)
            mem = {}
            if device.startswith("cuda"):
                mem["peak_mem_compiled_mb"] = torch.cuda.max_memory_allocated() / 2**20
                torch.cuda.reset_peak_memory_stats()
                make(model)()
                _sync(device)
                mem["peak_mem_eager_mb"] = torch.cuda.max_memory_allocated() / 2**20
            c.update(status="ok", eager_ms=eager_t * 1e3, compiled_ms=comp_t * 1e3, speedup=eager_t / comp_t,
                     first_call_s=compile_s, slower_than_eager=comp_t > eager_t * 1.05, **mem)
        except Exception as e:
            c.update(_error(e))
    return checks


def run_mode(spec, mode, device) -> list[dict]:
    evtr = ["eval_forward", "train_forward"]
    if mode == "generate_dynamic":
        return mode_generate(spec, device, static=False)
    if mode == "generate_static":
        return mode_generate(spec, device, static=True)
    if mode == "export_nonstrict":
        return mode_export(spec, device, strict=False)
    if mode == "export_strict":
        return mode_export(spec, device, strict=True)
    if mode == "aoti":
        return mode_export(spec, device, strict=False, aoti=True)
    if mode == "cudagraphs":
        return mode_compile_variant(spec, device, lambda m: torch.compile(m, mode="reduce-overhead"), evtr, calls=3)
    if mode == "max_autotune":
        return mode_compile_variant(spec, device, lambda m: torch.compile(m, mode="max-autotune-no-cudagraphs"), evtr)
    if mode == "freezing":
        return mode_compile_variant(spec, device, torch.compile, ["eval_forward"],
                                    ctx=lambda: inductor_config.patch(freezing=True))
    if mode == "autocast":
        return mode_compile_variant(spec, device, torch.compile, evtr, tol_dtype=torch.bfloat16,
                                    ctx=lambda: torch.autocast(device.split(":")[0], dtype=torch.bfloat16))
    if mode == "fp16":  # fp16 autocast eval/train vs eager (fp32/fp64 reference), then GradScaler training steps
        global _FP32_MASTER
        _FP32_MASTER = True
        try:
            return mode_compile_variant(spec, device, torch.compile, evtr, tol_dtype=torch.float16,
                                        ctx=lambda: torch.autocast(device.split(":")[0], dtype=torch.float16)) \
                + mode_fp16_scaler_steps(spec, device)
        finally:
            _FP32_MASTER = False
    if mode == "train_step":
        return mode_train_step(spec, device)
    if mode == "compiled_autograd":
        return mode_compile_variant(spec, device, torch.compile, ["train_forward"],
                                    ctx=lambda: dynamo.config.patch(compiled_autograd=True))
    if mode == "perf":
        return mode_perf(spec, device)
    raise ValueError(mode)


def applicable(spec, mode, device) -> bool:
    if mode in CUDA_ONLY and not device.startswith("cuda"):
        return False
    if mode in GENERATE_MODES:
        return spec.get("inputs", "text") in check._GENERATIVE_KINDS and "eval_with_cache" in spec.get(
            "scenarios", check.ALL_SCENARIOS)
    return True


def out_dir(device: str) -> Path:
    suffix = (f"_experts_{EXPERTS_IMPL}" if EXPERTS_IMPL else "") + os.environ.get("GB_RESULTS_SUFFIX", "")
    return ROOT / (("results_modes_cuda" if device.startswith("cuda") else "results_modes") + suffix)


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def _cell(r: dict) -> str:
    st = r["status"]
    if st == "ok":
        extra = []
        for c in r["checks"]:
            if c.get("slower_than_eager"):
                extra.append(f"{c['name']} {c['speedup']:.2f}x")
        return "✅" + (" " + ", ".join(extra) if extra else "")
    bad = [c for c in r["checks"] if c.get("status") == "error"]
    if bad:
        c = bad[0]
        return f"❌ {c['name']}: {c['error'].splitlines()[0][:90]}".replace("|", "\\|")
    if st == "mismatch":
        return "⚠️ mismatch: " + ", ".join(c["name"] for c in r["checks"] if c.get("accurate") is False)
    parts = []
    for c in r["checks"]:
        if c.get("recompiled_graphs"):
            parts.append(f"{c['recompiled_graphs']} recompiled")
        if c.get("graph_breaks"):
            parts.append(f"{sum(b.get('count', 1) for b in c['graph_breaks'])} breaks")
        if c.get("cudagraph_skips"):
            parts.append("cudagraphs skipped")
        if c.get("specialized_dims"):
            parts.append("specialized " + ", ".join(c["specialized_dims"][:3]))
    return "🟡 " + "; ".join(dict.fromkeys(parts))


def summarize() -> Path:
    L = ["# Compile modes beyond default torch.compile", "",
         "_Generated by `python -m graph_breaks.modes --summarize`. Do not edit by hand._", "",
         "✅ works and matches eager · 🟡 works with breaks / recompiles / cudagraph skips / specialized export dims"
         " · ⚠️ wrong values vs eager · ❌ error. Details in `results_modes[_cuda]/<model>/<mode>.json`.", ""]
    dirs = sorted(ROOT.glob("results_modes*"), key=lambda d: (len(d.name), d.name))
    for d in dirs:
        device = d.name.removeprefix("results_modes").lstrip("_") or "cpu"
        device = device.replace("experts_eager", "MoE experts_implementation=eager (no torch._grouped_mm)") \
                       .replace("identityfix", "with the Identity fix (GB_PATCH=identity)").replace("_", ", ")
        results = [json.loads(p.read_text()) for p in sorted(d.glob("*/*.json"))]
        if not results:
            continue
        env = results[0].get("env", {})
        L += [f"## {device}", "", f"torch `{env.get('torch')}`, transformers `{env.get('transformers')}`", ""]
        modes = [m for m in MODES if any(r["mode"] == m for r in results)]
        models = list(dict.fromkeys(r["model"] for r in results))
        by = {(r["model"], r["mode"]): r for r in results}
        L += ["| mode | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
        for m in modes:
            L.append(f"| {m} | " + " | ".join(_cell(by[(mod, m)]) if (mod, m) in by else "–" for mod in models) + " |")
        L.append("")
    # Distributed training (graph_breaks/distributed.py, torchrun on 2 GPUs)
    for d in sorted(ROOT.glob("results_distributed*"), key=lambda d: (len(d.name), d.name)):
        results = [json.loads(p.read_text()) for p in sorted(d.glob("*/*.json"))]
        if not results:
            continue
        label = d.name.removeprefix("results_distributed").lstrip("_").replace("_", ", ")
        L += [f"## Distributed training: {label}", "",
              "`python -m graph_breaks.distributed` (torchrun, 2 GPUs): eager-wrapped vs compiled-wrapped model, "
              "3 optimizer steps from the same weights; losses and final parameters compared. Notes: the DDP breaks in "
              "`torch/nn/parallel/distributed.py` (`Logger.set_runtime_stats_and_log`, `_inside_ddp_forward`) and the "
              "FSDP2 `_fsdp_common.py` `torch._dynamo.disable()` break are by design (DDP bookkeeping / FSDP2 hooks run "
              "eagerly) and do not split the compiled model itself.", ""]
        modes_ = list(dict.fromkeys(r["mode"] for r in results))
        models = list(dict.fromkeys(r["model"] for r in results))
        by = {(r["model"], r["mode"]): r for r in results}
        L += ["| mode | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
        for m in modes_:
            L.append(f"| {m} | " + " | ".join(_cell(by[(mod, m)]) if (mod, m) in by else "–" for mod in models) + " |")
        L.append("")
    SUMMARY_FILE.write_text("\n".join(L) + "\n")
    return SUMMARY_FILE


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--model", action="append")
    g.add_argument("--summarize", action="store_true")
    p.add_argument("--mode", action="append", choices=MODES)
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)
    if args.summarize:
        print(summarize().relative_to(ROOT))
        return 0
    by_name = {m["name"]: m for m in check.load_models()}
    code = 0
    for name in args.model:
        spec = by_name[name]
        for mode in args.mode or MODES:
            if not applicable(spec, mode, args.device):
                continue
            print(f"[{name}] {mode} ({args.device}) ...", flush=True)
            t = time.time()
            checks = run_mode(spec, mode, args.device)
            r = {"model": name, "hf_id": spec["hf_id"], "mode": mode, "device": args.device,
                 "status": _status(checks), "experts_implementation": EXPERTS_IMPL, "seconds": round(time.time() - t, 1),
                 "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                 "env": check.environment_info(), "checks": checks}
            out = out_dir(args.device) / name / f"{mode}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(r, indent=2, default=str) + "\n")
            print(f"    {r['status']}  {_cell(r)[:200]}", flush=True)
            code |= r["status"] == "error"
    return code


if __name__ == "__main__":
    sys.exit(main())
