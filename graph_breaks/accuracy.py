"""Does torch.compile (Inductor) give the same numbers as eager? Checks every model in models.yaml.

    python -m graph_breaks.accuracy --model gpt2 [--device cuda] [--scenario eval_forward]
    python -m graph_breaks.accuracy --summarize          # writes ACCURACY.md from results_accuracy*/

Method (the one PyTorch's own benchmark suite uses, `torch._dynamo.utils.same` with an fp64 reference):
  1. Build the model once (config-only, reduced layers, random weights, the checkpoint's dtype), then three copies:
     eager, compiled (`torch.compile(backend="inductor")`), and an fp64 copy run eagerly as the reference.
  2. Run the scenario on the same inputs. In train_forward, also run backward and collect every parameter gradient.
  3. A tensor fails only if the compiled result is clearly further from fp64 than eager is:
     rmse(compiled, fp64) > multiplier * rmse(eager, fp64) + tol / 10 (multiplier 2-3, see `same`).
     Low-precision models (bf16 / fp16) get a looser tol because eager itself is far from fp64.
Dropout, LayerDrop and drop-path are set to 0 so train mode is deterministic (the train code path is kept), and
`fallback_random` is on in case anything still draws random numbers.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import datetime as dt
import json
import sys
import traceback
from pathlib import Path

import torch
import torch._dynamo as dynamo
import torch._inductor.config as inductor_config
from torch._dynamo.utils import same

from graph_breaks import check

ROOT = Path(__file__).resolve().parent.parent
SCENARIOS = check.ALL_SCENARIOS
TOL = {torch.float32: 1e-3, torch.float64: 1e-3, torch.bfloat16: 1e-2, torch.float16: 1e-2}


def _zero_randomness(config) -> None:
    """Every training-time random probability -> 0 in the config and every sub-config: dropout, LayerDrop,
    drop-path / drop-connect (stochastic depth), and SpecAugment masking (`mask_time_prob`, `mask_feature_prob`,
    drawn with numpy, so no torch seed or `fallback_random` can make eager and compiled agree)."""
    seen = set()

    def visit(cfg):
        if id(cfg) in seen or not hasattr(cfg, "to_dict"):
            return
        seen.add(id(cfg))
        for k, v in list(vars(cfg).items()):
            key = k.lower()
            if isinstance(v, float) and ("drop" in key or (key.startswith("mask_") and key.endswith("_prob"))):
                setattr(cfg, k, 0.0)
            elif hasattr(v, "to_dict"):
                visit(v)
        for name in getattr(cfg, "sub_configs", {}) or {}:
            visit(getattr(cfg, name, None))

    visit(config)


def _build(spec: dict, device: str):
    config = check.build_config(spec)
    _zero_randomness(config)
    qconfig = check.pop_quantization_config(config) if spec.get("quantize") else None
    import transformers
    auto_cls = getattr(transformers, spec["auto_class"])
    remote = {"trust_remote_code": True} if spec.get("trust_remote_code") else {}
    torch.manual_seed(0)
    if spec.get("pretrained"):  # real (quantized) checkpoint, see check.build_model
        model = auto_cls.from_pretrained(spec["hf_id"], config=config, device_map=device, **remote)
    else:
        model = auto_cls.from_config(config, **remote)
    if spec.get("init_weights") == "normal":
        with torch.no_grad():
            for p in model.parameters():
                p.normal_(0.0, 0.02)
    if qconfig is not None:  # quantized checkpoints: no fp64 reference (quantized weights), compiled vs eager only
        model = check.quantize_model(spec, model, qconfig, device)
    for m in model.modules():  # modules that hard-code their dropout instead of reading the config
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    return model.to(device)


def _tensors(out, prefix="out") -> dict[str, torch.Tensor]:
    """Every floating-point tensor in a model output, by name (Cache objects and ints are skipped)."""
    found: dict[str, torch.Tensor] = {}
    if isinstance(out, torch.Tensor):
        if out.is_floating_point() or out.is_complex():
            found[prefix] = out
    elif isinstance(out, dict) or hasattr(out, "items"):
        for k, v in out.items():
            found.update(_tensors(v, f"{prefix}.{k}"))
    elif isinstance(out, (list, tuple)):
        for i, v in enumerate(out):
            found.update(_tensors(v, f"{prefix}[{i}]"))
    return found


def _to64(x):
    if isinstance(x, torch.Tensor) and x.is_floating_point():
        return x.double()
    return x


def _run(model, inputs: dict, train: bool) -> dict[str, torch.Tensor]:
    model.train(train)
    ctx = contextlib.nullcontext() if train else torch.no_grad()
    with ctx:
        out = model(**inputs)
    res = {k: v.detach() for k, v in _tensors(out).items()}
    if train:
        loss = getattr(out, "loss", None)
        if loss is None:  # no labels: reduce the main output, as dynamic.py does
            first = next(iter(_tensors(out).values()))
            loss = first.float().sum()
        loss.backward()
        for n, p in getattr(model, "_orig_mod", model).named_parameters():  # same names for the compiled wrapper
            if p.grad is not None:
                res[f"grad.{n}"] = p.grad.detach()
        getattr(model, "_orig_mod", model).zero_grad(set_to_none=True)
    return res


def _rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.sqrt(torch.mean(torch.square(a.double() - b.double()))).item()


def compare(eager: dict, compiled: dict, fp64: dict | None, dtype) -> dict:
    tol = TOL.get(dtype, 1e-3)
    rows, failed = [], []
    for name, ref in eager.items():
        res = compiled.get(name)
        if res is None or res.shape != ref.shape:
            failed.append(name)
            rows.append({"name": name, "ok": False, "why": "missing or shape mismatch in compiled output"})
            continue
        ref64 = (fp64 or {}).get(name)
        if ref64 is not None and ref64.shape != ref.shape:
            ref64 = None
        nan = bool(torch.isnan(res).any() and not torch.isnan(ref).any())
        # `same` only applies its RMSE-vs-reference test when the reference is float64. Models often return fp32
        # from the fp64 run anyway (`logits.float()`, losses), and the fp32 fallback reference is fp32 too, so
        # upcast here; otherwise `same` silently drops to a plain allclose.
        ok = (not nan) and same(ref, res, fp64_ref=None if ref64 is None else ref64.double(), tol=tol,
                                equal_nan=True, exact_dtype=False, log_error=lambda *a, **k: None)
        row = {"name": name, "ok": bool(ok), "shape": list(ref.shape),
               "max_abs_diff": (res.double() - ref.double()).abs().max().item() if ref.numel() else 0.0,
               "rmse_compiled_vs_eager": _rmse(res, ref)}
        if ref64 is not None:
            row["rmse_eager_vs_fp64"] = _rmse(ref, ref64)
            row["rmse_compiled_vs_fp64"] = _rmse(res, ref64)
        if nan:
            row["why"] = "NaN in compiled output only"
        rows.append(row)
        if not ok:
            failed.append(name)
    return {"tol": tol, "checked": len(rows), "failed": failed, "tensors": rows}


def run_scenario(spec: dict, scenario: str, device: str) -> dict:
    result = {"model": spec["name"], "hf_id": spec["hf_id"], "scenario": scenario, "device": device,
              "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
              "env": check.environment_info()}
    train = scenario == "train_forward"
    try:
        dynamo.reset()
        inductor_config.fallback_random = True
        model = _build(spec, device)
        dtype = next(model.parameters()).dtype
        inputs = check.make_inputs(spec, model, scenario, device)
        compiled_model = copy.deepcopy(model)
        ref_copy = copy.deepcopy(model)

        eager = _run(model, inputs, train)
        compiled = _run(torch.compile(compiled_model, backend="inductor"), inputs, train)
        # Reference: the model in fp64. Some kernels have no fp64 version (MoE grouped GEMM) or the model hard-codes
        # fp32 casts that clash with fp64 inputs; then, for bf16/fp16 models, use an fp32 copy, which is still far
        # more precise than the model's own dtype. fp32 models without fp64 compare eager vs compiled directly.
        fp64, fp64_error, reference = None, None, None
        try:
            fp64 = _run(copy.deepcopy(ref_copy).double(), {k: _to64(v) for k, v in inputs.items()}, train)
            reference = "fp64"
        except Exception as e:
            fp64_error = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
            if dtype in (torch.bfloat16, torch.float16):
                to32 = lambda x: x.float() if isinstance(x, torch.Tensor) and x.is_floating_point() else x
                try:
                    fp64 = _run(ref_copy.float(), {k: to32(v) for k, v in inputs.items()}, train)
                    reference = "fp32"
                except Exception as e2:
                    fp64_error += f" | fp32: {type(e2).__name__}: {str(e2).splitlines()[0][:120]}"
        cmp = compare(eager, compiled, fp64, dtype)
        result.update(status="ok", dtype=str(dtype).replace("torch.", ""), fp64_reference=fp64 is not None,
                      reference=reference, fp64_error=fp64_error, accurate=not cmp["failed"], **cmp)
    except Exception as e:
        result.update(status="error", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
    return result


def results_dir(device: str) -> Path:
    return ROOT / ("results_accuracy" if device == "cpu" else f"results_accuracy_{device}")


def summarize() -> Path:
    out = ROOT / "ACCURACY.md"
    L = ["# Accuracy: torch.compile (Inductor) vs eager", "",
         "_Generated by `python -m graph_breaks.accuracy --summarize`. Do not edit by hand._", "",
         "Each model/scenario is run eagerly, compiled with Inductor, and eagerly in fp64. A run fails when the compiled "
         "result is clearly further from fp64 than eager is (`torch._dynamo.utils.same` with `fp64_ref`, as in PyTorch's "
         "benchmarks). Outputs are every floating-point tensor the model returns; train_forward also checks every "
         "parameter gradient. Dropout is off so train mode is deterministic.", ""]
    for device in ("cpu", "cuda"):
        d = results_dir(device)
        runs = [json.loads(f.read_text()) for f in sorted(d.glob("*/*.json"))] if d.exists() else []
        if not runs:
            continue
        ok = [r for r in runs if r["status"] == "ok"]
        bad = [r for r in ok if not r["accurate"]]
        err = [r for r in runs if r["status"] != "ok"]
        L += [f"## {device.upper()}", "",
              f"- Runs: {len(runs)} · accurate: {len(ok) - len(bad)} · **mismatch: {len(bad)}** · errors: {len(err)}",
              f"- Without an fp64 reference (compared eager vs compiled only): {sum(1 for r in ok if not r['fp64_reference'])}", ""]
        if bad:
            L += ["| model | scenario | dtype | failed tensors | worst (compiled vs fp64 / eager vs fp64) |", "|---|---|---|---|---|"]
            for r in bad:
                worst = max((t for t in r["tensors"] if not t["ok"]),
                            key=lambda t: t.get("rmse_compiled_vs_fp64", t.get("rmse_compiled_vs_eager", 0)) /
                            max(t.get("rmse_eager_vs_fp64", 0), 1e-30))
                ratio = (f"{worst.get('rmse_compiled_vs_fp64', float('nan')):.3g} / {worst.get('rmse_eager_vs_fp64', float('nan')):.3g}"
                         if "rmse_eager_vs_fp64" in worst else f"vs eager {worst.get('rmse_compiled_vs_eager', 0):.3g}")
                L.append(f"| {r['model']} | {r['scenario']} | {r['dtype']} | {len(r['failed'])} of {r['checked']} "
                         f"(e.g. `{worst['name']}`) | {ratio} |")
            L.append("")
        if err:
            L += ["Errors:", ""] + [f"- {r['model']} / {r['scenario']}: {r['error'].splitlines()[0][:160]}" for r in err] + [""]
    out.write_text("\n".join(L) + "\n")
    print(f"Wrote {out.relative_to(ROOT)}")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--model", action="append")
    g.add_argument("--all", action="store_true")
    g.add_argument("--summarize", action="store_true")
    p.add_argument("--scenario", action="append", choices=SCENARIOS)
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)
    if args.summarize:
        summarize()
        return 0
    specs = check.load_models()
    if not args.all:
        by_name = {s["name"]: s for s in specs}
        specs = [by_name[m] for m in args.model]
    for spec in specs:
        for sc in spec.get("scenarios", SCENARIOS):
            if args.scenario and sc not in args.scenario:
                continue
            r = run_scenario(spec, sc, args.device)
            out = results_dir(args.device) / spec["name"] / f"{sc}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(r, indent=2) + "\n")
            tag = r["status"] if r["status"] != "ok" else ("ok" if r["accurate"] else f"MISMATCH {len(r['failed'])}/{r['checked']}")
            print(f"[{spec['name']}] {sc} {args.device}: {tag}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
