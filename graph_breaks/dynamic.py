"""Find dynamic-shape problems (recompiles / specializations) in HF models under torch.compile + Inductor.

`check.py` only runs Dynamo (`dynamo.explain`), so it never sees what Inductor
does with symbolic sizes. Many Inductor lowerings still take `sympy.Expr`
sizes and call `sizevars.guard_int` / `guard_equals` / `evaluate_expr` on them
(pytorch/pytorch#137093). On a dynamic dim this *specializes* it: a guard like
`Eq(s0, 40)` is installed and the next input size recompiles.

For every model this:
  1. compiles it with `torch.compile(backend="inductor", dynamic=True)`
  2. runs it on several input sizes (batch / sequence / audio length change)
  3. records the number of compiled graphs, the recompiles and their reasons
  4. records every guard and specialization added *while Inductor ran*,
     attributed to the Inductor function that added it (e.g. lowering.py:glu)

A model that recompiles because of an Inductor-side specialization at
`lowering.py:<op>` is a model that porting `<op>` to SymInt
(as in pytorch/pytorch#201740 for glu / split) would fix.

    results_dynamic/<name>/<scenario>.json
    DYNAMIC_SUMMARY.md   (python -m graph_breaks.dynamic --summarize)

Examples:
    python -m graph_breaks.dynamic --model gpt2
    python -m graph_breaks.dynamic --model gpt2 --scenario train_forward
    python -m graph_breaks.dynamic --summarize
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import logging
import sys
import traceback
from collections import Counter, OrderedDict
from pathlib import Path

import sympy
import torch
import torch._dynamo as dynamo
from torch._dynamo.testing import CompileCounterWithBackend
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from graph_breaks import check

ROOT = check.ROOT
OUT_DIR = ROOT / "results_dynamic"
SUMMARY_FILE = ROOT / "DYNAMIC_SUMMARY.md"
SCENARIOS = ["eval_forward", "train_forward"]

# (batch, seq_len, audio_samples) per run. Non-power-of-two and not multiples of
# each other, so a specialization on one of them shows up as a recompile.
SIZES = [(2, 32, 16000), (3, 40, 17600), (4, 56, 19200)]

RECOMPILE_LOGGER = "torch._dynamo.guards.__recompiles"

# A warm FX-graph / AOT-autograd cache replays saved guards instead of re-running the
# lowerings, which would hide where each guard came from.
torch._inductor.config.force_disable_caches = True

# Frames that are guard machinery rather than the code that asked for the guard.
_INFRA = (
    "torch/fx/",
    "torch/_compile.py",
    "torch/_utils_internal.py",
    "torch/utils/_stats.py",
    "torch/_subclasses/functional_tensor.py",
    "torch/_inductor/sizevars.py",
    "torch/_inductor/virtualized.py",
    "torch/utils/_sympy/",
    "torch/__init__.py",
    "torch/_dynamo/",
    "torch/_logging/",
    "torch/_guards.py",
    "torch/utils/_contextlib.py",
    "python/",
    "<",  # generated code (e.g. <string>, <eval_with_key>)
)


def _site(frame) -> str:
    return f"{check._short_path(frame.filename)}:{frame.name}"


def attribute(stack) -> dict:
    """Which component asked for this guard: inductor, aot/decomp, or the model (dynamo trace).

    Classified by the innermost frame that isn't guard machinery. (Note AOT tracing
    itself runs inside inductor's compile_fx, so "inductor is on the stack" isn't enough.)
    """
    frames = [(f, check._short_path(f.filename)) for f in stack]
    culprit = next(((f, p) for f, p in reversed(frames) if not p.startswith(_INFRA)), None)
    model_frame = next((f for f, p in reversed(frames) if p.startswith("transformers/")), None)
    if culprit and culprit[1].startswith("torch/_inductor/"):
        phase = "inductor"
    elif culprit and culprit[1].startswith(("transformers/", "torch/nn/modules/")):
        phase = "model"
    else:
        phase = "aot/decomp"
    if culprit is None:
        site, line = "<unknown>", 0
    else:
        site, line = _site(culprit[0]), culprit[0].lineno
    return {
        "phase": phase,
        "site": site,
        "line": line,
        "model_frame": _site(model_frame) + f":{model_frame.lineno}" if model_frame else None,
    }


@contextlib.contextmanager
def record_shape_env_events():
    """Patch ShapeEnv to log every guard and every symbol -> constant specialization, with attribution."""
    events: list[dict] = []
    orig_log_guard = ShapeEnv._log_guard
    orig_set_repl = ShapeEnv._set_replacement

    def _log_guard(self, prefix, g, forcing_spec):
        if prefix in ("eval", "runtime_assert"):
            events.append({"kind": "guard" if prefix == "eval" else "runtime_assert",
                           "expr": str(g), "forcing_spec": bool(forcing_spec),
                           **attribute(traceback.extract_stack()[:-1])})
        return orig_log_guard(self, prefix, g, forcing_spec)

    def _set_replacement(self, a, tgt, msg):
        before = self.replacements.get(a)
        out = orig_set_repl(self, a, tgt, msg)
        after = self.replacements.get(a)
        if after is not None and after != before and isinstance(after, sympy.Expr) and after.is_number \
                and not str(a).startswith(("u", "zuf", "zf")):  # unbacked symbols are expected to resolve
            events.append({"kind": "specialize", "expr": f"{a} = {after}", "msg": msg,
                           **attribute(traceback.extract_stack()[:-1])})
        return out

    ShapeEnv._log_guard = _log_guard
    ShapeEnv._set_replacement = _set_replacement
    try:
        yield events
    finally:
        ShapeEnv._log_guard = orig_log_guard
        ShapeEnv._set_replacement = orig_set_repl


@contextlib.contextmanager
def capture_logger(name: str, _set_logs: bool = True, **set_logs_kwargs):
    records: list[str] = []

    class _H(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    if _set_logs:
        torch._logging.set_logs(**set_logs_kwargs)
    logger = logging.getLogger(name)
    h = _H()
    old = logger.propagate
    logger.addHandler(h)
    logger.propagate = False
    try:
        yield records
    finally:
        logger.removeHandler(h)
        logger.propagate = old
        if _set_logs:
            torch._logging.set_logs()


@contextlib.contextmanager
def input_sizes(batch: int, seq: int, audio: int):
    old = check.BATCH, check.SEQ_LEN, check.AUDIO_SAMPLES
    check.BATCH, check.SEQ_LEN, check.AUDIO_SAMPLES = batch, seq, audio
    try:
        yield
    finally:
        check.BATCH, check.SEQ_LEN, check.AUDIO_SAMPLES = old


def _loss(out):
    loss = getattr(out, "loss", None)
    if loss is None:
        first = out[0] if isinstance(out, (tuple, list)) else next(
            v for v in out.values() if isinstance(v, torch.Tensor))
        loss = first.float().sum()
    return loss


def _group(events: list[dict]) -> list[dict]:
    g: "OrderedDict[tuple, dict]" = OrderedDict()
    for e in events:
        key = (e["kind"], e["phase"], e["site"], e["line"])
        if key not in g:
            g[key] = {"kind": e["kind"], "phase": e["phase"], "site": e["site"], "line": e["line"],
                      "count": 0, "examples": [], "model_frames": []}
        d = g[key]
        d["count"] += 1
        if e["expr"] not in d["examples"] and len(d["examples"]) < 5:
            d["examples"].append(e["expr"])
        if e.get("model_frame") and e["model_frame"] not in d["model_frames"] and len(d["model_frames"]) < 5:
            d["model_frames"].append(e["model_frame"])
    return list(g.values())


def run_scenario(spec: dict, scenario: str, args) -> dict:
    dynamo.reset()
    result: dict = {
        "model": spec["name"], "hf_id": spec["hf_id"], "scenario": scenario,
        "device": args.device, "dynamic": True, "backend": "inductor",
        "unbacked": dynamo.config.capture_dynamic_output_shape_ops,
        "sizes": [list(s) for s in SIZES],
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "env": check.environment_info(),
    }
    train = scenario == "train_forward"
    runs = []
    try:
        torch.manual_seed(0)
        model = check.build_model(spec, False, args.device)
        model.train(train)
        # HF's ForSequenceClassificationLoss sets config.problem_type on the first call when it
        # is None, so the 2nd call fails the `problem_type is None` guard and recompiles
        # regardless of shapes. That's HF behavior, not a dynamic-shape issue: pre-set it.
        if train and getattr(model.config, "problem_type", "unset") is None:
            model.config.problem_type = (
                "regression" if getattr(model.config, "num_labels", 2) == 1 else "single_label_classification"
            )
        cnt = CompileCounterWithBackend("inductor")
        compiled = torch.compile(model, backend=cnt, dynamic=True)
        grad_ctx = contextlib.nullcontext if train else torch.no_grad
        with record_shape_env_events() as events, \
                capture_logger(RECOMPILE_LOGGER, recompiles=True, graph_breaks=True) as recompiles, \
                capture_logger(check.GRAPH_BREAK_LOGGER, _set_logs=False) as break_logs:
            for i, (b, s, a) in enumerate(SIZES):
                with input_sizes(b, s, a):
                    # Inputs are built (and VLM placeholder counts probed) eagerly, outside the counters.
                    inputs = check.make_inputs(spec, model, scenario, args.device)
                n_events = len(events)
                before = cnt.frame_count
                with grad_ctx():
                    out = compiled(**inputs)
                    if train:
                        _loss(out).backward()
                        model.zero_grad(set_to_none=True)
                runs.append({"batch": b, "seq": s, "audio": a,
                             "new_graphs": cnt.frame_count - before,
                             "new_shape_events": len(events) - n_events})
        first = runs[0]["new_graphs"]
        recompiled_graphs = sum(r["new_graphs"] for r in runs[1:])
        grouped = _group(events)
        ind_spec = [e for e in grouped if e["phase"] == "inductor" and e["kind"] == "specialize"]
        ind_guard = [e for e in grouped if e["phase"] == "inductor" and e["kind"] == "guard"]
        result.update(
            status="ok",
            graphs_first_run=first,
            recompiled_graphs=recompiled_graphs,
            runs=runs,
            recompile_reasons=list(OrderedDict.fromkeys(r.strip() for r in recompiles))[:20],
            inductor_specialization_sites=sorted({e["site"] for e in ind_spec}),
            inductor_guard_sites=sorted({e["site"] for e in ind_guard}),
            events=grouped,
            # Every graph break seen under dynamic shapes (all sizes), same format as check.py.
            graph_breaks=check.summarize_breaks(break_logs, []),
        )
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)
        # Innermost Inductor frame that isn't guard machinery: which lowering/IR code failed.
        ind = [f for f in tb if check._short_path(f.filename).startswith("torch/_inductor/")
               and not check._short_path(f.filename).startswith(_INFRA)]
        result.update(status="error", runs=runs, error=f"{type(e).__name__}: {str(e)[:2000]}",
                      error_inductor_site=f"{_site(ind[-1])}:{ind[-1].lineno}" if ind else None,
                      traceback=traceback.format_exc()[-6000:])
    return result


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def _esc(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ")


def summarize() -> Path:
    results = [json.loads(p.read_text()) for p in sorted(OUT_DIR.glob("*/*.json"))]
    ok = [r for r in results if r["status"] == "ok"]
    err = [r for r in results if r["status"] != "error" and False] + [r for r in results if r["status"] == "error"]
    env = results[0]["env"] if results else {}

    # Which inductor sites specialize, and in which models.
    spec_sites: dict[str, set] = {}
    spec_sites_recompile: dict[str, set] = {}
    guard_sites: dict[str, set] = {}
    for r in ok:
        key = f"{r['model']}/{r['scenario']}"
        for e in r["events"]:
            if e["phase"] != "inductor":
                continue
            target = spec_sites if e["kind"] == "specialize" else guard_sites if e["kind"] == "guard" else None
            if target is None:
                continue
            target.setdefault(e["site"], set()).add(key)
            if e["kind"] == "specialize" and r["recompiled_graphs"]:
                spec_sites_recompile.setdefault(e["site"], set()).add(key)

    recompiling = [r for r in ok if r["recompiled_graphs"]]
    ind_recompiling = [r for r in recompiling if r["inductor_specialization_sites"]]
    models = lambda rs: len({r["model"] for r in rs})

    L = [
        "# Dynamic shapes: recompiles and Inductor specializations",
        "",
        "Generated by `python -m graph_breaks.dynamic --summarize`. Context: "
        "[pytorch/pytorch#137093](https://github.com/pytorch/pytorch/issues/137093) "
        "(use SymInt instead of sympy.Expr in Inductor).",
        "",
        f"`torch.compile(model, backend='inductor', dynamic=True)`, run on input sizes "
        f"(batch, seq_len, audio_samples) = {', '.join(str(tuple(s)) for s in SIZES)}. "
        f"torch `{env.get('torch')}`, transformers `{env.get('transformers')}`, device `{results[0]['device'] if results else '?'}`.",
        "",
        "- **Recompiled graphs**: graphs compiled on the 2nd/3rd size. 0 means dynamic shapes worked.",
        "- **Inductor specialization**: Inductor replaced a dynamic size symbol with a constant "
        "(e.g. `guard_int` in a lowering). This installs a guard, so the next size recompiles.",
        "- **Inductor guard**: a (non-specializing) guard added during Inductor lowering/codegen.",
        "- Sites are the innermost `torch/_inductor` frame that asked for the guard.",
        "",
        "## Totals",
        "",
        f"- Runs: {len(results)} ({models(results)} models); ok {len(ok)}, errors {len(err)}",
        f"- Runs that recompile: **{len(recompiling)}** ({models(recompiling)} models)",
        f"- ...of which Inductor specialized a symbol: **{len(ind_recompiling)}** ({models(ind_recompiling)} models)",
        f"- Runs with any Inductor specialization: {len([r for r in ok if r['inductor_specialization_sites']])}",
        "",
        "## Inductor specialization sites (what #137093-style ports would fix)",
        "",
        "| Site | Runs | Runs that recompile | Models |",
        "|---|---|---|---|",
    ]
    for site, keys in sorted(spec_sites.items(), key=lambda kv: -len(kv[1])):
        rec = spec_sites_recompile.get(site, set())
        names = sorted({k.split("/")[0] for k in keys})
        L.append(f"| `{site}` | {len(keys)} | {len(rec)} | {', '.join(names)} |")
    if not spec_sites:
        L.append("| (none) | | | |")

    L += ["", "## Inductor guard sites (non-specializing)", "", "| Site | Runs | Models |", "|---|---|---|"]
    for site, keys in sorted(guard_sites.items(), key=lambda kv: -len(kv[1])):
        names = sorted({k.split("/")[0] for k in keys})
        shown = ", ".join(names[:25]) + (f", … (+{len(names) - 25})" if len(names) > 25 else "")
        L.append(f"| `{site}` | {len(keys)} | {shown} |")

    L += ["", "## Per model", "",
          "| Model | Scenario | Graphs (1st size) | Recompiled graphs | Inductor specializations | Other specializations | Recompile reason (first) |",
          "|---|---|---|---|---|---|---|"]
    for r in results:
        if r["status"] != "ok":
            L.append(f"| {r['model']} | {r['scenario']} | ERROR | | | | {_esc(r['error'][:150])} |")
            continue
        other = sorted({f"{e['phase']}: {e['site']}" for e in r["events"]
                        if e["kind"] == "specialize" and e["phase"] != "inductor"})
        reason = r["recompile_reasons"][0] if r["recompile_reasons"] else ""
        reason = reason.split("triggered by the following guard failure(s):")[-1].strip()
        L.append(
            f"| {r['model']} | {r['scenario']} | {r['graphs_first_run']} | {r['recompiled_graphs']} | "
            f"{'<br>'.join(f'`{s}`' for s in r['inductor_specialization_sites'])} | "
            f"{'<br>'.join(f'`{s}`' for s in other[:4])}{' …' if len(other) > 4 else ''} | {_esc(reason[:200])} |"
        )
    SUMMARY_FILE.write_text("\n".join(L) + "\n")
    return SUMMARY_FILE


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--model", action="append")
    g.add_argument("--all", action="store_true")
    g.add_argument("--summarize", action="store_true", help="regenerate DYNAMIC_SUMMARY.md and exit")
    p.add_argument("--scenario", action="append", choices=SCENARIOS)
    p.add_argument("--device", default="cpu")
    p.add_argument("--unbacked", action="store_true",
                   help="capture .item()/nonzero()-style ops into the graph (unbacked SymInts reach "
                        "Inductor) and write to results_dynamic_unbacked/")
    args = p.parse_args(argv)
    if args.unbacked:
        global OUT_DIR, SUMMARY_FILE
        OUT_DIR = ROOT / "results_dynamic_unbacked"
        SUMMARY_FILE = ROOT / "DYNAMIC_UNBACKED_SUMMARY.md"
        dynamo.config.capture_scalar_outputs = True
        dynamo.config.capture_dynamic_output_shape_ops = True

    if args.summarize:
        print(summarize().relative_to(ROOT))
        return 0

    models = check.load_models()
    by_name = {m["name"]: m for m in models}
    selected = models if args.all else [by_name[n] for n in args.model]
    code = 0
    for spec in selected:
        scenarios = [s for s in spec.get("scenarios", check.ALL_SCENARIOS) if s in SCENARIOS]
        if args.scenario:
            scenarios = [s for s in scenarios if s in args.scenario]
        for sc in scenarios:
            print(f"[{spec['name']}] {sc} (dynamic, inductor) ...", flush=True)
            r = run_scenario(spec, sc, args)
            out = OUT_DIR / spec["name"] / f"{sc}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(r, indent=2) + "\n")
            if r["status"] == "ok":
                print(f"    graphs={r['graphs_first_run']} recompiled={r['recompiled_graphs']} "
                      f"inductor_specializations={r['inductor_specialization_sites']}")
                for reason in r["recompile_reasons"][:3]:
                    print("    recompile:", reason.splitlines()[-1].strip()[:200])
            else:
                code = 1
                print(f"    ERROR: {r['error'][:300]}")
    return code


if __name__ == "__main__":
    sys.exit(main())
