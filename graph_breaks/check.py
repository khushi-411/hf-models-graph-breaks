"""Detect torch.compile (Dynamo) graph breaks in Hugging Face transformers models.

For every model in models.yaml and every scenario, this runs
`torch._dynamo.explain` and writes:

    results/<name>/<scenario>.json   machine-readable result
    results/<name>/README.md         human-readable report for that model

Then run `python -m graph_breaks.summarize` to refresh SUMMARY.md.

Examples:
    python -m graph_breaks.check --model gpt2
    python -m graph_breaks.check --all
    python -m graph_breaks.check --model gpt2 --scenario eval_forward --device cuda
"""

from __future__ import annotations

import argparse
import datetime as dt
import inspect
import contextlib
import json
import logging
import platform
import re
import subprocess
import sys
import traceback
from collections import OrderedDict
from pathlib import Path

import torch
import torch._dynamo as dynamo
import yaml

import transformers
from transformers import AutoConfig

ROOT = Path(__file__).resolve().parent.parent
MODELS_FILE = ROOT / "models.yaml"
RESULTS_DIR = ROOT / "results"

ALL_SCENARIOS = ["eval_forward", "eval_with_cache", "train_forward"]
BATCH, SEQ_LEN = 2, 32


# --------------------------------------------------------------------------- #
# Config / environment
# --------------------------------------------------------------------------- #
def load_models() -> list[dict]:
    with open(MODELS_FILE) as f:
        return yaml.safe_load(f)["models"]


def _git_sha(path: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def environment_info() -> dict:
    tf_path = Path(transformers.__file__).resolve().parent
    torch_path = Path(torch.__file__).resolve().parent
    return {
        "torch": torch.__version__,
        "torch_git": getattr(torch.version, "git_version", None),
        "transformers": transformers.__version__,
        # Non-None when transformers is an editable install from a git checkout.
        "transformers_git": _git_sha(tf_path),
        "torch_path": str(torch_path),
        "transformers_path": str(tf_path),
        "python": platform.python_version(),
        "cuda": torch.version.cuda,
    }


# --------------------------------------------------------------------------- #
# Model + inputs
# --------------------------------------------------------------------------- #
def _set_override(config, key: str, value):
    """Set `key` on the config. Dotted keys reach sub-configs, e.g. `text_config.num_hidden_layers`."""
    *parents, attr = key.split(".")
    target = config
    for name in parents:
        target = getattr(target, name)
    # YAML has no tuples; keep the config's type (e.g. Reformer's axial_pos_shape).
    if isinstance(value, list) and isinstance(getattr(target, attr, None), tuple):
        value = tuple(value)
    setattr(target, attr, value)


def build_config(spec: dict):
    config = AutoConfig.from_pretrained(spec["hf_id"])
    if spec.get("use_text_config"):
        # Text-only LM built from a multimodal checkpoint (e.g. gemma3n, llama4).
        config = config.get_text_config()
    for k, v in spec.get("config_overrides", {}).items():
        _set_override(config, k, v)
    # Shrinking vocab_size can leave pad_token_id outside the embedding table.
    for cfg in {id(c): c for c in (config, config.get_text_config())}.values():
        vocab, pad = getattr(cfg, "vocab_size", None), getattr(cfg, "pad_token_id", None)
        if isinstance(vocab, int) and isinstance(pad, int) and pad >= vocab:
            cfg.pad_token_id = None
    return config


def build_model(spec: dict, pretrained: bool, device: str):
    auto_cls = getattr(transformers, spec["auto_class"])
    # Only config.json is downloaded unless --pretrained; weights are random.
    # Graph breaks depend on code paths, not weight values, so this is enough.
    config = build_config(spec)
    if pretrained:
        model = auto_cls.from_pretrained(spec["hf_id"], config=config)
    else:
        model = auto_cls.from_config(config)
    return model.to(device)


def _hw(size) -> tuple[int, int]:
    if isinstance(size, dict):  # e.g. {"height": 224, "width": 224}
        return size["height"], size["width"]
    if isinstance(size, (list, tuple)):
        return tuple(size)
    return size, size


def _text_ids(cfg, device, seq_len: int = SEQ_LEN) -> torch.Tensor:
    text_cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    vocab = getattr(text_cfg, "vocab_size", 1000)
    return torch.randint(0, min(vocab, 1000), (BATCH, seq_len), device=device)


def _image_token_id(cfg) -> int:
    for name in ("image_token_id", "image_token_index"):
        if getattr(cfg, name, None) is not None:
            return getattr(cfg, name)
    raise ValueError("config has no image_token_id")


def _vlm_inputs(spec: dict, model, device: str) -> dict:
    """LLaVA-style VLMs: N image placeholder tokens followed by text.

    N depends on the vision tower and projector (pooling, pixel shuffle, ...), so
    instead of hard-coding it per family we run one eager forward and record how
    many image features the vision tower + projector produce.
    """
    cfg = model.config
    vcfg = cfg.vision_config
    h, w = _hw(spec.get("image_size", getattr(vcfg, "image_size", 224)))
    channels = getattr(vcfg, "num_channels", 3)
    shape = (BATCH, 1, channels, h, w) if spec.get("pixel_ndim") == 5 else (BATCH, channels, h, w)
    pixel_values = torch.randn(*shape, device=device)
    image_token = _image_token_id(cfg)
    text = _text_ids(cfg, device, 16)

    def build(n_image_tokens: int) -> dict:
        ids = torch.cat([torch.full((BATCH, n_image_tokens), image_token, device=device), text], dim=1)
        out = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "pixel_values": pixel_values}
        if spec.get("token_type_ids"):
            out["token_type_ids"] = (ids == image_token).long()
        if spec.get("image_sizes"):
            out["image_sizes"] = torch.tensor([[h, w]] * BATCH, device=device)
        return out

    # LLaVA-style models validate placeholders in `get_placeholder_mask(input_ids,
    # inputs_embeds, image_features)`; intercept it to read the feature count.
    # (Parsing the mismatch error is unreliable: e.g. llava reports numel there.)
    inner = getattr(model, "model", model)
    if not hasattr(inner, "get_placeholder_mask"):
        # e.g. Idefics3/SmolVLM: fall back to the count in the mismatch error.
        with torch.no_grad():
            try:
                model(**build(1))
                return build(1)
            except Exception as e:  # "... tokens: 2, features: 128"
                m = re.search(r"features: (\d+)", str(e))
                if not m:
                    raise
        return build(int(m.group(1)) // BATCH)

    original = inner.get_placeholder_mask
    seen = {}

    class _Probe(Exception):
        pass

    def probe(input_ids=None, inputs_embeds=None, image_features=None, **kw):
        seen["n"] = image_features.numel() // inputs_embeds.shape[-1]
        raise _Probe

    inner.get_placeholder_mask = probe
    try:
        with torch.no_grad():
            model(**build(1))
    except _Probe:
        pass
    finally:
        del inner.get_placeholder_mask  # drop the instance attribute, back to the class method
    assert inner.get_placeholder_mask.__func__ is original.__func__
    return build(seen["n"] // BATCH)


def _qwen_vl_inputs(model, device: str) -> dict:
    """Qwen2-VL family: flattened patches + image_grid_thw, wrapped in vision start/end tokens."""
    cfg = model.config
    vcfg = cfg.vision_config
    grid = 8  # 8x8 patches per image
    patch, temporal, merge = vcfg.patch_size, vcfg.temporal_patch_size, vcfg.spatial_merge_size
    channels = getattr(vcfg, "in_channels", getattr(vcfg, "in_chans", 3))
    pixel_values = torch.randn(BATCH * grid * grid, channels * temporal * patch * patch, device=device)
    n_tokens = grid * grid // (merge * merge)
    text = _text_ids(cfg, device, 16)
    prefix = [cfg.vision_start_token_id] + [cfg.image_token_id] * n_tokens
    if getattr(cfg, "vision_end_token_id", None) is not None:
        prefix.append(cfg.vision_end_token_id)
    ids = torch.cat([torch.tensor([prefix] * BATCH, device=device), text], dim=1)
    out = {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "pixel_values": pixel_values,
        "image_grid_thw": torch.tensor([[1, grid, grid]] * BATCH, device=device),
    }
    if "mm_token_type_ids" in inspect.signature(model.forward).parameters:
        out["mm_token_type_ids"] = (ids == cfg.image_token_id).int()
    return out


# Input kinds whose models take `use_cache`.
_GENERATIVE_KINDS = {"text", "audio_features", "audio_waveform", "vlm", "qwen_vl"}


def make_inputs(spec: dict, model, scenario: str, device: str) -> dict:
    cfg = model.config
    kind = spec.get("inputs", "text")
    is_seq2seq = getattr(cfg, "is_encoder_decoder", False)
    train = scenario == "train_forward"
    inputs: dict = {}

    if kind == "text":
        ids = _text_ids(cfg, device)
        inputs["input_ids"] = ids
        inputs["attention_mask"] = torch.ones_like(ids)
        if train:
            inputs["labels"] = ids.clone()
        elif is_seq2seq:
            # Seq2seq models need decoder inputs when no labels are given
            # (with labels, they are derived by shifting the labels).
            inputs["decoder_input_ids"] = ids.clone()
    elif kind == "image":
        size = spec.get("image_size", getattr(cfg, "image_size", 224))
        h, w = _hw(size)
        channels = getattr(cfg, "num_channels", 3)
        inputs["pixel_values"] = torch.randn(BATCH, channels, h, w, device=device)
        if train and getattr(cfg, "num_labels", None):
            inputs["labels"] = torch.randint(0, cfg.num_labels, (BATCH,), device=device)
    elif kind in ("audio_features", "audio_waveform"):
        if kind == "audio_features":  # log-mel spectrogram (whisper)
            frames = 2 * cfg.max_source_positions
            inputs["input_features"] = torch.randn(BATCH, cfg.num_mel_bins, frames, device=device)
        else:  # raw 1s waveform at 16 kHz (wav2vec2, hubert, wavlm, moonshine)
            inputs["input_values"] = torch.randn(BATCH, 16000, device=device)
        dec = torch.randint(1, min(getattr(cfg, "vocab_size", 32), 1000), (BATCH, 8), device=device)
        if train:
            inputs["labels"] = dec
        elif is_seq2seq:
            inputs["decoder_input_ids"] = dec
    elif kind == "image_text":  # CLIP / SigLIP style dual encoders
        h, w = _hw(getattr(cfg.vision_config, "image_size", 224))
        ids = _text_ids(cfg, device, 16)
        inputs.update(input_ids=ids, pixel_values=torch.randn(BATCH, 3, h, w, device=device))
        if not spec.get("no_attention_mask"):
            inputs["attention_mask"] = torch.ones_like(ids)
        if train:
            inputs["return_loss"] = True
    elif kind == "vlm":
        inputs = _vlm_inputs(spec, model, device)
    elif kind == "qwen_vl":
        inputs = _qwen_vl_inputs(model, device)
    else:
        raise ValueError(f"Unknown inputs kind: {kind}")

    if train and kind in ("vlm", "qwen_vl"):
        inputs["labels"] = inputs["input_ids"].clone()
    if scenario == "eval_forward" and kind in _GENERATIVE_KINDS and (kind != "audio_waveform" or is_seq2seq):
        inputs["use_cache"] = False
    elif scenario == "eval_with_cache":
        inputs["use_cache"] = True
    return inputs


# --------------------------------------------------------------------------- #
# Running explain
# --------------------------------------------------------------------------- #
def _frame_str(frame) -> str:
    # traceback.FrameSummary
    return f"{frame.filename}:{frame.lineno} in {frame.name}"


def _short_path(path: str) -> str:
    """Make paths readable: strip everything before 'transformers/' or 'torch/'."""
    for marker in ("/transformers/", "/torch/"):
        if marker in path:
            return marker.strip("/") + "/" + path.split(marker, 1)[1]
    # Standard library, e.g. .../lib/python3.15/contextlib.py -> python/contextlib.py
    return re.sub(r"^.*/lib/python3\.\d+/", "python/", path)


GRAPH_BREAK_LOGGER = "torch._dynamo.symbolic_convert.__graph_breaks"


@contextlib.contextmanager
def capture_graph_break_logs(echo: bool = False):
    """Collect Dynamo's detailed `graph_breaks` log messages.

    `explain().break_reasons` often only reports the outermost frame with a
    terse reason (e.g. "step_unsupported"). The graph_breaks log artifact
    (what TORCH_LOGS=graph_breaks prints) has the real source line, a full
    explanation and hints, so we capture it as well.
    """
    records: list[str] = []

    class _Handler(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    torch._logging.set_logs(graph_breaks=True)
    logger = logging.getLogger(GRAPH_BREAK_LOGGER)
    handler = _Handler()
    old_propagate = logger.propagate
    logger.addHandler(handler)
    logger.propagate = echo  # don't spam the console unless asked
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.propagate = old_propagate
        torch._logging.set_logs()


_LOC_RE = re.compile(r"Graph break in user code at (.+?):(\d+)")
_FILE_RE = re.compile(r'File "(.+?)", line (\d+), in (.+)')


def parse_graph_break_log(msg: str) -> dict | None:
    m = _LOC_RE.search(msg)
    if not m:
        return None
    location = _short_path(f"{m.group(1)}:{m.group(2)}")
    reason = msg.split("Graph Break Reason:", 1)[-1]
    reason = reason.split("User code traceback:", 1)[0].strip()
    # Keep only model-library frames from the traceback (drop harness/torch.nn).
    stack = []
    if "User code traceback:" in msg:
        for fm in _FILE_RE.finditer(msg.split("User code traceback:", 1)[1]):
            path = fm.group(1)
            if "/transformers/" in path:
                stack.append(_short_path(f"{path}:{fm.group(2)} in {fm.group(3)}"))
    doc = re.search(r"https://\S+/gb/gb\d+\.html", msg)
    return {
        "location": location,
        "reason": reason,
        "user_stack": stack[-10:],
        "doc_url": doc.group(0) if doc else None,
    }


def short_reason(reason: str) -> str:
    """One-line headline, e.g. 'logging.Logger method not supported ...'."""
    lines = [ln.strip() for ln in reason.splitlines() if ln.strip()]
    if len(lines) > 1 and lines[0].startswith("Encountered graph break"):
        return lines[1]
    return lines[0] if lines else ""


def summarize_breaks(log_messages: list[str], break_reasons) -> list[dict]:
    """Group identical breaks (same reason + same source line) and count them.

    Uses the detailed graph_breaks log when available, falling back to
    explain()'s break_reasons. Repeated layers / Dynamo restarts produce the
    same break several times; grouping keeps the report readable.
    """
    parsed = [b for b in (parse_graph_break_log(m) for m in log_messages) if b]
    if not parsed:
        for br in break_reasons:
            stack = list(br.user_stack or [])
            parsed.append({
                "location": _short_path(_frame_str(stack[-1])) if stack else "<unknown>",
                "reason": str(br.reason).strip(),
                "user_stack": [_short_path(_frame_str(f)) for f in stack[-10:]],
                "doc_url": None,
            })

    grouped: "OrderedDict[tuple, dict]" = OrderedDict()
    for b in parsed:
        key = (b["location"], b["reason"])
        if key not in grouped:
            grouped[key] = {**b, "short_reason": short_reason(b["reason"]), "count": 0}
        grouped[key]["count"] += 1
    return list(grouped.values())


def run_scenario(spec: dict, scenario: str, args) -> dict:
    dynamo.reset()
    result: dict = {
        "model": spec["name"],
        "hf_id": spec["hf_id"],
        "auto_class": spec["auto_class"],
        "scenario": scenario,
        "pretrained": args.pretrained,
        "device": args.device,
        "config_overrides": spec.get("config_overrides", {}),
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "env": environment_info(),
    }
    try:
        torch.manual_seed(0)
        model = build_model(spec, args.pretrained, args.device)
        inputs = make_inputs(spec, model, scenario, args.device)

        grad_ctx = contextlib.nullcontext() if scenario == "train_forward" else torch.no_grad()
        model.train(scenario == "train_forward")
        with capture_graph_break_logs(echo=args.verbose) as logs, grad_ctx:
            explanation = dynamo.explain(model)(**inputs)

        breaks = summarize_breaks(logs, explanation.break_reasons)
        result.update(
            status="ok",
            graph_count=explanation.graph_count,
            graph_break_count=explanation.graph_break_count,
            op_count=explanation.op_count,
            unique_breaks=len(breaks),
            breaks=breaks,
        )
    except Exception as e:  # record failures instead of crashing the whole sweep
        result.update(
            status="error",
            error=f"{type(e).__name__}: {e}",
            traceback=traceback.format_exc(),
        )
    return result


# --------------------------------------------------------------------------- #
# Writing reports
# --------------------------------------------------------------------------- #
def write_model_report(spec: dict, results: list[dict]) -> Path:
    out_dir = RESULTS_DIR / spec["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        (out_dir / f"{r['scenario']}.json").write_text(json.dumps(r, indent=2) + "\n")

    # Rebuild README from all scenario JSONs on disk (so partial reruns keep
    # previously recorded scenarios).
    all_results = [json.loads(p.read_text()) for p in sorted(out_dir.glob("*.json"))]
    env = all_results[0]["env"] if all_results else {}

    lines = [
        f"# {spec['name']} — graph breaks",
        "",
        f"- **HF id:** `{spec['hf_id']}`",
        f"- **Class:** `{spec['auto_class']}`",
        f"- **Config overrides:** `{spec.get('config_overrides', {})}`",
        f"- **torch:** `{env.get('torch')}`  |  **transformers:** `{env.get('transformers')}`"
        + (f" (git `{env['transformers_git']}`)" if env.get("transformers_git") else ""),
    ]
    if spec.get("notes"):
        lines.append(f"- **Notes:** {spec['notes']}")
    lines += [
        "",
        "| Scenario | Status | Graphs | Graph breaks | Unique breaks |",
        "|---|---|---|---|---|",
    ]
    for r in all_results:
        if r["status"] == "ok":
            lines.append(
                f"| {r['scenario']} | ok | {r['graph_count']} | "
                f"{r['graph_break_count']} | {r['unique_breaks']} |"
            )
        else:
            lines.append(f"| {r['scenario']} | ERROR | - | - | - |")

    for r in all_results:
        lines += ["", f"## {r['scenario']}", ""]
        if r["status"] != "ok":
            lines += ["**Error while running:**", "", "```", r["error"], "```"]
            continue
        if not r["breaks"]:
            lines.append("No graph breaks. ✅")
            continue
        for i, b in enumerate(r["breaks"], 1):
            lines += [
                f"### {i}. {b['short_reason']}",
                "",
                f"- **Location:** `{b['location']}`  (hit ×{b['count']})",
                "",
                "```",
                b["reason"],
                "```",
                "",
                *([f"Docs: {b['doc_url']}", ""] if b.get("doc_url") else []),
                "<details><summary>User stack</summary>",
                "",
                "```",
                *b["user_stack"],
                "```",
                "</details>",
                "",
            ]
    readme = out_dir / "README.md"
    readme.write_text("\n".join(lines).rstrip() + "\n")
    return readme


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--model", action="append", help="model name from models.yaml (repeatable)")
    g.add_argument("--all", action="store_true", help="run every model in models.yaml")
    g.add_argument("--list", action="store_true", help="list models in models.yaml and exit")
    p.add_argument("--scenario", action="append", choices=ALL_SCENARIOS,
                   help="restrict to these scenarios (repeatable)")
    p.add_argument("--device", default="cpu", help="cpu or cuda (default: cpu)")
    p.add_argument("--pretrained", action="store_true",
                   help="download real weights instead of random init from config")
    p.add_argument("--verbose", action="store_true",
                   help="also print Dynamo's full graph_breaks log to the console")
    args = p.parse_args(argv)

    models = load_models()
    if args.list:
        for m in models:
            print(f"{m['name']:20s} {m['hf_id']}")
        return 0

    by_name = {m["name"]: m for m in models}
    if args.all:
        selected = models
    else:
        missing = [n for n in args.model if n not in by_name]
        if missing:
            p.error(f"unknown model(s): {missing}. Known: {sorted(by_name)}")
        selected = [by_name[n] for n in args.model]

    exit_code = 0
    for spec in selected:
        scenarios = spec.get("scenarios", ALL_SCENARIOS)
        if args.scenario:
            scenarios = [s for s in scenarios if s in args.scenario]
        results = []
        for sc in scenarios:
            print(f"[{spec['name']}] {sc} ...", flush=True)
            r = run_scenario(spec, sc, args)
            if r["status"] == "ok":
                print(f"    graphs={r['graph_count']} breaks={r['graph_break_count']} "
                      f"unique={r['unique_breaks']}")
                for b in r["breaks"]:
                        print(f"    - (x{b['count']}) {b['location']}: {b['short_reason']}")
            else:
                exit_code = 1
                print(f"    ERROR: {r['error']}")
            results.append(r)
        report = write_model_report(spec, results)
        print(f"  -> {report.relative_to(ROOT)}")
    print("\nRun `python -m graph_breaks.summarize` to refresh SUMMARY.md")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
