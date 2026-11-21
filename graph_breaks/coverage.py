"""Every graph break in every model, CPU and CUDA, static (check.py) and dynamic (dynamic.py), with its issue.

    python -m graph_breaks.coverage      # writes COVERAGE.md

Reads results/, results_cuda/, results_dynamic/, results_dynamic_cuda/ and matches each break against
known_issues.yaml (same matcher as summarize.py). Breaks that match nothing are listed at the top as
UNMATCHED and make the command exit 1.
"""

from __future__ import annotations

import json
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

from graph_breaks.summarize import display_location, esc, issue_label, load_known_issues, match_issue

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "COVERAGE.md"
SOURCES = OrderedDict([
    (("static", "cpu"), "results"),
    (("static", "cuda"), "results_cuda"),
    (("dynamic", "cpu"), "results_dynamic"),
    (("dynamic", "cuda"), "results_dynamic_cuda"),
])


# Errors (the run crashed, so it has no graph breaks to match) -> cause. First match on error + traceback wins.
# An issue id links to known_issues.yaml; anything else is a non-issue cause shown as text.
ERROR_CAUSES = [
    ("Invalid NaN comparison", "inductor-bounds-truncdiv-nan"),
    ("is not comparable", "inductor-merge-loops-minmax-not-comparable"),
    ("Invalid ranges [0:-0.5", "dynamo-guard-python-mod-value-range"),
    ("at_Half_tmp_acc0_vec", "inductor-cpp-half-reduction-redeclare"),
    # The stored error doesn't keep the generated kernel. Verified only for pythia train (fp16 parallel residual);
    # check the kernel of any other model that lands here before trusting this label.
    ("CppCompileError: C++ compile error", "inductor-cpp-half-reduction-redeclare"),
    ("grad can be implicitly created only for scalar outputs", "harness-dynamic-loss-selection"),
    ("does not require grad and does not have a grad_fn", "harness-dynamic-loss-selection"),
    ("axial_pos_shape", "harness: input sizes don't satisfy Reformer's axial_pos_shape / chunk_length"),
    ("has to be a multiple of least common multiple chunk_length", "harness: input sizes don't satisfy Reformer's axial_pos_shape / chunk_length"),
    ("requires compiling PyTorch with LAPACK", "build: torch built without LAPACK (orthogonal init)"),
    ("requires compiling PyTorch with BLAS", "build: torch built without BLAS"),
    ("InvalidCxxCompiler", "env: no C++ compiler on the node (stale result, rerun pending)"),
    ("Not run to completion: CPU dynamic compile cancelled", "not run: cancelled after a very slow CPU compile, rerun pending (see issues/inductor-cpu-tiling-stride-at-simplify.md, issues/floordiv-simplify-compile-time.md)"),
    ("did not finish within 2 h on CPU","compile time: CPU Inductor compile exceeded the 2 h cap (CPU-only slowdown, see issues/inductor-cpu-tiling-stride-at-simplify.md)"),
]


def _error_cause(r: dict, by_id: dict) -> tuple[str, bool]:
    text = (r.get("error") or "") + "\n" + (r.get("traceback") or "")
    for needle, cause in ERROR_CAUSES:
        if needle in text:
            return (issue_label(by_id[cause]) if cause in by_id else cause), True
    return "**UNEXPLAINED ERROR**", False


def _breaks(r: dict) -> list[dict]:
    out = r.get("breaks") if "breaks" in r else r.get("graph_breaks")
    for b in out or []:
        b.setdefault("short_reason", b["reason"].splitlines()[0])
    return out or []


def _reason(b: dict) -> str:
    # The first line is often a generic wrapper ("Encountered graph break when attempting to trace ...");
    # the specific reason is the next non-empty line.
    lines = [l.strip() for l in b["reason"].splitlines() if l.strip()]
    if len(lines) > 1 and lines[0].startswith(("Encountered graph break", "Graph break encountered", "Failed to handle")):
        return lines[1][:110]
    return lines[0][:110] if lines else ""


CURRENT_TORCH = ""


def main() -> int:
    global CURRENT_TORCH
    from collections import Counter
    seen = Counter(json.loads(f.read_text()).get("env", {}).get("torch")
                   for d in SOURCES.values() for f in (ROOT / d).glob("*/*.json"))
    CURRENT_TORCH = seen.most_common(1)[0][0] if seen else ""
    issues = load_known_issues()
    by_id = {}
    for i in issues:
        by_id.setdefault(i["id"], i)
    unexplained: list[str] = []
    rows: dict[str, list[tuple]] = defaultdict(list)       # model -> rows
    unmatched: dict[tuple, list[str]] = defaultdict(list)
    totals = {k: [0, 0, 0, 0, set()] for k in SOURCES}     # runs, errors, unique breaks, total breaks, torch versions
    issue_hits: dict[str, set] = defaultdict(set)

    for (check, device), d in SOURCES.items():
        for f in sorted((ROOT / d).glob("*/*.json")):
            r = json.loads(f.read_text())
            t = totals[(check, device)]
            t[0] += 1
            t[4].add(r.get("env", {}).get("torch"))
            model, sc = r["model"], r["scenario"]
            if r["status"] != "ok":
                t[1] += 1
                cause, ok = _error_cause(r, by_id)
                if not ok:
                    unexplained.append(f"{model}/{sc} ({check} {device})")
                rows[model].append((check, device, sc, "-", "-",
                                    f"ERROR {cause}: {esc((r.get('error') or '').splitlines()[0][:100])}"))
                continue
            items = []
            bs = _breaks(r)
            t[2] += len(bs)
            t[3] += r.get("graph_break_count", sum(b.get("count", 1) for b in bs)) or 0
            for b in bs:
                i = match_issue(b, issues)
                if i is None:
                    unmatched[(display_location(b["location"]).rsplit(":", 1)[0], _reason(b))].append(f"{model}/{sc} ({check} {device})")
                else:
                    issue_hits[i["id"]].add(f"{model} ({check} {device})")
                items.append(f"{issue_label(i)} x{b.get('count', 1)} {esc(_reason(b))} (`{display_location(b['location'])}`)")
            graphs = r.get("graph_count", r.get("graphs_first_run", "-"))
            torch_v = r.get("env", {}).get("torch") or ""
            if CURRENT_TORCH and torch_v != CURRENT_TORCH:
                graphs = f"{graphs} (older torch `{torch_v}`)"
            nb = r.get("graph_break_count", len(bs))
            extra = ""
            if check == "dynamic" and r.get("recompiled_graphs"):
                extra = f" · {r['recompiled_graphs']} recompile(s)"
            rows[model].append((check, device, sc, graphs, f"{nb}{extra}", "<br>".join(items) or "✅ none"))

    L = ["# Graph-break coverage: every model, CPU and CUDA", "",
         "_Generated by `python -m graph_breaks.coverage`. Do not edit by hand._", "",
         "Every graph break found by `check.py` (static, Dynamo `explain`) and `dynamic.py`",
         "(`torch.compile(dynamic=True)` + Inductor) on CPU and CUDA, matched to `known_issues.yaml`.", "",
         "| check | device | runs | errors | unique breaks | total breaks | torch |", "|---|---|---|---|---|---|---|"]
    for (check, device), t in totals.items():
        L.append(f"| {check} | {device} | {t[0]} | {t[1]} | {t[2]} | {t[3] if check == 'static' else '-'} | "
                 f"{', '.join(sorted(v for v in t[4] if v))} |")
    L += ["", f"**Unmatched breaks: {sum(len(v) for v in unmatched.values())}** "
          f"({len(unmatched)} groups){' — triage these and add them to known_issues.yaml' if unmatched else ''}", ""]
    for (loc, reason), hits in sorted(unmatched.items(), key=lambda kv: -len(kv[1])):
        L.append(f"- **UNMATCHED** `{loc}` {esc(reason)} — {len(hits)} run(s): {', '.join(hits)}")
    L += ["", f"**Unexplained errors: {len(unexplained)}**" + (f" — {', '.join(unexplained)}" if unexplained else "")]
    L += ["", "Caveats: models run with reduced layers (per-layer breaks scale with depth); one input per scenario",
          "(paths it doesn't reach aren't counted); after \"Failed to handle graph break gracefully\" Dynamo runs the",
          "rest of that frame eagerly, so breaks further inside it are not seen. Counts are lower bounds.", "",
          "## Issues and where they occur", "", "| issue | models (check device) |", "|---|---|"]
    for iid, hits in sorted(issue_hits.items(), key=lambda kv: -len(kv[1])):
        L.append(f"| {issue_label(by_id[iid])} | {', '.join(sorted(hits))} |")
    L += ["", "## Per model", ""]
    for model in sorted(rows):
        L += [f"### {model}", "", "| check | device | scenario | graphs | breaks | breaks (issue · reason · location) |",
              "|---|---|---|---|---|---|"]
        order = {k: n for n, k in enumerate(SOURCES)}
        for row in sorted(rows[model], key=lambda x: (order[(x[0], x[1])], x[2])):
            L.append("| " + " | ".join(str(c) for c in row) + " |")
        L.append("")
    OUT.write_text("\n".join(L) + "\n")
    print(f"Wrote {OUT.relative_to(ROOT)}: {len(rows)} models, "
          f"{sum(len(v) for v in unmatched.values())} unmatched break(s) in {len(unmatched)} group(s), "
          f"{len(unexplained)} unexplained error(s)")
    return 1 if unmatched or unexplained else 0


if __name__ == "__main__":
    sys.exit(main())
