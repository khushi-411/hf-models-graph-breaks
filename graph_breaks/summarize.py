"""Aggregate results/<model>/*.json into SUMMARY.md, linked to known_issues.yaml.

    python -m graph_breaks.summarize

SUMMARY.md gets:
  1. A per-model/scenario table; each break is tagged with its known issue id or NEW.
  2. A "Breaks by root cause" section grouping identical breaks across models.

Untracked (NEW) root causes are also printed with a ready-to-paste
known_issues.yaml stub.
"""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
SUMMARY = ROOT / "SUMMARY.md"
KNOWN_ISSUES = ROOT / "known_issues.yaml"

STATUS_ICON = {
    "draft": "📝",
    "open": "🔴",
    "fixed-upstream": "✅",
    "wontfix": "⚪",
    "not-filing": "⚪",
}


def load_known_issues() -> list[dict]:
    if not KNOWN_ISSUES.exists():
        return []
    data = yaml.safe_load(KNOWN_ISSUES.read_text()) or {}
    return data.get("issues") or []


def _normalize(path: str) -> str:
    """Older results kept absolute stdlib paths; show them as python/<module>.py.
    trust_remote_code modules live in the HF modules cache under a revision hash; show them as
    hub/<org>/<repo>/<file>.py so known_issues.yaml can match them across users and revisions.
    `kernels` hub kernels likewise become kernels/<org>/<repo>/<file>.py."""
    m = re.match(r"^.*/transformers_modules/([^/]+)/([^/]+)/(?:[0-9a-f]{7,40}/)?(.+)$", path)
    if m:
        return f"hub/{m.group(1)}/{m.group(2).replace('_hyphen_', '-').replace('_dot_', '.')}/{m.group(3)}"
    # `kernels` hub kernels: .../hub/kernels--<org>--<repo>/snapshots/<rev>/build/<variant>/<file> -> kernels/<org>/<repo>/<file>
    m = re.match(r"^.*/kernels--([^/]+?)--([^/]+)/snapshots/[0-9a-f]+/build/[^/]+/(.+)$", path)
    if m:
        return f"kernels/{m.group(1)}/{m.group(2)}/{m.group(3)}"
    return re.sub(r"^.*/lib/python3\.\d+/", "python/", path)


def display_location(loc: str) -> str:
    """`file:line` with the file shown as _normalize() does (no home dirs / revision hashes)."""
    file, _, line = loc.rpartition(":")
    return f"{_normalize(file)}:{line}" if file else _normalize(loc)


def break_file_and_function(b: dict) -> tuple[str, str | None]:
    """'transformers/x.py:123' -> ('transformers/x.py', func from user_stack)."""
    file = _normalize(b["location"].rsplit(":", 1)[0])
    func = None
    for frame in reversed(b.get("user_stack", [])):
        # frames look like 'transformers/x.py:123 in func'
        loc, _, name = frame.partition(" in ")
        if _normalize(loc) == _normalize(b["location"]):
            func = name
            break
    return file, func


def match_issue(b: dict, issues: list[dict]) -> dict | None:
    file, func = break_file_and_function(b)
    for issue in issues:
        if issue.get("file") and issue["file"] != file:
            continue
        if issue.get("function") and issue["function"] != func:
            continue
        if issue.get("match") and issue["match"] not in b["reason"]:
            continue
        return issue
    return None


def issue_label(issue: dict | None) -> str:
    if issue is None:
        return "**NEW**"
    icon = STATUS_ICON.get(issue.get("status", ""), "")
    target = issue.get("issue") or issue.get("draft")
    label = f"{icon} {issue['id']}".strip()
    return f"[{label}]({target})" if target else label


def esc(text: str) -> str:
    return text.replace("|", "\\|")


def main(argv=None) -> None:
    import argparse
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cpu", help="cpu (results/ -> SUMMARY.md) or cuda (results_cuda/ -> SUMMARY_CUDA.md)")
    p.add_argument("--results-dir", default=None, help="summarize this results folder instead (e.g. results_fulldepth)")
    p.add_argument("--out", default=None, help="summary file to write (default SUMMARY.md / SUMMARY_CUDA.md)")
    args = p.parse_args(argv)
    global RESULTS_DIR, SUMMARY
    if args.device.startswith("cuda"):
        RESULTS_DIR = ROOT / "results_cuda"
        SUMMARY = ROOT / "SUMMARY_CUDA.md"
    if args.results_dir:
        RESULTS_DIR = (ROOT / args.results_dir).resolve()
    if args.out:
        SUMMARY = (ROOT / args.out).resolve()

    issues = load_known_issues()
    rows, envs = [], set()
    # root-cause key -> info about it and which model/scenarios hit it
    causes: "OrderedDict[tuple, dict]" = OrderedDict()

    for model_dir in sorted(p for p in RESULTS_DIR.iterdir() if p.is_dir()):
        for f in sorted(model_dir.glob("*.json")):
            r = json.loads(f.read_text())
            env = r.get("env", {})
            envs.add((env.get("torch"), env.get("transformers"), env.get("transformers_git")))
            model_link = f"[{r['model']}]({RESULTS_DIR.name}/{model_dir.name}/README.md)"

            if r["status"] != "ok":
                rows.append(
                    f"| {model_link} | {r['scenario']} | - | - | - "
                    f"| ERROR: {esc(r['error'].splitlines()[0][:80])} |"
                )
                continue

            cells = []
            for b in r["breaks"]:
                issue = match_issue(b, issues)
                file, func = break_file_and_function(b)
                key = (issue["id"],) if issue else (file, func, b["short_reason"])
                cause = causes.setdefault(key, {
                    "issue": issue,
                    "short_reason": b["short_reason"],
                    "location": display_location(b["location"]),
                    "file": file,
                    "function": func,
                    "doc_url": b.get("doc_url"),
                    "hits": [],
                })
                cause["hits"].append(f"{r['model']}/{r['scenario']}")
                cells.append(f"{issue_label(issue)} {b['short_reason']} (`{display_location(b['location'])}`)")

            rows.append(
                f"| {model_link} | {r['scenario']} | {r['graph_count']} "
                f"| {r['graph_break_count']} | {r['unique_breaks']} "
                f"| {esc('<br>'.join(cells)) if cells else '✅ none'} |"
            )

    new_causes = [c for c in causes.values() if c["issue"] is None]

    lines = [
        "# Graph break summary",
        "",
        "_Generated by `python -m graph_breaks.summarize`. Do not edit by hand._",
        "",
        "Versions used:",
        "",
    ]
    for torch_v, tf_v, tf_git in sorted(envs, key=str):
        lines.append(f"- torch `{torch_v}`, transformers `{tf_v}`" + (f" (git `{tf_git}`)" if tf_git else ""))
    lines += [
        "",
        f"**{len(causes)} root cause(s)**, {len(new_causes)} not yet triaged (NEW). "
        "Issue status: 📝 draft · 🔴 open · ✅ fixed upstream · ⚪ wontfix/not filing.",
        "",
        "## Per model",
        "",
        "| Model | Scenario | Graphs | Breaks | Unique | Breaks (issue · reason @ location) |",
        "|---|---|---|---|---|---|",
        *rows,
        "",
        "## Breaks by root cause",
        "",
        "| Issue | Status | Reason | Location | Affected (model/scenario) |",
        "|---|---|---|---|---|",
    ]
    for c in causes.values():
        issue = c["issue"]
        status = issue.get("status", "?") if issue else "untriaged"
        where = c["location"] + (f" in {c['function']}" if c["function"] else "")
        lines.append(
            f"| {issue_label(issue)} | {status} | {esc(c['short_reason'])} "
            f"| `{where}` | {', '.join(c['hits'])} |"
        )
    if not causes:
        lines.append("| - | - | No graph breaks recorded | - | - |")
    lines.append("")

    SUMMARY.write_text("\n".join(lines))
    print(f"Wrote {SUMMARY.relative_to(ROOT)} ({len(rows)} rows, {len(causes)} root causes)")

    if new_causes:
        print(f"\n{len(new_causes)} untracked root cause(s). Triage them, then add to known_issues.yaml:\n")
        for c in new_causes:
            stub = {
                "id": "TODO-short-id",
                "title": c["short_reason"],
                "file": c["file"],
                **({"function": c["function"]} if c["function"] else {}),
                "match": c["short_reason"][:60],
                "repo": "huggingface/transformers  # or pytorch/pytorch",
                "issue": None,
                "status": "draft",
                "notes": f"Seen in: {', '.join(c['hits'])}"
                         + (f". Docs: {c['doc_url']}" if c["doc_url"] else ""),
            }
            print(yaml.safe_dump([stub], sort_keys=False, width=200))


if __name__ == "__main__":
    main()
