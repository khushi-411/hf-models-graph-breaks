"""Shared helper for the pattern repros: run torch._dynamo.explain on the
original code pattern and on the proposed fix, and print graphs / breaks."""
import torch


def report(name, fn, *args, **kwargs):
    torch._dynamo.reset()
    e = torch._dynamo.explain(fn)(*args, **kwargs)
    reasons = sorted({str(b.reason).splitlines()[0][:90] for b in e.break_reasons})
    print(f"[{name}] graphs={e.graph_count} breaks={e.graph_break_count}")
    for r in reasons:
        print(f"    - {r}")
    return e.graph_count, e.graph_break_count
