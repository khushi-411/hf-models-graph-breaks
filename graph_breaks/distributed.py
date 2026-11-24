"""torch.compile with distributed training wrappers: DDP and FSDP2 (fully_shard), on 2+ GPUs.

Launch with torchrun (one process per GPU):

    torchrun --nproc_per_node=2 -m graph_breaks.distributed --model inkling2 --mode ddp
    torchrun --nproc_per_node=2 -m graph_breaks.distributed --model inkling2 --mode fsdp2

Each mode trains the model for 3 AdamW steps twice from the same weights and data: eager-wrapped and compiled-wrapped
(`torch.compile(DDP(model))`, where Dynamo's DDPOptimizer splits the graph at gradient-bucket boundaries; and
`fully_shard` per decoder layer + root, then `torch.compile`). Rank 0 compares losses and final parameters and
records errors, graph breaks, recompiles and the number of compiled graphs.

    results_distributed_cuda/<name>/<mode>.json
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import sys
import time
import traceback

import torch
import torch.distributed as dist

from graph_breaks import accuracy, check, dynamic, modes

ROOT = check.ROOT
# fsdp2_traced traces through the FSDP2 hooks (torch._dynamo.config.skip_fsdp_hooks=False). Only on torch builds that
# still have that flag; newer builds always run the hooks under torch._dynamo.disable (_fsdp_common._dynamo_disable).
MODES = ["ddp", "fsdp2"] + (["fsdp2_traced"] if hasattr(torch._dynamo.config, "skip_fsdp_hooks") else [])
STEPS = 3


def _layer_lists(model):
    """The decoder / encoder layer ModuleLists (FSDP units), e.g. model.model.layers, vision blocks."""
    out = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.ModuleList) and len(m) > 1 and name.split(".")[-1] in ("layers", "blocks", "layer", "h"):
            out.append((name, m))
    return out


def _full(p):
    return p.full_tensor() if hasattr(p, "full_tensor") else p


def run(spec, mode, rank, world, device) -> dict:
    modes._fresh()
    # Default FSDP2 + compile skips the FSDP hooks (a graph break per FSDP unit); fsdp2_traced traces through them.
    if hasattr(torch._dynamo.config, "skip_fsdp_hooks"):
        torch._dynamo.config.skip_fsdp_hooks = mode != "fsdp2_traced"
    c: dict = {"name": f"{mode} x{world} GPUs, {STEPS} AdamW steps"}
    try:
        torch.manual_seed(0)
        base = modes._build(spec, device).train()
        inputs = modes._inputs(spec, base, "train_forward", device)
        m_e, m_c = copy.deepcopy(base), copy.deepcopy(base)
        del base
        if mode == "ddp":
            from torch.nn.parallel import DistributedDataParallel as DDP
            kw = dict(device_ids=[rank], find_unused_parameters=False)
            w_e, w_c = DDP(m_e, **kw), DDP(m_c, **kw)
            fwd_c = torch.compile(w_c)
        else:
            from torch.distributed.fsdp import fully_shard
            units = []
            for w in (m_e, m_c):
                # FSDP2 needs one original dtype per unit. Models that keep a few params in fp32 (routers, gates) inside
                # bf16 layers get those submodules as their own units first, as users do.
                main = max({p.dtype for p in w.parameters()}, key=lambda d: sum(p.numel() for p in w.parameters() if p.dtype == d))
                odd = [m for m in w.modules() if any(p.dtype != main for p in m.parameters(recurse=False))]
                for m in odd:
                    fully_shard(m)
                c["fsdp_extra_dtype_units"] = len(odd)
                lists = _layer_lists(w)
                for _, ml in lists:
                    for layer in ml:
                        fully_shard(layer)
                fully_shard(w)
                units.append(sum(len(ml) for _, ml in lists))
            c["fsdp_units"] = units[0]
            w_e, w_c = m_e, m_c
            fwd_c = torch.compile(w_c)
        # GB_OPT=sgd: plain SGD, which doesn't amplify tiny gradient differences the way AdamW's normalization does.
        opt_cls = torch.optim.SGD if os.environ.get("GB_OPT") == "sgd" else torch.optim.AdamW
        c["optimizer"] = opt_cls.__name__
        opt_e = opt_cls(m_e.parameters(), lr=1e-3)
        opt_c = opt_cls(m_c.parameters(), lr=1e-3)
        losses = []
        t0 = time.time()
        with modes.capture_logs() as logs:
            for _ in range(STEPS):
                le = dynamic._loss(w_e(**inputs)); le.backward(); opt_e.step(); opt_e.zero_grad(set_to_none=True)
                lc = dynamic._loss(fwd_c(**inputs)); lc.backward(); opt_c.step(); opt_c.zero_grad(set_to_none=True)
                losses.append([le.item(), lc.item()])
        pe = {n: _full(p).detach() for n, p in m_e.named_parameters()}
        pc = {n: _full(p).detach() for n, p in m_c.named_parameters()}
        cmp = accuracy.compare(pe, pc, None, modes._dtype(m_e))
        c.update(status="ok", seconds=round(time.time() - t0, 1), losses_eager_compiled=losses,
                 accurate=not cmp["failed"], failed=cmp["failed"][:20], graphs=modes._unique_graphs(),
                 worst=sorted(({"name": t["name"], "max_abs_diff": t.get("max_abs_diff"),
                                "rmse": t.get("rmse_compiled_vs_eager"), "ok": t["ok"]} for t in cmp["tensors"]),
                              key=lambda t: -(t["max_abs_diff"] or 0))[:5],
                 **modes._log_summary(logs))
    except Exception as e:
        c.update(modes._error(e))
    return c


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", action="append", required=True)
    p.add_argument("--mode", action="append", choices=["ddp", "fsdp2", "fsdp2_traced"])
    args = p.parse_args(argv)
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    device = f"cuda:{local}"
    dist.init_process_group("nccl", device_id=torch.device(device))
    by_name = {m["name"]: m for m in check.load_models()}
    try:
        for name in args.model:
            for mode in args.mode or MODES:
                if rank == 0:
                    print(f"[{name}] {mode} x{world}", flush=True)
                c = run(by_name[name], mode, local, world, device)
                gathered = [None] * world
                dist.all_gather_object(gathered, {"rank": rank, "status": c.get("status"), "error": c.get("error")})
                if rank == 0:
                    r = {"model": name, "hf_id": by_name[name]["hf_id"], "mode": mode, "device": "cuda",
                         "world_size": world, "status": modes._status([c]),
                         "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                         "env": check.environment_info(), "ranks": gathered, "checks": [c]}
                    out = ROOT / ("results_distributed_cuda" + os.environ.get("GB_RESULTS_SUFFIX", "")) / name / f"{mode}.json"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text(json.dumps(r, indent=2, default=str) + "\n")
                    print(f"    {r['status']}  {modes._cell(r)[:200]}", flush=True)
                dist.barrier()
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
