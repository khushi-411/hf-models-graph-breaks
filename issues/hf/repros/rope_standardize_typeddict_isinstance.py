"""PhiMoE LongRoPE under torch.compile(dynamic=True): the first long-context update calls the rope init
function, which calls `config.standardize_rope_params()` inside forward; assigning `config.rope_parameters`
runs huggingface_hub strict-dataclass validation, which does `isinstance(dict, RopeParameters)` on a
TypedDict -> graph break gb1583 ("builtin isinstance() with classinfo that does not support issubclass()")."""
import torch
import transformers
from transformers import PhimoeConfig
from transformers.models.phimoe.modeling_phimoe import PhimoeRotaryEmbedding

print(torch.__version__, transformers.__version__)
cfg = PhimoeConfig.from_pretrained("microsoft/Phi-3.5-MoE-instruct")   # config.json only (LongRoPE)
print("rope_type:", cfg.rope_parameters.get("rope_type"))


def set_params(x):
    cfg.rope_parameters = dict(cfg.rope_parameters)     # what standardize_rope_params ends with
    return x + 1


def report(label, fn, *args, **kw):
    torch._dynamo.reset()
    e = torch._dynamo.explain(fn, **kw)(*args)
    print(f"[{label}] graphs={e.graph_count} breaks={e.graph_break_count}")
    for r in sorted({str(b.reason).replace("\n", " ")[:150] for b in e.break_reasons}):
        print("    -", r)


report("assign config.rope_parameters", set_params, torch.randn(4))
rope = PhimoeRotaryEmbedding(cfg)
x = torch.randn(2, 32, cfg.hidden_size)
pos = torch.arange(32)[None].expand(2, -1)
report("rotary embedding, static", rope, x, pos)
torch._dynamo.reset()
cnt = []
f = torch.compile(rope, dynamic=True, backend=lambda gm, ex: (cnt.append(1), gm.forward)[1])
for s in (32, 40):
    f(torch.randn(2, s, cfg.hidden_size), torch.arange(s)[None].expand(2, -1))
print("[rotary embedding, dynamic=True] compiled graphs:", len(cnt))
