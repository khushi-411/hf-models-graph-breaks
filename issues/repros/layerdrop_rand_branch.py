"""LayerDrop's `if torch.rand([]) < self.layerdrop` breaks the graph in train mode,
even when layerdrop == 0.0 (the default). Also checks that guarding the branch with
`self.layerdrop > 0` gives a single graph (a patched copy is loaded from a temp
dir; the installed transformers is not modified)."""
import importlib.util
import pathlib
import tempfile

import torch
import transformers
from transformers import OPTConfig
from transformers.models.opt import modeling_opt

print(torch.__version__, transformers.__version__)
input_ids = torch.randint(0, 1000, (2, 32))


def check(label, module):
    torch._dynamo.reset()
    model = module.OPTForCausalLM(OPTConfig(num_hidden_layers=2)).train()
    print(f"[{label}] layerdrop:", model.config.layerdrop)
    explanation = torch._dynamo.explain(model)(input_ids=input_ids, labels=input_ids)
    print(f"[{label}] graphs:", explanation.graph_count, "breaks:", explanation.graph_break_count)

    torch._dynamo.reset()
    try:
        torch.compile(model, fullgraph=True)(input_ids=input_ids, labels=input_ids)
        print(f"[{label}] fullgraph OK")
    except Exception as e:
        print(f"[{label}] fullgraph failed:", type(e).__name__, str(e).splitlines()[0])


check("as is", modeling_opt)

# Proposed fix: skip the random draw when LayerDrop is disabled. Load a patched
# copy of modeling_opt.py under its real module name (only in this process).
src = pathlib.Path(modeling_opt.__file__).read_text()
assert src.count("if self.training:\n") == 1
src = src.replace("if self.training:\n", "if self.training and self.layerdrop > 0:\n")
path = pathlib.Path(tempfile.mkdtemp()) / "models" / "opt" / "modeling_opt.py"
path.parent.mkdir(parents=True)
path.write_text(src)
spec = importlib.util.spec_from_file_location(modeling_opt.__name__, path)
patched = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patched)
check("with fix", patched)
