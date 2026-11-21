"""DeBERTa (v1) `disentangled_att_bias` slices `rel_embeddings` with `att_span`, a 0-d tensor from
`compute_attention_span` (modeling_deberta.py), which costs a `.item()` graph break per layer.
Also checks the proposed fix (return a Python int) on a patched copy of the module."""
import importlib.util, pathlib, tempfile
import torch
import transformers
from transformers import DebertaConfig
from transformers.models.deberta import modeling_deberta

print(torch.__version__, transformers.__version__)
cfg = DebertaConfig(num_hidden_layers=2, hidden_size=64, num_attention_heads=4, intermediate_size=128,
                    relative_attention=True, pos_att_type=["c2p", "p2c"], max_relative_positions=64)
input_ids = torch.randint(0, 1000, (2, 32))


def check(label, module):
    torch._dynamo.reset()
    model = module.DebertaModel(cfg).eval()
    e = torch._dynamo.explain(model)(input_ids=input_ids)
    print(f"[{label}] graphs={e.graph_count} breaks={e.graph_break_count}",
          sorted({str(b.reason).splitlines()[0][:70] for b in e.break_reasons}))


check("as is", modeling_deberta)

src = pathlib.Path(modeling_deberta.__file__).read_text()
old = "    return torch.tensor(min(max(query_layer.size(-2), key_layer.size(-2)), max_relative_positions))"
assert src.count(old) == 1
src = src.replace(old, "    return min(max(query_layer.size(-2), key_layer.size(-2)), max_relative_positions)")
path = pathlib.Path(tempfile.mkdtemp()) / "modeling_deberta.py"
path.write_text(src)
spec = importlib.util.spec_from_file_location(modeling_deberta.__name__, path)
patched = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patched)
check("att_span as int", patched)
