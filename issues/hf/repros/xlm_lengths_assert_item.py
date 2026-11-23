"""XLM forward: `assert lengths.max().item() <= slen` (modeling_xlm.py) costs a graph break in eval
and train. lengths is computed from the attention mask when not passed."""
import torch
import transformers
from transformers import XLMConfig, XLMModel

print(torch.__version__, transformers.__version__)
model = XLMModel(XLMConfig(n_layers=2, emb_dim=64, n_heads=4)).eval()
ids = torch.randint(5, 1000, (2, 32))
e = torch._dynamo.explain(model)(input_ids=ids)
print(f"graphs={e.graph_count} breaks={e.graph_break_count}")
for r in sorted({str(b.reason).splitlines()[0][:100] for b in e.break_reasons}):
    print("   -", r)
