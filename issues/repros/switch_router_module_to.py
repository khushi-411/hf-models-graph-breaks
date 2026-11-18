"""SwitchTransformers router (modeling_switch_transformers.py) breaks the graph in eval and train:
`self.classifier = self.classifier.to(self.dtype)` in forward builds new nn.Parameters, and the
expert loop iterates over `(...).nonzero()`."""
import torch
import transformers
from transformers import SwitchTransformersConfig, SwitchTransformersModel

print(torch.__version__, transformers.__version__)
cfg = SwitchTransformersConfig(num_layers=2, num_decoder_layers=2, d_model=64, d_ff=128, d_kv=16, num_heads=4,
                               num_experts=4, encoder_sparse_step=1, decoder_sparse_step=1)
print("router_dtype:", cfg.router_dtype)
model = SwitchTransformersModel(cfg).eval()
ids = torch.randint(0, 1000, (2, 16))
e = torch._dynamo.explain(model)(input_ids=ids, decoder_input_ids=ids)
print(f"graphs={e.graph_count} breaks={e.graph_break_count}")
for r in sorted({str(b.reason).splitlines()[0][:100] for b in e.break_reasons}):
    print("   -", r)
