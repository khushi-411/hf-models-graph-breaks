import torch
import transformers
from transformers import GPT2Config, GPT2LMHeadModel

print(torch.__version__, transformers.__version__)
model = GPT2LMHeadModel(GPT2Config(n_layer=2)).train()
print("loss_type:", model.loss_type)
input_ids = torch.randint(0, 1000, (2, 32))

explanation = torch._dynamo.explain(model)(input_ids=input_ids, labels=input_ids)
print("graph breaks:", explanation.graph_break_count)

torch._dynamo.reset()
compiled = torch.compile(model, fullgraph=True)
try:
    compiled(input_ids=input_ids, labels=input_ids)
    print("fullgraph OK")
except Exception as e:
    print("fullgraph failed:", type(e).__name__, str(e).splitlines()[0])
