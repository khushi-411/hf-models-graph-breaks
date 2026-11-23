"""ForSequenceClassificationLoss sets `config.problem_type` the first time it runs, so the guard on
`config.problem_type is None` fails on the second call and the model recompiles, whatever the input
size (static shapes too). Seen with ViTForImageClassification in train mode."""
import torch
import transformers
from transformers import ViTConfig, ViTForImageClassification

print(torch.__version__, transformers.__version__)
cfg = ViTConfig(num_hidden_layers=2, hidden_size=64, num_attention_heads=4, intermediate_size=128,
                image_size=32, patch_size=8, num_labels=3)
model = ViTForImageClassification(cfg).train()
print("problem_type before:", model.config.problem_type)

frames = []
def counting_backend(gm, example_inputs):
    frames.append(1)
    return gm.forward

compiled = torch.compile(model, backend=counting_backend)
x, y = torch.randn(2, 3, 32, 32), torch.randint(0, 3, (2,))
for step in range(3):
    compiled(pixel_values=x, labels=y).loss.backward()
    print(f"step {step}: problem_type={model.config.problem_type!r} compiled graphs so far={len(frames)}")
