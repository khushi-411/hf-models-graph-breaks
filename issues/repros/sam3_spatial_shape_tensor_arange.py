"""SAM3 (Sam3Model) graph-breaks in every DETR decoder layer: the encoder stores its feature-map sizes as
`spatial_shapes = torch.tensor(...)` (modeling_sam3.py:1463), the decoder passes `spatial_shapes[0, 0]` /
`[0, 1]` (0-d tensors) to `_get_coords`, and `torch.arange(0, height)` with a tensor bound needs `.item()`
(aten._local_scalar_dense) -> "Data dependent operator" at modeling_sam3.py:1719 / 1720.
Small config, random weights, image + text prompt."""
import torch
import transformers
from transformers import Sam3Config, Sam3Model

print(torch.__version__, transformers.__version__)
cfg = Sam3Config()
cfg.vision_config.backbone_config.num_hidden_layers = 2
cfg.text_config.num_hidden_layers = 2
cfg.geometry_encoder_config.num_layers = 1
cfg.detr_encoder_config.num_layers = 1
for n_dec in (1, 3):
    cfg.detr_decoder_config.num_layers = n_dec
    model = Sam3Model(cfg).eval()
    size = cfg.vision_config.backbone_config.image_size
    pixel_values = torch.randn(1, 3, size, size)
    input_ids = torch.randint(1, 1000, (1, 16))
    torch._dynamo.reset()
    e = torch._dynamo.explain(model)(pixel_values=pixel_values, input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
    sites = sorted({str(b.user_stack[-1].filename).split("transformers/")[-1] + f":{b.user_stack[-1].lineno}"
                    for b in e.break_reasons if b.user_stack})
    print(f"decoder layers={n_dec}: graphs={e.graph_count} breaks={e.graph_break_count} sites={sites}")
