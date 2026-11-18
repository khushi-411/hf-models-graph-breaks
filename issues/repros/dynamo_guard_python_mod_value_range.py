"""InternVL-style pixel shuffle under torch.compile(dynamic=True): `height % scale_factor` with a float
scale_factor (config.downsample_ratio = 0.5), where height = int(sqrt(num_patches)) of a dynamic image
size. Guard building crashes in value_ranges.python_mod with `ValueRangeError: Invalid ranges [0:-0.5]`.
Code copied from transformers modeling_internvl.py (get_image_features + pixel_shuffle)."""
import types
import torch
from torch import nn

print(torch.__version__)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(downsample_ratio=0.5)
        self.patch = nn.Conv2d(3, 32, 14, 14)

    def pixel_shuffle(self, vision_features, scale_factor: float = 0.5):
        batch_size, width, height, channels = vision_features.size()
        if height % scale_factor != 0 or width % scale_factor != 0:
            raise ValueError("Height and width must be divisible by scale_factor for proper downsampling.")
        vision_features = vision_features.view(batch_size, width, int(height * scale_factor), int(channels / scale_factor))
        vision_features = vision_features.permute(0, 2, 1, 3).contiguous()
        vision_features = vision_features.view(
            batch_size, int(height * scale_factor), int(width * scale_factor), int(channels / (scale_factor**2))
        )
        return vision_features.permute(0, 2, 1, 3).contiguous()

    def forward(self, pixel_values):
        downsample_ratio = self.config.downsample_ratio
        vision_features = self.patch(pixel_values).flatten(2).transpose(1, 2).contiguous()   # (B, (H/14)*(W/14), C)
        channels = vision_features.shape[1]
        feature_size = int(channels**0.5)
        batch_size = vision_features.shape[0]
        vision_features = vision_features.reshape(batch_size, feature_size, feature_size, -1)
        vision_features = self.pixel_shuffle(vision_features, scale_factor=downsample_ratio)
        return vision_features.reshape(batch_size, -1, vision_features.shape[-1])


m = Model()
for backend in ("eager", "inductor"):
    torch._dynamo.reset()
    try:
        out = torch.compile(m, backend=backend, dynamic=True)(torch.randn(2, 3, 448, 448))
        print(f"[dynamic=True, backend={backend}] OK {tuple(out.shape)}")
    except Exception as e:
        print(f"[dynamic=True, backend={backend}] {type(e).__name__}: {str(e).splitlines()[0][:110]}")
torch._dynamo.reset()
out = torch.compile(m, backend="eager")(torch.randn(2, 3, 448, 448))
print(f"[static] OK {tuple(out.shape)}")
