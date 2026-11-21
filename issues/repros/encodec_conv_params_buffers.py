"""EnCodec / WavTokenizer / Mimi padded conv: stride, kernel_size and padding_total are stored as
nn.Buffer tensors, so `_get_extra_padding_for_conv1d` returns a tensor and `if length <= max_pad`
in `_pad1d` (or F.pad with tensor sizes in Mimi) is data-dependent. Hit once per conv layer."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
import math
import torch
from torch import nn
import torch.nn.functional as F
from _pattern import report


class ConvOriginal(nn.Module):
    def __init__(self, k=7, s=1):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, k, s)
        self.stride = nn.Buffer(torch.tensor(s), persistent=False)
        self.kernel_size = nn.Buffer(torch.tensor(k), persistent=False)
        self.padding_total = nn.Buffer(torch.tensor(k - s), persistent=False)

    def extra_padding(self, x):
        length = x.shape[-1]
        n_frames = (length - self.kernel_size + self.padding_total) / self.stride + 1
        n_frames = torch.ceil(n_frames).to(torch.int64) - 1
        ideal = n_frames * self.stride + self.kernel_size - self.padding_total
        return ideal - length

    def forward(self, x):
        extra = self.extra_padding(x)
        left = self.padding_total // 2
        right = self.padding_total - left
        length = x.shape[-1]
        max_pad = max(left, right)
        if length <= max_pad:                         # tensor comparison
            x = F.pad(x, (0, max_pad - length + 1))
        return self.conv(F.pad(x, (left, right + extra)))


class ConvFixed(nn.Module):
    def __init__(self, k=7, s=1):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, k, s)
        self.stride, self.kernel_size, self.padding_total = s, k, k - s   # plain ints

    def forward(self, x):
        length = x.shape[-1]
        n_frames = math.ceil((length - self.kernel_size + self.padding_total) / self.stride + 1) - 1
        extra = n_frames * self.stride + self.kernel_size - self.padding_total - length
        left = self.padding_total // 2
        right = self.padding_total - left
        if length <= max(left, right):
            x = F.pad(x, (0, max(left, right) - length + 1))
        return self.conv(F.pad(x, (left, right + extra)))


x = torch.randn(2, 1, 400)
m_orig, m_fix = ConvOriginal(), ConvFixed()
m_fix.conv.load_state_dict(m_orig.conv.state_dict())
report("buffers (one conv layer)", m_orig, x)
report("python ints", m_fix, x)
torch.testing.assert_close(m_orig(x), m_fix(x))
print("outputs match")
