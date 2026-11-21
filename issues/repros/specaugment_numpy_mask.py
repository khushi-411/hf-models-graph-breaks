"""wav2vec2-family SpecAugment: `_compute_mask_indices` uses numpy RNG (`np.random.rand(1).item()`)
and numpy index building, called from forward in train mode whenever mask_time_prob > 0
(the default, 0.05). Eval mode is a single graph."""
import torch
import transformers
from transformers import Wav2Vec2Config, Wav2Vec2Model

print(torch.__version__, transformers.__version__)
cfg = Wav2Vec2Config(num_hidden_layers=2, hidden_size=64, num_attention_heads=4, intermediate_size=128,
                     conv_dim=(32, 32), conv_stride=(5, 2), conv_kernel=(10, 3), num_feat_extract_layers=2,
                     num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=4)
print("mask_time_prob:", cfg.mask_time_prob)
x = torch.randn(2, 16000)
for mode, prob in [("train", cfg.mask_time_prob), ("train, mask_time_prob=0", 0.0), ("eval", cfg.mask_time_prob)]:
    torch._dynamo.reset()
    cfg.mask_time_prob = prob
    model = Wav2Vec2Model(cfg)
    model.train() if mode.startswith("train") else model.eval()
    e = torch._dynamo.explain(model)(x)
    print(f"[{mode}] graphs={e.graph_count} breaks={e.graph_break_count}",
          sorted({str(b.reason).splitlines()[0][:60] for b in e.break_reasons})[:3])
