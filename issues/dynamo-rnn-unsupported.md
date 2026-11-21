<!-- Draft for: pytorch/pytorch (feature request; label: module: dynamo). Previously marked not-filing in known_issues.yaml (reason kept below). Once filed, put the URL in known_issues.yaml and set status: open -->

# Dynamo graph-breaks on `nn.LSTM` / `nn.GRU` / `nn.RNN` by default (`allow_rnn=False`), which splits audio codecs like EnCodec

### System info

- torch: `2.16.0a0+git57f02f8` (source build, CUDA 12.8)
- transformers: `5.24.0.dev0` (git `b7fe69cb19`)
- Python 3.15, Linux; CPU and CUDA (B200)

### Description

`torch.compile` graph-breaks on recurrent modules unless `torch._dynamo.config.allow_rnn = True`:

```
Dynamo does not support RNN, GRU, or LSTM
```

In Hugging Face models this hits the audio codecs that put an LSTM between conv stacks:

| model | where | seen in |
|---|---|---|
| encodec | `EncodecLSTM` in encoder and decoder | static + dynamic, CPU + CUDA |
| wavtokenizer | LSTM block | static + dynamic, CPU + CUDA |

Each LSTM forces the surrounding module to split into separate graphs (on top of the conv-padding breaks in
`encodec-conv-params-buffers`).

### Why this was marked "not filing" before

It is by design: RNN support in Dynamo is experimental and behind `allow_rnn=True`.

### Ask

Is `allow_rnn` close to being on by default, or is there guidance for library code (e.g. wrapping the LSTM
call in a custom op / `torch.compiler.disable` with a single, clean break)? A pointer in the graph-break
message to `allow_rnn=True` would also help.

### Reproduction

The LSTM has to be called as a submodule (Dynamo decides when it inlines `self.lstm`; a bare call in a
lambda compiles fine):

```python
import torch

class Block(torch.nn.Module):          # conv -> LSTM -> residual, like EncodecLSTM
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(16, 16, 3, padding=1)
        self.lstm = torch.nn.LSTM(16, 16, num_layers=2)

    def forward(self, x):
        x = self.conv(x)
        y, _ = self.lstm(x.permute(2, 0, 1))
        return (y.permute(1, 2, 0) + x).relu()

m, x = Block().eval(), torch.randn(2, 16, 8)
for allow in (False, True):
    torch._dynamo.reset()
    torch._dynamo.config.allow_rnn = allow
    e = torch._dynamo.explain(m)(x)
    print(f"allow_rnn={allow}: graphs={e.graph_count} breaks={e.graph_break_count}")
```

```
allow_rnn=False: graphs=2 breaks=1      Unsupported function call (delayed): ... Dynamo does not support RNN, GRU, or LSTM
allow_rnn=True:  graphs=1 breaks=0
```

In encodec the break is at `modeling_encodec.py:247` (`EncodecLSTM.forward`, called from `:312`).
