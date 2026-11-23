<!-- Draft for: huggingface/transformers. Once filed, put the URL in known_issues.yaml and set status: open -->

# `torch.compile(dynamic=True)`: Python `for` loops over a sequence-length-derived `range` (MiniMax lightning attention, RWKV CPU path) recompile for every new length

### System info

- transformers: `5.24.0.dev0` (main, git `b7fe69cb19`)
- torch: `2.16.0a0+git57f02f8` (source build)
- Python 3.15, Linux; CPU and CUDA (B200)

### Description

Dynamo unrolls Python loops, so `for i in range(n)` with a symbolic `n` must guard on `n`'s exact value.
Two models loop over the sequence:

**MiniMax** lightning attention, block by block (`modeling_minimax.py:214`):

```python
num_blocks = (seq_len + self.block_size - 1) // self.block_size
for i in range(num_blocks):
    ...
```

```
- 0/0: ((15 + kwargs['input_ids'].size()[1]) // 16) == 2  # for i in range(num_blocks):  # modeling_minimax.py:214
```

**RWKV** CPU/fallback WKV kernel, token by token (`modeling_rwkv.py:180`, `rwkv_linear_attention_cpu`):

```python
for current_index in range(seq_length):
    ...
```

This path is also taken on CUDA when the `kernels-community/rwkv` kernel isn't loaded, which is the
default without the `kernels` package.

`torch.compile(dynamic=True)`, Inductor, seq 32 / 40 / 56. Graphs per size are 2 / 1 / 1 for both models,
in eval and train, on CPU and CUDA: every new length recompiles. For RWKV the unrolled graph also grows
linearly with sequence length.

### Possible fix

- MiniMax: pad to a multiple of `block_size`, reshape to `(…, num_blocks, block_size, …)`, compute the
  intra-block part batched over blocks, and carry the inter-block state (`attn_weights_inter`) with
  `torch._higher_order_ops.scan` (or a cumulative-product formulation). The loop then no longer depends
  on the block count.
- RWKV: express the recurrence with `torch._higher_order_ops.scan` over the time dim. Alternatively,
  document that compiling the CPU path specializes on sequence length.

Both are recurrences, so this is a larger change than the one-line fixes in other drafts. Worth filing
mainly so the behavior is known.

<!-- affected-models:manual -->
### Affected models

All runs use `torch.compile(backend="inductor", dynamic=True)` over (batch, seq, audio samples) =
(2, 32, 16000), (3, 40, 17600), (4, 56, 19200), on a model built from the checkpoint's config with reduced
layers. "Graphs per size" is the number of graphs compiled on the 1st / 2nd / 3rd size (anything after the
1st size is a recompile).

| model | config from | scenario · device | graphs per size |
|---|---|---|---|
| minimax | `MiniMaxAI/MiniMax-Text-01-hf` | eval · CPU / CUDA | 2 / 1 / 1 · 2 / 1 / 1 |
| minimax | | train · CPU / CUDA | 2 / 1 / 1 · 2 / 1 / 1 |
| rwkv | `RWKV/rwkv-4-169m-pile` | eval · CPU / CUDA | 2 / 1 / 1 · 2 / 1 / 1 |
| rwkv | | train · CPU / CUDA | 2 / 1 / 1 · 2 / 1 / 1 |

rwkv's CUDA runs take the same Python loop, because the `kernels-community/rwkv` kernel isn't installed.
