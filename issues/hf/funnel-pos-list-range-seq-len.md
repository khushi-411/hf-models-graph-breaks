<!-- Draft for: huggingface/transformers. Once filed, put the URL in known_issues.yaml and set status: open -->

# `torch.compile(dynamic=True)`: Funnel relative-shift attention builds positions with `list(range(seq_len))`, specializing the sequence length

### System info

- transformers: `5.24.0.dev0` (main, git `b7fe69cb19`)
- torch: `2.16.0a0+git57f02f8` (source build)
- Python 3.15, Linux; CPU and CUDA (B200)

### Description

`FunnelAttentionStructure.get_position_embeds` (relative-shift attention, the default) turns the sequence
length into a Python list (`modeling_funnel.py:152`):

```python
# Python ints, not tensors, so the relative-position `arange` sizes stay static for export
pos = list(range(seq_len))
pooled_pos = pos
for block_index in range(0, self.config.num_blocks):
    ...
```

`range(seq_len)` on a SymInt specializes it (`s = 32`), so every new sequence length is a new graph. The
comment shows this is deliberate for export, but under `torch.compile(dynamic=True)` it means Funnel can't
be dynamic over sequence length at all:

| scenario | graphs per seq length 32 / 40 / 56 |
|---|---|
| eval, CPU | 2 / 2 / 1 |
| eval, CUDA | 2 / 2 / 1 |
| train, CUDA | 2 / 2 / 1 |

(CPU train was cancelled after a very slow CPU Inductor compile, and a rerun is pending. See `issues/inductor-cpu-tiling-stride-at-simplify.md`.)

### Possible fix

Keep the positions as tensors: `pos = torch.arange(seq_len, device=device)`. Do the pooling
(`stride_pool_pos`) and relative-position math (`relative_pos`) with tensor ops, so the sizes are
`seq_len // 2**k` expressions instead of Python ints. Funnel always pools by 2, so the shapes stay
expressible symbolically. If export needs static sizes, gate the list path on `torch.compiler.is_exporting()`.

<!-- affected-models:manual -->
### Affected models

All runs use `torch.compile(backend="inductor", dynamic=True)` over (batch, seq, audio samples) =
(2, 32, 16000), (3, 40, 17600), (4, 56, 19200), on a model built from the checkpoint's config with reduced
layers. "Graphs per size" is the number of graphs compiled on the 1st / 2nd / 3rd size (anything after the
1st size is a recompile).

| model | config from | scenario · device | graphs per size |
|---|---|---|---|
| funnel | `funnel-transformer/small` | eval · CPU | 2 / 2 / 1 |
| funnel | | eval · CUDA | 2 / 2 / 1 |
| funnel | | train · CUDA | 2 / 2 / 1 |
| funnel | | train · CPU | not finished (compile cancelled after 1–2.5 h, rerun pending) |
