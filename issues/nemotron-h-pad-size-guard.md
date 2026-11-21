<!-- Draft for: huggingface/transformers. Once filed, put the URL in known_issues.yaml and set status: open -->

# `torch.compile(dynamic=True)`: NemotronH `if pad_size > 0:` (cut off chunk padding) recompiles when `seq_len % chunk_size` changes

### System info

- transformers: `5.24.0.dev0` (main, git `b7fe69cb19`)
- torch: `2.16.0a0+git57f02f8` (source build)
- Python 3.15, Linux; checked on CPU and on CUDA (B200)

### Description

`mamba2_chunk_scan` in `modeling_nemotron_h.py` always pads the sequence to a multiple of `chunk_size`
(`:302`-`:314`), and then removes the padding only when there was some (`:372`):

```python
pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
D_residual = D[..., None] * pad_tensor_by_size(hidden_states, pad_size)
hidden_states, A, B, C = [reshape_into_chunks(tensor, pad_size, chunk_size) for tensor in (hidden_states, A, B, C)]
...
# Cutting off padded chunks
if pad_size > 0:
    output = output[:, :sequence_length]
```

With dynamic shapes, `if pad_size > 0` installs the guard
`((16 + (-1)*(seq_len % 16)) % 16) <= 0`, so the compiled model recompiles when an input crosses between
"already a multiple of the chunk size" and "padded".

`torch.compile(dynamic=True)` + Inductor over sequence lengths 32 / 40 / 56 (CUDA, B200): 1 recompile in
`eval_forward` and 1 in `train_forward`, with exactly that guard as the recompile reason
(`# if pad_size > 0:`).

### Reproduction

`issues/repros/nemotron_h_pad_size_guard.py`:

```
[if pad_size > 0: slice] compiled graphs over seq 32/40/56: 2
[always slice]           compiled graphs over seq 32/40/56: 1
```

### Possible fix

Slice unconditionally: `output = output[:, :sequence_length]` is a no-op when nothing was padded, and the
guard goes away.

### Affected

nemotron_h. The same `if pad_size > 0:` is in bamba, granitemoehybrid, zamba2 and falcon_h1 (one
occurrence each in their modeling files); worth fixing together.
