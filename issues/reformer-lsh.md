<!-- Draft for: huggingface/transformers. Previously marked not-filing in known_issues.yaml (reason kept below). Once filed, put the URL in known_issues.yaml and set status: open -->

# `torch.compile`: Reformer (LSH attention) graph-breaks 15-17 times per forward (`.item()` in hashing, data-dependent branches, skipped RNG-seed helpers)

### System info

- torch: `2.16.0a0+git57f02f8` (source build, CUDA 12.8)
- transformers: `5.24.0.dev0` (git `b7fe69cb19`)
- Python 3.15, Linux; CPU and CUDA (B200)

### Description

Harness (config-only, reduced layers), CPU and CUDA:

| scenario | graphs / breaks (CPU) | graphs / breaks (CUDA) |
|---|---|---|
| eval_forward | 16 / 15 | (same root causes) |
| train_forward | 18 / 17 | 16 / 15 |

Break sites (`transformers/models/reformer/modeling_reformer.py`):

| line | reason |
|---|---|
| 271 | `Tensor.item()` (`_hash_vectors` / bucket computation) |
| 751 | `Tensor.item()`, then "Failed to handle graph break gracefully" |
| 1517, 1534 | "Attempted to call function marked as skipped" (the `_init_*_seed` RNG helpers used by reversible layers, train) |
| 2023, 2028, 2073 | data-dependent branching in `forward` |
| `modeling_utils.py:4778` | `logger.warning` (train) |

The dynamic-shape check also can't run Reformer with the harness's sequence lengths (40, 56) because of
Reformer's own constraints (sequence length a multiple of `lcm(chunk_length)` = 16, and within
`axial_pos_shape`).

### Why this was marked "not filing" before

Reformer is a legacy model whose LSH attention is intentionally data-dependent (hash buckets, sorting,
per-chunk logic) and whose reversible layers rely on saving/restoring RNG state. Making it compile-friendly
would be a rewrite, not a fix. File only if Reformer is still maintained for `torch.compile`.

### Reproduction

```bash
python -m graph_breaks.check --model reformer
```
