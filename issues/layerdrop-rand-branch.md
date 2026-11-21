<!-- Draft for: huggingface/transformers. Once filed, put the URL in known_issues.yaml and set status: open -->

# `torch.compile` graph break from LayerDrop (`if torch.rand([]) < self.layerdrop`) in train mode, even when `layerdrop=0.0`

### System info

- transformers: `5.22.1`
- torch: `2.16.0a0+git41b037b`
- Python 3.14, Linux

### Description

Many encoder/decoder loops implement LayerDrop like this (OPT shown,
`modeling_opt.py:374`):

```python
for idx, decoder_layer in enumerate(self.layers):
    if self.training:
        dropout_probability = torch.rand([])
        if dropout_probability < self.layerdrop:
            continue
```

`dropout_probability < self.layerdrop` is a tensor, so the `if` is
data-dependent branching, which Dynamo can't trace. In train mode:

- `torch.compile(model)` gets a graph break **in every layer** (OPT with
  2 layers: 4 graphs, 3 breaks).
- `torch.compile(model, fullgraph=True)` fails with
  `Unsupported: Data-dependent branching`.
- The break also leads to follow-on `Reconstruction failure` breaks
  (`sourceless variable DictItemsIterator()` / `DictIterator()`) in the
  `output_capturing.py` wrapper around `forward`.

The default is `layerdrop=0.0` for the checkpoints I checked
(`facebook/opt-125m`, `facebook/bart-base`), so the branch can never be taken,
yet every training step pays for it. Most users training these models never
turned LayerDrop on.

The same pattern is in 43 model families: autoformer, bart, bigbird_pegasus,
biogpt, blenderbot, blenderbot_small, conditional_detr, data2vec, flaubert, fsmt,
hubert, informer, kosmos2, lasr, led, m2m_100, marian, mask2former, mbart, mvp,
nemotron_asr_streaming, nllb_moe, opt, parakeet, pegasus, pegasus_x, plbart,
pp_formulanet, qwen2_audio, seamless_m4t, seamless_m4t_v2, sew, speecht5,
speech_to_text, table_transformer, time_series_transformer, trocr, unispeech,
unispeech_sat, wav2vec2, wav2vec2_bert, wav2vec2_conformer, wavlm, whisper, xglm.

I confirmed the break in train mode for 10 of them, all at their default
`layerdrop=0.0`: bart, biogpt, blenderbot, led, m2m_100 (NLLB), marian, mbart,
opt, pegasus, xglm. Eval mode compiles to a single graph in each of them.

### Reproduction

```python
import torch
from transformers import OPTConfig, OPTForCausalLM

model = OPTForCausalLM(OPTConfig(num_hidden_layers=2)).train()
print(model.config.layerdrop)  # 0.0

input_ids = torch.randint(0, 1000, (2, 32))
explanation = torch._dynamo.explain(model)(input_ids=input_ids, labels=input_ids)
print(explanation.graph_count, explanation.graph_break_count)  # 4 3

torch._dynamo.reset()
torch.compile(model, fullgraph=True)(input_ids=input_ids, labels=input_ids)
# torch._dynamo.exc.Unsupported: Data-dependent branching
```

Break location (from `TORCH_LOGS=graph_breaks`):

```
transformers/models/opt/modeling_opt.py:378 in forward   -> if dropout_probability < self.layerdrop:
transformers/models/bart/modeling_bart.py:537 in forward  -> (encoder; decoder at :660)
```

### Expected behavior

With `layerdrop=0.0`, a training-mode forward should compile to a single graph,
as it does in eval mode.

### Possible fix

Skip the random draw when LayerDrop is disabled:

```python
if self.training and self.layerdrop > 0:
    dropout_probability = torch.rand([])
    if dropout_probability < self.layerdrop:
        continue
```

I tested this on OPT: the train-mode forward goes from 4 graphs to 1, and
`fullgraph=True` succeeds. The follow-on `output_capturing` breaks go away too.
Behavior is unchanged: with `layerdrop=0.0` the layer was never skipped.

`layerdrop > 0` with `torch.compile` would still break. That case could use a
Python-side draw (`random.random() < self.layerdrop`), but that changes RNG
streams, so it's left out of this minimal fix.

Most of these files are generated from `# Copied from` / modular sources, so the
change can be made once in the source models and propagated with
`make fix-copies`. I'm happy to open a PR.
