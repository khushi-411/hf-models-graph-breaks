<!-- Draft for: huggingface/transformers. Once filed, put the URL in known_issues.yaml and set status: open -->

# `torch.compile(dynamic=True)`: placeholder-mask checks build their error f-string eagerly, which specializes the number of image/audio features

### System info

- transformers: `5.24.0.dev0` (main, git `b7fe69cb19`)
- torch: `2.16.0a0+git57f02f8` (source build)
- Python 3.15, Linux; CPU and CUDA (B200)

### Description

`get_placeholder_mask` in multimodal models checks that the number of placeholder tokens matches the number
of features. The message is an f-string built **before** the call:

```python
n_image_features = image_features.shape[0] * image_features.shape[1]
torch_compilable_check(
    inputs_embeds[special_image_mask].numel() == image_features.numel(),
    f"Image features and image tokens do not match, tokens: {n_image_tokens}, features: {n_image_features}",
)
```

Under Dynamo, formatting `n_image_features` (a SymInt) into a `str` forces it to a concrete value and
installs a guard on it. So the feature count (number of images × tokens per image, or audio frames) is
specialized, and each new value recompiles:

```
llava       - 4/0: (image_features.size()[1]*image_features.size()[0]) == 1179648  # f"Image features and image tokens do not match, ..."  # modeling_llava.py:213
gemma3_vlm  - 4/0: (image_features.size()[1]*image_features.size()[0]) == 512      # modeling_gemma3.py:810
omniasr     - n_audio_features == 162 / == 98                                      # modeling_omniasr.py:494
```

The message only matters when the check fails. `torch_compilable_check` already accepts a callable message
(`msg: str | Callable[[], str]`), and while tracing it ignores `msg` entirely. Only the eager f-string at
the call site causes the guard.

Seen with `torch.compile(dynamic=True)`, Inductor, (batch, seq, audio) = (2, 32, 16000), (3, 40, 17600),
(4, 56, 19200), on CPU and CUDA, eval and train. Each of these models has other recompile causes too
(the `.tolist()` / `.item()` breaks tracked in other drafts), so this guard is one of the reasons, not
all of them:

| model | graphs per size, CPU eval | guard from the f-string |
|---|---|---|
| llava | 6 / 3 / 3 | yes |
| gemma3_vlm | 6 / 1 / 1 | yes |
| paligemma | 6 / 1 / 1 | yes |
| omniasr | 7 / 1 / 1 | yes (`n_audio_features`) |

### Possible fix

Make the message lazy at every call site:

```python
torch_compilable_check(
    inputs_embeds[special_image_mask].numel() == image_features.numel(),
    lambda: f"Image features and image tokens do not match, tokens: {n_image_tokens}, features: {n_image_features}",
)
```

There are 41 call sites with `features: {n_...}` f-strings in `models/*/modeling_*.py` (aria, audioflamingo3,
aya_vision, chameleon, cohere2_vision, deepseek_vl, deepseek_vl_hybrid, emu3, fast_vlm, florence2, fuyu,
fun_asr_nano, gemma3, glmasr, got_ocr2, granite_speech, granite_speech_plus, internvl, janus, lfm2_vl,
lighton_ocr, llava, minicpm_ocr, minicpmv4_6, minicpmv4_7, mistral3, mistral_large4, molmo2,
moss_transcribe_diarize, musicflamingo, nemotron3_5_transcribe, omniasr, ovis2, paligemma, qianfan_ocr,
qwen2_audio, qwen3_asr, vibevoice, vipllava, voxtral). Most are `# Copied from` / modular, so fixing the
templates covers them.

<!-- affected-models:manual -->
### Affected models

All runs use `torch.compile(backend="inductor", dynamic=True)` over (batch, seq, audio samples) =
(2, 32, 16000), (3, 40, 17600), (4, 56, 19200), on a model built from the checkpoint's config with reduced
layers. "Graphs per size" is the number of graphs compiled on the 1st / 2nd / 3rd size (anything after the
1st size is a recompile).

| model | config from | scenario · device | graphs per size |
|---|---|---|---|
| llava | `llava-hf/llava-1.5-7b-hf` | eval · CPU / CUDA | 6 / 3 / 3 · 6 / 3 / 3 |
| llava | | train · CPU / CUDA | 6 / 3 / 3 · 6 / 5 / 3 |
| gemma3_vlm | `unsloth/gemma-3-4b-it` | eval · CPU / CUDA | 6 / 1 / 1 · 6 / 1 / 1 |
| gemma3_vlm | | train · CPU / CUDA | 6 / 1 / 3 · 6 / 3 / 1 |
| paligemma | `google/paligemma2-3b-pt-224` | eval · CPU / CUDA | 6 / 1 / 1 · 6 / 1 / 1 |
| paligemma | | train · CPU / CUDA | 6 / 1 / 1 · 6 / 1 / 1 |
| omniasr | `hf-audio/omniasr-llm-300m-v2` | eval · CPU / CUDA | 7 / 1 / 1 · 7 / 1 / 1 |
| omniasr | | train · CPU / CUDA | 13 / 4 / 3 · 13 / 3 / 1 |

These counts include recompiles from the other VLM issues. The f-string guard is one of the recorded
reasons in each of them.

Same code, also in this sweep, but the guard wasn't seen:
- **mistral3** (19–21 recompiles): it already breaks on `image_sizes.tolist()` before the check.
- **internvl**: every dynamic run crashes first (`issues/dynamo-guard-python-mod-value-range.md`).

The other models in the 41-site list weren't in this sweep.
