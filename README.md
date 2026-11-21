# hf-models-graph-breaks

Tracks **`torch.compile` graph breaks** in Hugging Face `transformers` models.

- [`models.yaml`](models.yaml): the models being tracked (edit this to add models)
- [`SUMMARY.md`](SUMMARY.md): one table with all models, scenarios and breaks (generated)
- `results/<model>/README.md`: a detailed report for each model: the reason, source location, Dynamo hints, docs link and stack (generated)
- `results/<model>/<scenario>.json`: raw results (generated)

## How it works

For each model and scenario, [`graph_breaks/check.py`](graph_breaks/check.py):

1. Builds the model from its **config only**, with random weights and a reduced
   number of layers (`config_overrides`). Graph breaks come from code paths, not
   from weight values, so this is fast and downloads only `config.json`. Pass
   `--pretrained` to load the real weights.
2. Runs `torch._dynamo.explain(model)(**inputs)` to count graphs and breaks.
3. Captures Dynamo's `graph_breaks` log, the same output as `TORCH_LOGS=graph_breaks`.
   This log gives the real source line and the explanation. `explain()` on its
   own often reports only a vague reason such as `step_unsupported`.
4. Groups duplicate breaks (repeated layers hit the same line), then writes
   JSON and markdown reports.

Scenarios:

| Scenario | Mode | Notes |
|---|---|---|
| `eval_forward` | `eval()`, `no_grad` | `use_cache=False` |
| `eval_with_cache` | `eval()`, `no_grad` | `use_cache=True` (decoder LMs) |
| `train_forward` | `train()`, grad on | passes `labels`, so the loss code is traced too |

Each report records the exact torch and transformers versions (plus the git sha
when transformers is installed from a checkout), so you can compare results
across versions.

## Setup

```bash
# Use an env with the torch build you want to test (stable/nightly/source build)
conda activate pt2dev
pip install -r requirements.txt

# Optional: test your own transformers checkout instead of the pip release
pip install -e ../transformers
```

## Usage

```bash
python -m graph_breaks.check --list                    # show tracked models
python -m graph_breaks.check --model gpt2              # one model, all its scenarios
python -m graph_breaks.check --model gpt2 --model bert # several models
python -m graph_breaks.check --all                     # everything in models.yaml
python -m graph_breaks.check --model gpt2 --scenario train_forward
python -m graph_breaks.check --model gpt2 --device cuda
python -m graph_breaks.check --model gpt2 --pretrained # real weights
python -m graph_breaks.check --model gpt2 --verbose    # also print the full Dynamo log

python -m graph_breaks.summarize                       # regenerate SUMMARY.md
```

For gated models (Llama, Gemma, …): `export HF_TOKEN=hf_...`.

## Adding a model

Add an entry to `models.yaml`:

```yaml
  - name: bert
    hf_id: google-bert/bert-base-uncased
    auto_class: AutoModelForMaskedLM
    inputs: text            # or: image
    config_overrides: {num_hidden_layers: 2}
    scenarios: [eval_forward, train_forward]
```

Then run `python -m graph_breaks.check --model bert && python -m graph_breaks.summarize`
and commit `results/bert/` and `SUMMARY.md`.

## Tracking breaks as upstream issues

File **one issue per root cause**, not one per model. A break in shared code
(e.g. `modeling_utils.py`) shows up in many models.

- [`known_issues.yaml`](known_issues.yaml) lists each root cause and its upstream
  issue. Breaks are matched by file, function and reason text. Line numbers are
  not used because they change between versions.
- [`issues/`](issues/) holds local issue drafts. [`issues/repros/`](issues/repros/)
  holds minimal repro scripts.
- `SUMMARY.md` tags every break with its issue. A break that matches no entry is
  tagged **NEW**, and `summarize` prints a YAML stub for it that you can paste in.

Triage loop for a NEW break:

1. Read its section in `results/<model>/README.md` (reason, hints, stack).
2. Decide where it belongs: the model code does something compile-unfriendly →
   `huggingface/transformers`; Dynamo should support it → `pytorch/pytorch`
   (label `module: dynamo`).
3. Check whether it's still on `main`:
   `PYTHONPATH=../transformers/src python issues/repros/<repro>.py`
4. Search upstream for an existing issue. If there isn't one, write a draft in
   `issues/<id>.md`.
5. Paste the stub into `known_issues.yaml` (status `draft`). After filing, set
   `issue:` to the URL and `status: open`. When it's fixed, set `fixed-upstream`
   and rerun to confirm the break is gone.

Statuses: `draft`, `open`, `fixed-upstream`, `wontfix`, `not-filing`.

## Debugging a specific break further

```bash
TORCH_LOGS="graph_breaks" python -m graph_breaks.check --model gpt2 --scenario train_forward
TORCH_LOGS="+dynamo"      python -m graph_breaks.check --model gpt2 --scenario train_forward  # very verbose
TORCH_TRACE=/tmp/trace    python -m graph_breaks.check --model gpt2 && tlparse /tmp/trace       # HTML report (pip install tlparse)
```
