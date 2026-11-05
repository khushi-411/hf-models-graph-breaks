<!-- Draft for: huggingface/transformers. Once filed, put the URL in known_issues.yaml and set status: open -->

# `torch.compile` graph break in `PreTrainedModel.loss_function` (`logger.warning_once` in forward) for models with `loss_type=None`

### System info

- transformers: `5.23.0.dev0` (main @ `58d05ca572`), also `5.22.1`
- torch: `2.16.0a0+git9933f27`
- Python 3.15, Linux

### Description

For models whose class name doesn't match any key in `LOSS_MAPPING` (for example
`GPT2LMHeadModel`), `PreTrainedModel.__init__` sets `self.loss_type = None`.
Then every forward pass with `labels` goes through this branch in the
`loss_function` property:

```python
loss_type = getattr(self, "loss_type", None)
if loss_type is None or loss_type not in LOSS_MAPPING:
    logger.warning_once(
        f"`loss_type={loss_type}` was set in the config but it is unrecognized. "
        f"Using the default loss: `ForCausalLMLoss`."
    )
    loss_type = "ForCausalLM"
```

Dynamo can't trace `logger.warning_once`, so:

- `torch.compile(model)` gets a graph break in every training step, splitting the
  forward into 2 graphs.
- `torch.compile(model, fullgraph=True)` fails with
  `Unsupported: logging.Logger method not supported for non-export cases`.

The break happens even after the warning has already been emitted once. Dynamo
breaks on the call itself, whether or not it would print anything.

The warning text is also misleading: it says `loss_type=None` "was set in the
config", but the user never set it. It's the default for these model classes.

### Reproduction

```python
import torch
from transformers import GPT2Config, GPT2LMHeadModel

model = GPT2LMHeadModel(GPT2Config(n_layer=2)).train()
print(model.loss_type)  # None

input_ids = torch.randint(0, 1000, (2, 32))
explanation = torch._dynamo.explain(model)(input_ids=input_ids, labels=input_ids)
print(explanation.graph_break_count)  # 1

torch._dynamo.reset()
torch.compile(model, fullgraph=True)(input_ids=input_ids, labels=input_ids)
# torch._dynamo.exc.Unsupported: logging.Logger method not supported for non-export cases
```

Break location (from `TORCH_LOGS=graph_breaks`):

```
transformers/models/gpt2/modeling_gpt2.py:702 in forward
transformers/modeling_utils.py:4685 in loss_function   -> logger.warning_once(...)
```

### Expected behavior

No graph break: training-mode forward with `labels` should compile to a single
graph, as it already does for models with a valid `loss_type`.

### Possible fix

Resolve the fallback once at init instead of on every forward. For example, in
`PreTrainedModel.__init__`:

```python
if loss_type is None:
    logger.warning_once(f"No loss_type found for {self.__class__.__name__}; defaulting to ForCausalLMLoss.")
    loss_type = "ForCausalLM"
self.loss_type = loss_type
```

Or, as a smaller change, guard the warning in `loss_function` with
`if not is_torchdynamo_compiling():`.

I'm happy to open a PR.
