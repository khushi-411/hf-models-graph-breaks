# gpt_bigcode — graph breaks

- **HF id:** `bigcode/gpt_bigcode-santacoder`
- **Class:** `AutoModelForCausalLM`
- **Config overrides:** `{'n_layer': 2}`
- **torch:** `2.16.0a0+git9933f27`  |  **transformers:** `5.22.1`
- **Notes:** StarCoder 1 architecture (multi-query attention).

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ok | 1 | 0 | 0 |
| eval_with_cache | ok | 1 | 0 | 0 |
| train_forward | ok | 1 | 0 | 0 |

## eval_forward

No graph breaks. ✅

## eval_with_cache

No graph breaks. ✅

## train_forward

No graph breaks. ✅
