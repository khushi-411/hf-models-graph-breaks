# iquestcoder — graph breaks

- **HF id:** `IQuestLab/IQuest-Coder-V1-40B-Instruct`
- **Class:** `AutoModelForCausalLM`
- **Config overrides:** `{'num_hidden_layers': 2, 'hidden_size': 1024, 'intermediate_size': 2048, 'num_attention_heads': 8, 'num_key_value_heads': 2, 'vocab_size': 32000}`
- **torch:** `2.16.0a0+git9933f27`  |  **transformers:** `5.24.0.dev0` (git `84fe6ceb8a`)
- **Notes:** Config-only model; transformers maps it to LlamaForCausalLM.

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
