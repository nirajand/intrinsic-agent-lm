# Dataset registry and processing contract

Default recipe includes a curated subset of Hugging Face datasets as YAML references, not vendored copies.

## Stage placement

| Dataset | Stage | Processing | Default local cap |
|---|---|---|---:|
| `HuggingFaceFW/fineweb` (`sample-10BT`) | Pretrain | text normalization | 100,000 streamed examples |
| `secemp9/arxiv-complete` (`paper_text`) | Pretrain | full-text normalization | 50,000 streamed examples |
| `zgcagi/ZGCM-1-Data` (`zgcm-1-pretrain-stage1`) | Pretrain | full-text normalization | 100,000 streamed examples |
| `openbmb/UltraData-Code` (`UltraData-Code-L3/py`) | Pretrain | task/analysis/solution serialization | 100,000 streamed examples |
| `teknium/OpenHermes-2.5` | SFT | chat normalization | 500,000 rows |
| `openbmb/UltraData-SFT-Agent-2609` (`Code-Agent`) | SFT | preserved chat/tool traces | 50,000 streamed examples |
| `openbmb/UltraData-SFT-Agent-2609` (`Tool-Use`) | SFT | preserved chat/tool traces | 30,000 streamed examples |
| `saidutta69/fable-5-premium-v2` | SFT | preserved OpenAI chat/tool structure | 50,000 streamed examples |
| `MoreThought/Fable-5.1-Max-Reasoning-Filtered-5000x` | SFT | preserved reasoning conversations | 5,000 streamed examples |
| `OpenDataArena/Spark-234K` | SFT | instruction/output → chat | 100,000 streamed examples |
| `echel0nn1881/kimi-cyber-reasoning` | SFT | explicit reasoning/tool-call chat | 997 streamed examples |
| `oi-uae/cyber-security` | SFT | normalized chat | 50,000 streamed examples; **non-commercial only** |
| `nvidia/HelpSteer2` (`preference`) | reward_model / rlhf | pairwise preference | 100,000 rows |
| `ActiveUltraFeedback/ultrafeedback` | dpo / rlaif | preference pairs | 60,000 rows |
| `open-r1/OpenR1-Math-220k` | RLVR/GRPO/PPO/DAPO | verified math traces | 60,000 rows |
| `openbmb/UltraData-RL-2609` (`Math`) | RLVR/GRPO/PPO/DAPO | query + ground truth | 30,000 streamed examples |
| `openbmb/UltraData-RL-2609` (`Code`) | RLVR/GRPO/PPO/DAPO | query + executable ground truth | 15,000 streamed examples |

## Why these subsets

Capability coverage, not dataset-name count. arXiv: scientific text. ZGCM: broad foundation data. UltraData-Code: code. UltraData-SFT-Agent and Fable: tool/agent trajectories. Spark: scientific reasoning. Kimi Cyber and oi-uae: security specialization. UltraData-RL and OpenR1: verifiable reward targets.

Large corpora are streamed and capped. `max_examples` is a safety valve; full materialization requires `IALM_ALLOW_FULL_DOWNLOAD=1`. Pin `revision` before reproducible runs.

## Processing rules

Processors preserve provenance at the dataset-manifest level. Agent chat normalization retains `tool_calls`, `tool_call_id`, and `name` rather than flattening them into plain assistant text.

`UltraData-Code-L3` rows serialize from `task`, `analysis`, and `solution`. `UltraData-RL-2609` rows retain `query` and structured `ground_truth`; Code examples must be evaluated in a real sandbox before assigning execution rewards.

Local fixtures: `repo` may point at a `.jsonl` path (used by `configs/recipe_smoke.yaml`).

## Licensing gate

License and access metadata are recorded in `data/manifest.jsonl`. `research-only-non-commercial` datasets must not be used for commercial training. Mixed-license corpora (arXiv, ZGCM-1, UltraData-Code) require upstream provenance review; repository-level license does not automatically grant rights to every underlying record.

## Source schema / output schema (adapter contract)

**Source (typical Hub row):** free-form; may include `text`/`content`/`paper_text`, `messages`/`conversations`, `instruction`/`output`, `prompt`/`chosen`/`rejected` or `response_1`/`response_2`/`preference_strength`, `problem`/`query`/`answer`/`ground_truth`, optional `tools`, `verifier`, `domain`, `source`.

**Output (prepared JSONL, per stage):**

| Stage | Output keys |
|---|---|
| pretrain | `text` (+ `input_ids`/`attention_mask` when tokenizer set) |
| sft / tool_sft | `messages` (normalized roles), `tools` (+ token fields) |
| reward_model / dpo / rlaif | `prompt`, `chosen`, `rejected` (+ `chosen_ids`/`rejected_ids` when tokenized) |
| rlhf / rlvr / grpo / ppo / dapo | `prompt`, `answer`, `verifier`, `domain`, `source` (+ `prompt_ids`/`answer_ids` when tokenized) |
