# Dataset registry and processing contract

The default recipe now includes a curated subset of the user-requested Hugging Face datasets. They are references in YAML, not vendored copies.

## Stage placement

| Dataset | Stage | Processing | Default local cap |
|---|---|---|---:|
| `secemp9/arxiv-complete` (`paper_text`) | Pretrain | full-text normalization | 50,000 streamed examples |
| `zgcagi/ZGCM-1-Data` (`zgcm-1-pretrain-stage1`) | Pretrain | full-text normalization | 100,000 streamed examples |
| `openbmb/UltraData-Code` (`UltraData-Code-L3/py`) | Pretrain | task/analysis/solution serialization | 100,000 streamed examples |
| `openbmb/UltraData-SFT-Agent-2609` (`Code-Agent`) | SFT | preserved chat/tool traces | 50,000 streamed examples |
| `openbmb/UltraData-SFT-Agent-2609` (`Tool-Use`) | SFT | preserved chat/tool traces | 30,000 streamed examples |
| `saidutta69/fable-5-premium-v2` | SFT | preserved OpenAI chat/tool structure | 50,000 streamed examples |
| `MoreThought/Fable-5.1-Max-Reasoning-Filtered-5000x` | SFT | preserved reasoning conversations | 5,000 streamed examples |
| `OpenDataArena/Spark-234K` | SFT | instruction/output → chat | 100,000 streamed examples |
| `echel0nn1881/kimi-cyber-reasoning` | SFT | explicit reasoning/tool-call chat | 997 streamed examples |
| `oi-uae/cyber-security` | SFT | normalized chat | 50,000 streamed examples; **non-commercial only** |
| `openbmb/UltraData-RL-2609` (`Math`) | RLVR/GRPO/PPO/DAPO | query + ground truth | 30,000 streamed examples |
| `openbmb/UltraData-RL-2609` (`Code`) | RLVR/GRPO/PPO/DAPO | query + executable ground truth | 15,000 streamed examples |

## Why the recipe uses these subsets

The point is capability coverage, not simply increasing the number of dataset names. arXiv contributes scientific/academic text; ZGCM contributes broad foundation/midtraining-style data; UltraData-Code contributes code; UltraData-SFT-Agent and Fable provide tool/agent trajectories; Spark provides scientific reasoning; Kimi Cyber and the UAE cybersecurity set provide security specialization; UltraData-RL provides explicit verifiable reward targets.

The defaults deliberately cap and stream the very large datasets. Several of the upstream releases are vastly larger than a normal workstation can materialize. The recipe therefore treats `max_examples` as a safety valve rather than assuming a full mirror is appropriate.

## Processing rules

All processors preserve provenance at the dataset-manifest level. Agent chat normalization now retains `tool_calls`, `tool_call_id`, and `name` fields rather than flattening them into plain assistant text.

`UltraData-Code-L3` rows can be serialized from `task`, `analysis`, and `solution`. `UltraData-RL-2609` rows retain `query` and structured `ground_truth`; Code examples must be evaluated in a real sandbox before assigning execution rewards.

## Licensing gate

The dataset manager records declared license and access metadata in `data/manifest.jsonl`. A dataset marked `research-only-non-commercial` must not be used for commercial training. Mixed-license corpora such as arXiv, ZGCM-1-Data and UltraData-Code require upstream provenance review; a repository-level license does not automatically grant rights to every underlying record.
