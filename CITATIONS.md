# Dataset and method sources

This project references upstream datasets but does not redistribute their contents. Check the upstream dataset card and item-level provenance before training or redistributing a model.

## Added Hugging Face datasets

- `secemp9/arxiv-complete` — arXiv complete corpus; mixed author licenses. https://huggingface.co/datasets/secemp9/arxiv-complete
- `MoreThought/Fable-5.1-Max-Reasoning-Filtered-5000x` — 5,000 filtered agentic coding/reasoning traces; Apache-2.0 per card. https://huggingface.co/datasets/MoreThought/Fable-5.1-Max-Reasoning-Filtered-5000x
- `openbmb/UltraData-SFT-Agent-2609` — agent SFT trajectories with Code-Agent, Search-Agent, General-Agent and Tool-Use configs; Apache-2.0 per card with upstream terms applying. https://huggingface.co/datasets/openbmb/UltraData-SFT-Agent-2609
- `zgcagi/ZGCM-1-Data` — bilingual pretraining/midtraining/SFT collection; repository is marked `other` and requires accepting Hugging Face access conditions. https://huggingface.co/datasets/zgcagi/ZGCM-1-Data
- `OpenDataArena/Spark-234K` — 234K scientific reasoning instruction/output examples; Apache-2.0 per card. https://huggingface.co/datasets/OpenDataArena/Spark-234K
- `openbmb/UltraData-Code` — L2/L3 code data across 11 languages; Apache-2.0 for the dataset project, with source-repository licenses still applying. https://huggingface.co/datasets/openbmb/UltraData-Code
- `echel0nn1881/kimi-cyber-reasoning` — 997 explicit reasoning/tool-call cybersecurity traces; WTFPL per card. https://huggingface.co/datasets/echel0nn1881/kimi-cyber-reasoning
- `saidutta69/fable-5-premium-v2` — 100,000 agent traces with tool-call structures; MIT per card. https://huggingface.co/datasets/saidutta69/fable-5-premium-v2
- `saidutta69/fable-5-premium` — 12,730-record predecessor; kept as a disabled legacy fallback because v2 supersedes it in the default mixture. https://huggingface.co/datasets/saidutta69/fable-5-premium
- `openbmb/UltraData-RL-2609` — 85,995 verifiable RL samples across Math, Knowledge, Long-Context and Code; Apache-2.0 for the release while upstream licenses continue to apply. https://huggingface.co/datasets/openbmb/UltraData-RL-2609
- `oi-uae/cyber-security` — cybersecurity instruction/chat dataset; research-only, non-commercial license per card. https://huggingface.co/datasets/oi-uae/cyber-security

## Relevant research

- ReAct — https://arxiv.org/abs/2210.03629
- Toolformer — https://arxiv.org/abs/2302.04761
- DeepSeek-R1 — https://arxiv.org/abs/2501.12948
- DAPO — https://arxiv.org/abs/2503.14476
