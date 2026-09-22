# Intrinsic Agent LM v0.2

Training-system reference: dataset acquisition, cataloging, deterministic preparation, tokenizer integration, stage-specific processors, staged training pipeline, preference/reward/RL objectives, checkpointing, and one recipe advancing through pretraining, SFT, reward modeling, DPO, RLHF, RLAIF, RLVR, GRPO, PPO, and DAPO.

## What is in the checkpoint

`.safetensors` holds learned parameters: language model weights plus agentic cognition modules — persistent latent memory, iterative planner state, action routing, verifier, risk, confidence, termination, and value heads. These are trainable tensor parameters in the same artifact. Which heads receive gradient signal depends on the stage objectives that actually run: the verifier/value path is trained in reward modeling; action/risk/confidence/termination heads are architecture-present and receive signal only when a stage supplies matching labels (not currently wired in the default recipe).

Not in the tensor file: OS, network, filesystem, Python interpreter, arbitrary tool execution. Model emits structured actions; minimal runtime executes external effects. Architectural boundary, not packaging omission.

## Training flow

```text
Hub / local datasets
        |
        v
Dataset Registry -> Download/Stream -> Raw cache -> Processor -> Tokenized cache
        |
        +--> PRETRAIN  (causal LM)
        +--> SFT       (conversation/tool traces)
        +--> REWARD    (pairwise reward model / verifier head)
        +--> DPO       (offline preference optimization)
        +--> RLHF      (reward model -> PPO)
        +--> RLAIF     (AI-judged pairs -> DPO; judge is separate from policy)
        +--> RLVR      (external verifiers -> rewards)
        +--> GRPO      (group-relative online RL)
        +--> PPO       (clipped policy optimization)
        +--> DAPO      (decoupled clipping + dynamic sampling hooks + overlong shaping)
        |
        v
final.safetensors
```

`TrainingPipeline` executes configured stage order. Alignment stages activate when recipe enters post-training with explicit human, AI, reward-model, or verifiable task signals — not by self-declaration.

## Data management

Hugging Face Datasets for Hub acquisition, plus local `.jsonl` paths for fixtures. Load normally or stream; streaming avoids local materialization of massive corpora.

Every prepare/download requires an explicit `max_examples` cap (or `IALM_ALLOW_FULL_DOWNLOAD=1`). `DatasetSpec` fields: repository, subset, split, revision, streaming mode, example cap, weight. `DatasetManager` keeps raw data, prepared JSONL shards, and an append-only manifest with preparation events and SHA-256 fingerprints.

## Dataset mixture (full recipe)

Different corpora per objective (`configs/recipe_full.yaml`). Weights are metadata for external mixing; training consumes rows in recipe order (`mixture_policy: sequential_by_recipe_order`).

- **Pretrain**: FineWeb `sample-10BT` (capped, e.g. 100k examples), arXiv paper text, ZGCM-1 stage1, UltraData-Code L3/python.
- **SFT**: OpenHermes-2.5, UltraData-SFT-Agent (Code/Tool-Use), Fable-5 traces, Spark-234K, kimi-cyber, oi-uae (non-commercial).
- **Reward/preference**: HelpSteer2 preference + ActiveUltraFeedback.
- **Reasoning RL**: OpenR1-Math-220k + UltraData-RL Math/Code, verifier-oriented.

See `configs/datasets/2026_curated_additions.yaml` and `docs/DATASETS.md` for caps, processing, licensing gates.

## Algorithm coverage

| Stage | Objective |
|-------|-----------|
| Pretraining | next-token causal LM |
| SFT / tool_sft | supervised causal LM over normalized conversational/tool traces |
| Reward modeling | pairwise Bradley-Terry/logistic preference loss on native verifier head |
| DPO | reference-model-relative preference optimization |
| RLHF | reward model → PPO-style clipped policy updates |
| RLAIF | pre-judged AI preference pairs → DPO (`judges.AIFeedbackJudge` is a separate offline tool) |
| RLVR | explicit verifiers: exact-string, numeric-math, JSON validity; registry for external verifiers |
| GRPO | group-normalized relative advantages + reference-policy KL |
| PPO | clipped policy-ratio, behavior-policy old-log-prob snapshots, multi-epoch updates |
| DAPO | asymmetric clip ranges, dynamic resampling, overlong penalty from actual generated length |

RLAIF judge must be separately controlled; policy grading itself creates correlated failure mode.

DAPO ≠ renamed GRPO: decoupled clipping and dynamic sampling are distinct system components. Overlong penalty uses `overlong_penalty * max(0, gen_len - max_new_tokens)` per sequence.

## Agentic training

Tool traces as ordinary supervised tokens plus explicit action head architecture. Follows Toolformer (when/which API, how to use results) and ReAct (interleaved reasoning + actions). Default recipe trains LM loss on tool traces; dedicated action-label supervision is not wired yet.

Reasoning RL compatible with DeepSeek-R1 philosophy: warm/cold-start supervised data → RL to elicit additional reasoning behaviors.

## Limitation

Scaffold for training and experimentation. Not a claim that a 4k-dimension 32-layer model matches large frontier systems. Frontier capability = model/data/compute scale + optimizer choices + data quality + evaluation + distributed systems + post-training. Checkpoint format cannot manufacture those.

"Alignment begins automatically" = training controller activates configured alignment stages/objectives. Not = model autonomously knows human values or proves own alignment.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
```

## Smoke test

```bash
python scripts/smoke_local.py
pytest -q
```

## Download and prepare data

```bash
ialm data download configs/recipe_full.yaml --stage pretrain
ialm data prepare configs/recipe_full.yaml pretrain
```

Large corpora: use streaming + `max_examples` (required unless `IALM_ALLOW_FULL_DOWNLOAD=1`), not full download.

## Run staged recipe

```bash
ialm train configs/recipe_full.yaml
```

Production: replace demonstrator dimensions with cluster-sized config; add distributed sharding, fused kernels, checkpoint resumption, activation checkpointing, experiment tracking. Those concerns stay orthogonal to dataset contracts and stage algorithms.

## Curated 2026 dataset additions

Capability-diverse subset of Hugging Face datasets in default recipe. Full registry: `configs/datasets/2026_curated_additions.yaml`. Stage placement, processing, streaming caps, licensing gates: `docs/DATASETS.md`.

No silent download of enormous upstream corpora. Each source streamed and capped by default. Raise `max_examples` only after checking storage, bandwidth, compute budget, upstream terms. Pin `revision` before any reproducible run.

## References

- ReAct — https://arxiv.org/abs/2210.03629
- Toolformer — https://arxiv.org/abs/2302.04761
- DeepSeek-R1 — https://arxiv.org/abs/2501.12948
- DAPO — https://arxiv.org/abs/2503.14476
- Hugging Face Datasets — https://huggingface.co/docs/datasets/
- Hugging Face TRL — https://huggingface.co/docs/trl/
