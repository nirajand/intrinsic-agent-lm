# Intrinsic Agent LM v0.2

The previous artifact was an architectural skeleton. This revision is a training-system reference: it contains dataset acquisition, dataset cataloging, deterministic preparation, tokenizer integration, stage-specific processors, a staged training pipeline, preference/reward/RL objectives, checkpointing, and a single recipe that can advance through pretraining, SFT, reward modeling, DPO, RLHF, RLAIF, RLVR, GRPO, PPO, and DAPO.

## What is actually baked into the model

The `.safetensors` checkpoint contains the learned parameters for the language model and its agentic cognition modules: persistent latent memory, iterative planner state, action routing, verifier, risk, confidence, termination and value heads. These are not prompt conventions; they are trainable tensor parameters and are serialized into the same tensor artifact.

What is not in the tensor file is the operating system, network, filesystem, Python interpreter or arbitrary tool execution. The model emits structured actions; a minimal runtime is still required to execute external effects. This is an architectural boundary, not a packaging omission.

## Complete training flow

The intended lifecycle is:

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
        +--> RLAIF     (AI judge -> preference pairs -> DPO or PPO)
        +--> RLVR      (external verifiers -> rewards)
        +--> GRPO      (group-relative online RL)
        +--> PPO       (clipped policy optimization)
        +--> DAPO      (decoupled clipping + dynamic sampling hooks + overlong shaping)
        |
        v
final.safetensors
```

The `TrainingPipeline` automatically executes the configured stage order. Alignment does not begin because the model magically declared itself aligned. It begins because the recipe enters post-training stages and supplies explicit human, AI, reward-model, or verifiable task signals.

## Data management

The project uses Hugging Face Datasets for Hub acquisition. Datasets can be loaded normally or streamed; streaming avoids materializing massive datasets locally and is suitable for iterative training. Hugging Face documents both behaviors and their tradeoffs. citeturn812531search0turn812531search6

Every dataset is represented by a `DatasetSpec`, with repository, subset, split, revision, streaming mode, example cap and weight. `DatasetManager` keeps raw data, prepared JSONL shards, and an append-only manifest describing preparation events and SHA-256 fingerprints.

## Dataset mixture in the full recipe

The supplied full recipe intentionally uses different corpora for different objectives rather than pretending one dataset can serve every stage.

Pretraining uses FineWeb's `sample-10BT` configuration in the template. FineWeb publishes smaller sample configurations in addition to the full corpus; its current dataset card describes `sample-10BT` as approximately 10B GPT-2 tokens. citeturn543556search1

SFT uses OpenHermes-2.5, an instruction/chat compilation with roughly one million samples in its current dataset card. citeturn577401search5turn577401search9

Reward/preference stages use HelpSteer2 and UltraFeedback-family data. HelpSteer2 provides human-feedback data intended for reward-model and alignment research, while the current ActiveUltraFeedback release contains preference subsets and reports a 60k-sample active-learning preference dataset. citeturn577401search0turn577401search10turn577401search12

Reasoning RL stages use OpenR1-Math-220k. Its current card describes verified mathematical reasoning traces, multiple generations per problem, and a `default` subset intended for strong SFT/RL use; it is explicitly described as suitable for rejection sampling and preference optimization. citeturn543556search0turn543556search11

## Algorithm coverage

Pretraining: next-token causal language modeling.

SFT: supervised causal LM over normalized conversational/tool traces.

Reward modeling: pairwise Bradley-Terry/logistic preference loss on the native verifier/value head.

DPO: reference-model-relative preference optimization.

RLHF: reward-model training followed by PPO-style clipped policy updates.

RLAIF: AI-judged preference data can be converted into the same DPO/PPO interfaces. In production, the judge must be a separately controlled evaluator; using the policy to grade itself creates a correlated failure mode.

RLVR: rewards are produced by explicit verifiers rather than subjective model scores. The reference implementation includes exact-string, numeric-math, and JSON validity verifiers, with a clean registry for stronger external verifiers.

GRPO: groups of sampled trajectories are normalized within the group to form relative advantages, with a reference-policy KL term.

PPO: clipped policy-ratio objective with old-log-probability snapshots and advantages.

DAPO: the reference implementation exposes asymmetric clip ranges and dynamic-resampling hooks, plus an explicit overlong penalty. DAPO is not merely renamed GRPO: the paper presents decoupled clipping and dynamic sampling as key system components. citeturn206286academia3

## Agentic training

The model's training data can include tool traces. Tool-use behavior is represented as ordinary supervised tokens plus an explicit action head. This follows the basic research direction of Toolformer—learning when/which API to call and how to use results—and ReAct-style interleaving of reasoning and actions. citeturn206286academia0turn206286academia1

For reasoning RL, the pipeline is compatible with the broad training philosophy demonstrated by DeepSeek-R1: warm-start or cold-start supervised data followed by reinforcement learning can elicit additional reasoning behaviors. citeturn206286academia2

## Important limitation

This project is a serious scaffold for training and experimentation, not a claim that a 4k-dimension 32-layer model is GPT-4-equivalent. Frontier capability is determined by model/data/compute scale, optimizer choices, data quality, evaluation, distributed systems and post-training. A checkpoint format cannot manufacture those.

Likewise, “alignment happens itself” must be interpreted as “the training controller automatically activates the configured alignment stages and objectives.” It cannot honestly mean “the model autonomously knows what human values are and proves its own alignment.”

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[all]'
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

For extremely large corpora, use streaming and an explicit step/token budget rather than attempting to download the whole dataset. Hugging Face documents streaming specifically for datasets too large for local storage. citeturn812531search0

## Run the staged recipe

```bash
ialm train configs/recipe_full.yaml
```

For a production run, replace the demonstrator model dimensions with a cluster-sized configuration and integrate distributed sharding, fused kernels, checkpoint resumption, activation checkpointing, and experiment tracking. The project is intentionally structured so those concerns are orthogonal to the dataset contracts and stage algorithms.

## References

ReAct — https://arxiv.org/abs/2210.03629
Toolformer — https://arxiv.org/abs/2302.04761
DeepSeek-R1 — https://arxiv.org/abs/2501.12948
DAPO — https://arxiv.org/abs/2503.14476
Hugging Face Datasets — https://huggingface.co/docs/datasets/
Hugging Face TRL — https://huggingface.co/docs/trl/

## Curated 2026 dataset additions

The default recipe now incorporates a capability-diverse subset of the requested Hugging Face datasets. See `configs/datasets/2026_curated_additions.yaml` for the full registry and `docs/DATASETS.md` for stage placement, processing, streaming caps and licensing gates.

The recipe deliberately does not silently download the enormous upstream corpora. Each selected source is streamed and capped by default. Increase `max_examples` only after checking storage, bandwidth, compute budget and upstream terms.
