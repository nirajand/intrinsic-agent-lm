# End-to-end training recipe

This is the operational recipe. `configs/recipe_full.yaml` is the machine-readable counterpart.

## 1. Bootstrap infrastructure

Use a distributed GPU environment for frontier-scale runs. The 70B-style configuration is only an architecture target; it requires substantial GPU memory, interconnect bandwidth, checkpoint storage, and data throughput.

Install:

```bash
pip install -e '.[all]'
```

Authenticate to Hugging Face when using gated/private artifacts:

```bash
huggingface-cli login
```

## 2. Dataset acquisition

The dataset manager supports either ordinary Hub caching or streaming. For massive corpora, streaming is preferred during experiments; for a fixed reproducible release, materialize selected shards and pin the dataset revision.

Example:

```bash
ialm data download configs/recipe_full.yaml --stage pretrain
ialm data prepare configs/recipe_full.yaml pretrain
```

Repeat per stage or run the full pipeline directly:

```bash
ialm train configs/recipe_full.yaml
```

The manager records source, split, subset, row count, preparation stage and SHA-256 of every generated prepared shard in `data/manifest.jsonl`.

## 3. Pretraining

Use a broad high-quality web mixture plus code, mathematics, scientific text, books and permissibly licensed corpora. FineWeb is included as the default web corpus template; its dataset card documents 10BT/100BT/350BT sample configurations. Do not treat FineWeb alone as sufficient for a frontier model.

Objective:

```text
L_pretrain = CE(next_token_logits, target_tokens)
```

Train on packed fixed-length sequences. Use gradient accumulation to obtain a large effective token batch. Save optimizer, scheduler, dataloader and RNG state for exact resume in the production implementation.

## 4. SFT / instruction tuning

Mix general instruction data, reasoning traces, tool-use traces, coding, math, long-context tasks and domain-specific exemplars. The supplied OpenHermes-2.5 dataset is a starting corpus, not a complete frontier-quality mixture.

Important processing rules:

- normalize role names into system/developer/user/tool/assistant;
- preserve tool-call/result boundaries;
- mask labels for prompt tokens when using assistant-only loss;
- cap and audit conversation length;
- deduplicate near-identical examples;
- record source provenance.

The native action head receives action labels when traces provide them.

## 5. Reward modeling

Build one or more reward models rather than relying on a single scalar objective. The native verifier/value head can be trained with a Bradley-Terry-style pairwise loss:

```text
L_rm = -log sigma(r(chosen) - r(rejected))
```

Keep a held-out preference set. Track reward-model calibration and disagreement; do not evaluate the reward model only on the training preferences.

HelpSteer2 is included as a reward-model dataset template.

## 6. DPO

Use high-quality chosen/rejected pairs. Keep a frozen reference model:

```text
L_DPO = -log sigma(beta * ((log pi(y+) - log pi_ref(y+))
                         - (log pi(y-) - log pi_ref(y-))))
```

The reference must actually be frozen. Do not replace it with the current policy.

## 7. RLHF

Classical RLHF in this project means:

```text
human preferences -> reward model -> rollout -> PPO -> updated policy
```

The PPO stage must snapshot old log-probabilities before updating the policy. Value learning should use an explicit value head rather than conflating verifier and policy likelihood.

## 8. RLAIF

Generate or collect preference judgments from a separate AI evaluator. Store evaluator model/version, rubric, scores, and prompt template as dataset metadata. Convert the resulting pairwise labels into DPO or reward-model examples.

Do not let the policy under training be the only evaluator of itself. That creates a correlated evaluator/policy failure mode and can turn self-consistency into false confidence.

## 9. RLVR

Prefer deterministic or independently executable verifiers whenever possible. The reference registry implements numeric math, exact string, and JSON validity rewards. Add stronger verifiers for coding, theorem proving and structured tool execution behind controlled sandboxes.

Reward should be a function of verifiable task success rather than stylistic similarity:

```text
R = verifier(output, target)
```

## 10. GRPO

For each prompt, sample a group of completions. Compute rewards for each completion and normalize within the group:

```text
A_i = (R_i - mean(R)) / (std(R) + eps)
```

Optimize the policy while constraining drift against a reference policy. Keep group sampling reproducible and log the complete group reward distribution.

## 11. PPO

Maintain old policy log-probabilities and a value estimate. Optimize the clipped surrogate:

```text
r_t(theta) = exp(log pi_theta - log pi_old)
L_policy = -min(r_t A_t, clip(r_t, 1-eps_low, 1+eps_high) A_t)
```

The code exposes separate lower and upper clip controls.

## 12. DAPO

The project exposes a DAPO-style stage with separate clip bounds, dynamic group resampling hooks, and explicit overlong-response shaping. The exact implementation must remain faithful to the research configuration being reproduced; do not claim that simply selecting asymmetric clipping reproduces the published DAPO system.

The published DAPO system specifically emphasizes decoupled clipping and dynamic sampling as important techniques in large-scale RL. Pin the paper/repository version before a reproducibility run.

## 13. Automatic progression / alignment

The pipeline stage order is declarative:

```yaml
stage_order:
  - pretrain
  - sft
  - reward_model
  - dpo
  - rlhf
  - rlaif
  - rlvr
  - grpo
  - ppo
  - dapo
```

That is the correct interpretation of "alignment begins automatically": once the pipeline reaches the configured stage, the matching data processor, loss and optimizer are activated automatically. It is not an autonomous proof that the model is aligned.

## 14. Agentic evaluation before promotion

Do not promote a checkpoint from training loss alone. Evaluate at minimum:

- language modeling loss/perplexity;
- instruction following;
- reasoning and math exact-match/verifier reward;
- tool selection accuracy and argument validity;
- multi-step task success;
- refusal/safety policy tests;
- reward-model exploit tests;
- long-context retrieval;
- calibration/confidence;
- regression against previous checkpoint;
- contamination and data leakage checks.

Save all metrics and the exact dataset/recipe revisions alongside the checkpoint.
