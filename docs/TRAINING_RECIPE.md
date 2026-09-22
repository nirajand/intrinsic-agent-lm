# End-to-end training recipe

Operational recipe. Machine-readable counterpart: `configs/recipe_full.yaml`.

## 1. Bootstrap infrastructure

Distributed GPU environment required for large runs. `configs/recipe_full.yaml` is a ~4k-dimension, 32-layer demonstrator-scale configuration, not a production frontier size. Large runs need GPU memory, interconnect, checkpoint storage, and data throughput.

Install:

```bash
pip install -r requirements.txt
pip install -e . --no-deps
```

Authenticate to Hugging Face when using gated/private artifacts:

```bash
huggingface-cli login
```

## 2. Dataset acquisition

Hub caching or streaming. For massive corpora, stream with an explicit `max_examples` cap; for a fixed reproducible release, materialize selected shards and pin `revision`.

Example:

```bash
ialm data download configs/recipe_full.yaml --stage pretrain
ialm data prepare configs/recipe_full.yaml pretrain
```

Repeat per stage or run the full pipeline directly:

```bash
ialm train configs/recipe_full.yaml
```

The manager records source, split, subset, row count, preparation stage and SHA-256 of every prepared shard in `data/manifest.jsonl`.

Full materialization without a cap requires `IALM_ALLOW_FULL_DOWNLOAD=1`.

## 3. Pretraining

Broad high-quality mixture: web + code + math + scientific text + licensed corpora. FineWeb is a template entry only; its card documents sample-10BT and larger configs. Do not treat FineWeb alone as sufficient.

Objective:

```text
L_pretrain = CE(next_token_logits, target_tokens)
```

Effective batch = `micro_batch_size * grad_accum_steps`. Linear warmup for `warmup_steps`, then decay toward `min_lr_ratio * lr` over `max_steps`.

## 4. SFT / instruction tuning

Mix general instruction data, reasoning traces, tool-use traces, coding, math, long-context tasks and domain-specific exemplars. OpenHermes-2.5 is a starting corpus, not a complete mixture.

Processing rules:

- normalize role names into system/developer/user/tool/assistant;
- preserve tool-call/result boundaries;
- optional assistant-only loss (prompt masking) is a future extension — default pipeline trains full tokenized sequences;
- cap and audit conversation length;
- deduplicate near-identical examples;
- record source provenance.

The native action head exists in the architecture. Dedicated action-label supervision is not wired in the default recipe; tool traces train via LM loss on supervised tokens.

## 5. Reward modeling

Pairwise Bradley-Terry loss on the native verifier head (optimizer must target reward-model parameters):

```text
L_rm = -log sigma(r(chosen) - r(rejected))
```

Keep a held-out preference set. Track calibration and disagreement; do not evaluate only on training preferences.

HelpSteer2 is the default reward-model dataset template.

## 6. DPO

High-quality chosen/rejected pairs. Frozen reference model:

```text
L_DPO = -log sigma(beta * ((log pi(y+) - log pi_ref(y+))
                         - (log pi(y-) - log pi_ref(y-))))
```

The reference must be frozen, not the current policy. Sequence log-probs exclude padding via attention masks.

## 7. RLHF

```text
human preferences -> reward model -> rollout -> PPO -> updated policy
```

PPO snapshots old log-probabilities from the behavior policy, then runs `ppo_epochs` gradient updates per rollout so the clip ratio can leave 1. Value learning should use an explicit value head rather than conflating verifier and policy likelihood (value-head training not fully wired in the reference PPO path).

## 8. RLAIF

Preference judgments from a separate AI evaluator. Store evaluator model/version, rubric, scores, and prompt template as dataset metadata. Convert pairwise labels into DPO examples (`ialm` stage `rlaif` runs DPO on prepared pairs). `judges.AIFeedbackJudge` is an offline helper for generating judgments — not run inside the training loop by default.

Do not let the policy under training be the only evaluator of itself. That creates a correlated evaluator/policy failure mode.

## 9. RLVR

Deterministic or independently executable verifiers. Registry: numeric math, exact string, JSON validity. Add stronger verifiers for coding, theorem proving, structured tool execution behind controlled sandboxes.

```text
R = verifier(output, target)
```

## 10. GRPO

Per prompt, sample a group; normalize rewards within the group:

```text
A_i = (R_i - mean(R)) / (std(R) + eps)
```

Optimize with reference-policy KL (`grpo_kl_beta`). Keep group sampling reproducible; log group reward distribution.

## 11. PPO

Behavior-policy old log-probabilities + clipped surrogate:

```text
r_t(theta) = exp(log pi_theta - log pi_old)
L_policy = -min(r_t A_t, clip(r_t, 1-eps_low, 1+eps_high) A_t)
```

Separate lower/upper clip controls. `ppo_epochs` > 1 required for clipping to engage after the first parameter update.

## 12. DAPO

Asymmetric clip bounds, dynamic group resampling when group rewards are constant, and overlong penalty:

```text
L_overlong = mean_i( overlong_penalty * max(0, gen_len_i - max_new_tokens) * log pi_i )
```

Penalty uses actual generated length (not `max_seq_len`) and multiplies sequence log-prob so it is gradient-bearing. This is not a claim that selecting asymmetric clipping reproduces the full published DAPO system. Pin the paper/repository version before a reproducibility run.

## 13. Automatic progression / alignment

Stage order is declarative:

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

"Alignment begins automatically" means: when the pipeline reaches the configured stage, the matching data processor, loss and optimizer activate. It is not an autonomous proof that the model is aligned.

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

Save metrics and exact dataset/recipe revisions alongside the checkpoint.
