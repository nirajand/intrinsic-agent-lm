from __future__ import annotations
from dataclasses import dataclass, field, asdict

@dataclass
class ModelConfig:
    vocab_size: int = 128256
    max_seq_len: int = 32768
    d_model: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 8
    ffn_mult: float = 2.6875
    rope_theta: float = 500000.0
    dropout: float = 0.0
    role_classes: int = 6
    memory_slots: int = 32
    planning_steps: int = 2
    action_classes: int = 8
    tie_embeddings: bool = True
    use_qk_norm: bool = True
    use_moe: bool = False
    num_experts: int = 0
    top_k_experts: int = 0
    gradient_checkpointing: bool = False

    def validate(self):
        if self.d_model % self.n_heads: raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads: raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.max_seq_len < 16: raise ValueError("max_seq_len too small")
        if self.planning_steps < 1: raise ValueError("planning_steps must be >= 1")
        if self.use_moe and (self.num_experts < 2 or self.top_k_experts < 1): raise ValueError("invalid MoE settings")

    @property
    def head_dim(self): return self.d_model // self.n_heads

    def to_dict(self): return asdict(self)

@dataclass
class TrainConfig:
    output_dir: str = "runs/default"
    device: str = "cuda"
    seed: int = 42
    batch_size: int = 2
    micro_batch_size: int = 1
    grad_accum_steps: int = 1
    max_steps: int = 1000
    eval_every: int = 200
    save_every: int = 200
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = False
    max_grad_tokens: int = 2_000_000
    alignment_beta: float = 0.1
    kl_beta: float = 0.02
    dpo_loss_type: str = "sigmoid"
    ppo_clip_low: float = 0.2
    ppo_clip_high: float = 0.2
    ppo_epochs: int = 2
    grpo_group_size: int = 8
    grpo_kl_beta: float = 0.02
    dapo_clip_low: float = 0.2
    dapo_clip_high: float = 0.28
    dapo_max_resamples: int = 4
    overlong_penalty: float = 0.0
    entropy_coef: float = 0.0
    value_coef: float = 0.5
    reward_scale: float = 1.0
    use_ema: bool = True
    ema_decay: float = 0.999
    alignment_start_step: int = 0
    auto_align: bool = True
    stage_order: list[str] = field(default_factory=lambda: [
        "pretrain", "sft", "reward_model", "dpo", "rlhf", "rlaif", "rlvr", "grpo", "ppo", "dapo"
    ])
