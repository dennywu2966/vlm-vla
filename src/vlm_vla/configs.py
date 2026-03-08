"""Centralized configuration for VLA training project."""
from dataclasses import dataclass, field


@dataclass
class SmolVLATrainConfig:
    """SmolVLA LoRA fine-tuning config optimized for T4 16GB."""
    # Model
    policy_path: str = "lerobot/smolvla_base"
    # Dataset
    dataset_repo_id: str = "HuggingFaceVLA/libero"
    # Training
    batch_size: int = 8
    steps: int = 20_000
    optimizer_lr: float = 1e-3
    scheduler_decay_lr: float = 1e-4
    # LoRA
    peft_method: str = "LORA"
    lora_rank: int = 32
    # Environment
    env_type: str = "libero"
    env_task: str = "libero_object"
    # Eval
    eval_n_episodes: int = 20
    eval_batch_size: int = 1


@dataclass
class OpenVLAInferConfig:
    """OpenVLA 7B 4-bit inference config for T4."""
    model_id: str = "openvla/openvla-7b"
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    env_type: str = "libero"
    task_suites: list[str] = field(default_factory=lambda: ["libero_object"])
    n_episodes_per_task: int = 20
    max_steps_per_episode: int = 400
    success_threshold: float = 0.70  # V1 target
    min_viable_threshold: float = 0.50
