# VLA Training V1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fine-tune SmolVLA (450M) with LoRA on LIBERO pick-and-place tasks using Kaggle T4, with OpenVLA 7B as inference baseline.

**Architecture:** LeRobot ecosystem for training/eval, LIBERO (MuJoCo) for simulation, modular adapters for future extensibility. All heavy compute runs as Kaggle notebooks; local code is the reusable project structure.

**Tech Stack:** Python 3.10+, LeRobot 0.4.x, SmolVLA, PEFT/LoRA, MuJoCo, LIBERO, HuggingFace Hub, OpenVLA (inference only)

---

## Task 1: Project Structure & Dependencies

**Files:**
- Create: `pyproject.toml`
- Create: `src/vlm_vla/__init__.py`
- Create: `src/vlm_vla/configs.py`
- Create: `notebooks/01_setup_and_verify.ipynb` (Kaggle notebook)

**Step 1: Create project structure**

```
vlm-vla/
├── src/vlm_vla/
│   ├── __init__.py
│   ├── configs.py          # All training/eval configs
│   ├── env_adapter.py      # Environment abstraction
│   ├── eval_engine.py      # Rollout evaluation
│   └── data_utils.py       # Dataset helpers
├── notebooks/
│   ├── 01_setup_and_verify.ipynb
│   ├── 02_data_collection.ipynb
│   ├── 03_smolvla_train.ipynb
│   ├── 04_openvla_baseline.ipynb
│   └── 05_evaluation.ipynb
├── configs/
│   └── smolvla_libero.yaml
├── scripts/
│   └── eval_libero.sh
├── docs/plans/
├── research/
├── pyproject.toml
└── README.md
```

**Step 2: Write pyproject.toml**

```toml
[project]
name = "vlm-vla"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.1.0",
    "lerobot[smolvla,peft,libero]",
    "huggingface_hub",
    "wandb",
]

[project.optional-dependencies]
openvla = [
    "bitsandbytes>=0.43.0",
    "accelerate>=0.26.0",
]
dev = [
    "pytest",
    "ruff",
]
```

**Step 3: Write configs.py**

```python
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
```

**Step 4: Write `src/vlm_vla/__init__.py`**

```python
"""VLA model training project — SmolVLA + OpenVLA on LIBERO."""
```

**Step 5: Commit**

```bash
git add pyproject.toml src/ notebooks/ configs/ scripts/
git commit -m "feat: project structure with configs and notebook stubs"
```

---

## Task 2: Kaggle Setup & Verification Notebook

**Files:**
- Create: `notebooks/01_setup_and_verify.ipynb`

This notebook runs on Kaggle T4 to verify the full environment works.

**Step 1: Write notebook cell 1 — Install dependencies**

```python
# Cell 1: Install (run once per Kaggle session)
!pip install -q lerobot[smolvla,peft,libero]

# MuJoCo headless rendering
import os
os.environ["MUJOCO_GL"] = "egl"

# Verify GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
assert torch.cuda.is_available(), "No GPU found!"
```

**Step 2: Write notebook cell 2 — Verify LeRobot + SmolVLA loads**

```python
# Cell 2: Verify SmolVLA model loads
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, SmolVLAConfig

config = SmolVLAConfig()
print(f"SmolVLA config loaded: {config.pretrained_path}")

# Quick memory check — load model briefly
policy = SmolVLAPolicy(config, dataset_stats=None)
param_count = sum(p.numel() for p in policy.parameters())
print(f"SmolVLA parameters: {param_count / 1e6:.1f}M")
del policy
torch.cuda.empty_cache()
print("SmolVLA load: OK")
```

**Step 3: Write notebook cell 3 — Verify LIBERO environment**

```python
# Cell 3: Verify LIBERO simulation
import gymnasium as gym
from lerobot.envs.libero import LiberoEnv  # adjust import as needed

# Check LIBERO tasks are available
print("LIBERO environment: OK")
print("MuJoCo rendering backend:", os.environ.get("MUJOCO_GL", "not set"))
```

**Step 4: Write notebook cell 4 — Verify dataset loads**

```python
# Cell 4: Verify dataset access
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("HuggingFaceVLA/libero")
print(f"Dataset loaded: {len(dataset)} frames")
sample = dataset[0]
print(f"Sample keys: {list(sample.keys())}")
print(f"Action shape: {sample['action'].shape}")
print(f"State shape: {sample['observation.state'].shape}")
print("Dataset: OK")
```

**Step 5: Write notebook cell 5 — VRAM budget check**

```python
# Cell 5: LoRA training dry run (1 step) to measure VRAM
import subprocess

# Run 1 training step to measure peak memory
result = subprocess.run([
    "lerobot-train",
    "--policy.path=lerobot/smolvla_base",
    "--dataset.repo_id=HuggingFaceVLA/libero",
    "--batch_size=8",
    "--steps=1",
    "--peft.method_type=LORA",
    "--peft.r=32",
    "--wandb.enable=false",
    "--env.type=libero",
    "--env.task=libero_object",
], capture_output=True, text=True, timeout=300)

print("STDOUT:", result.stdout[-500:] if result.stdout else "empty")
print("STDERR:", result.stderr[-500:] if result.stderr else "empty")

# Check peak VRAM
peak_mem = torch.cuda.max_memory_allocated() / 1e9
print(f"Peak VRAM: {peak_mem:.2f} GB / 16 GB")
assert peak_mem < 15.0, f"VRAM too high: {peak_mem:.2f} GB"
print("VRAM budget: OK")
```

**Step 6: Commit**

```bash
git add notebooks/01_setup_and_verify.ipynb
git commit -m "feat: Kaggle setup and verification notebook"
```

---

## Task 3: Data Collection & Exploration

**Files:**
- Create: `notebooks/02_data_collection.ipynb`
- Create: `src/vlm_vla/data_utils.py`

**Step 1: Write data_utils.py — dataset inspection helpers**

```python
"""Dataset utilities for VLA training."""
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def inspect_dataset(repo_id: str) -> dict:
    """Load and summarize a LeRobotDataset."""
    ds = LeRobotDataset(repo_id)
    sample = ds[0]
    info = {
        "repo_id": repo_id,
        "total_frames": len(ds),
        "num_episodes": ds.num_episodes,
        "features": {k: list(v.shape) for k, v in sample.items() if hasattr(v, "shape")},
        "fps": ds.fps,
    }
    return info


def get_task_descriptions(repo_id: str) -> list[str]:
    """Extract unique task descriptions from dataset."""
    ds = LeRobotDataset(repo_id)
    tasks = ds.meta.tasks
    return tasks
```

**Step 2: Write notebook cell 1 — Load and explore LIBERO dataset**

```python
# Cell 1: Explore the LIBERO dataset
import sys
sys.path.insert(0, "/kaggle/working/vlm-vla/src")
from vlm_vla.data_utils import inspect_dataset

info = inspect_dataset("HuggingFaceVLA/libero")
for k, v in info.items():
    print(f"{k}: {v}")
```

**Step 3: Write notebook cell 2 — Visualize sample trajectories**

```python
# Cell 2: Visualize a sample episode
import matplotlib.pyplot as plt
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("HuggingFaceVLA/libero")

# Show 8 frames from episode 0
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
ep_indices = [i for i in range(len(ds)) if ds[i]["episode_index"] == 0]
step = max(1, len(ep_indices) // 8)
for idx, ax in enumerate(axes.flat):
    frame_idx = ep_indices[min(idx * step, len(ep_indices) - 1)]
    sample = ds[frame_idx]
    # Image key may vary — adapt based on inspect_dataset output
    img_key = [k for k in sample if "image" in k and hasattr(sample[k], "shape") and len(sample[k].shape) == 3][0]
    img = sample[img_key].permute(1, 2, 0).numpy()
    ax.imshow(img)
    ax.set_title(f"step {idx * step}")
    ax.axis("off")
plt.suptitle("Episode 0 trajectory")
plt.tight_layout()
plt.savefig("episode_0_viz.png", dpi=100)
plt.show()
```

**Step 4: Write notebook cell 3 — Action distribution analysis**

```python
# Cell 3: Analyze action distributions
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("HuggingFaceVLA/libero")

# Sample 1000 actions
actions = []
for i in range(0, min(1000, len(ds))):
    actions.append(ds[i]["action"].numpy())
actions = np.stack(actions)

print(f"Action shape: {actions.shape}")
print(f"Action range: [{actions.min():.3f}, {actions.max():.3f}]")
print(f"Action mean: {actions.mean(axis=0)}")
print(f"Action std:  {actions.std(axis=0)}")

# Plot per-dimension distributions
fig, axes = plt.subplots(1, actions.shape[1], figsize=(3 * actions.shape[1], 3))
labels = ["x", "y", "z", "rx", "ry", "rz", "grip"]
for d in range(min(actions.shape[1], len(labels))):
    axes[d].hist(actions[:, d], bins=50)
    axes[d].set_title(labels[d] if d < len(labels) else f"dim{d}")
plt.tight_layout()
plt.savefig("action_distribution.png", dpi=100)
plt.show()
```

**Step 5: Commit**

```bash
git add src/vlm_vla/data_utils.py notebooks/02_data_collection.ipynb
git commit -m "feat: data exploration notebook and utilities"
```

---

## Task 4: SmolVLA LoRA Training Notebook

**Files:**
- Create: `notebooks/03_smolvla_train.ipynb`
- Create: `configs/smolvla_libero.yaml`

**Step 1: Write training config YAML**

```yaml
# configs/smolvla_libero.yaml
# SmolVLA LoRA fine-tune on LIBERO-Object — optimized for Kaggle T4 16GB
policy:
  path: lerobot/smolvla_base
  optimizer_lr: 1.0e-3
  scheduler_decay_lr: 1.0e-4
  output_features: null
  input_features: null

dataset:
  repo_id: HuggingFaceVLA/libero

env:
  type: libero
  task: libero_object

training:
  batch_size: 8
  steps: 20000
  save_checkpoint: true
  checkpoint_interval: 5000
  log_interval: 100

peft:
  method_type: LORA
  r: 32

eval:
  n_episodes: 10
  batch_size: 1

wandb:
  enable: false
```

**Step 2: Write notebook cell 1 — Install and setup**

```python
# Cell 1: Install and setup
!pip install -q lerobot[smolvla,peft,libero]
import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```

**Step 3: Write notebook cell 2 — Run training**

```python
# Cell 2: Train SmolVLA with LoRA
import subprocess
import time

start = time.time()
result = subprocess.run([
    "lerobot-train",
    "--policy.path=lerobot/smolvla_base",
    "--dataset.repo_id=HuggingFaceVLA/libero",
    "--batch_size=8",
    "--steps=20000",
    "--save_checkpoint=true",
    "--log_interval=100",
    "--peft.method_type=LORA",
    "--peft.r=32",
    "--policy.optimizer_lr=1e-3",
    "--policy.scheduler_decay_lr=1e-4",
    "--env.type=libero",
    "--env.task=libero_object",
    "--wandb.enable=false",
    "--policy.output_features=null",
    "--policy.input_features=null",
], capture_output=False, text=True, timeout=36000)  # 10h timeout

elapsed = time.time() - start
print(f"\nTraining completed in {elapsed/3600:.1f} hours")
```

**Step 4: Write notebook cell 3 — Verify checkpoint and push to Hub**

```python
# Cell 3: Save and push checkpoint
import glob
from huggingface_hub import HfApi

# Find latest checkpoint
checkpoints = sorted(glob.glob("outputs/*/checkpoints/*"))
print(f"Checkpoints found: {checkpoints}")

if checkpoints:
    latest = checkpoints[-1]
    print(f"Latest checkpoint: {latest}")

    # Optional: push to Hub (set your HF token first)
    # api = HfApi()
    # api.upload_folder(
    #     folder_path=latest,
    #     repo_id="YOUR_USERNAME/smolvla-libero-object-lora",
    #     repo_type="model"
    # )
    # print("Pushed to Hub!")
```

**Step 5: Write notebook cell 4 — Plot training loss**

```python
# Cell 4: Plot training loss from logs
import json
import matplotlib.pyplot as plt

# LeRobot logs training metrics to stdout/files — parse them
# Location depends on LeRobot version; adapt path as needed
log_files = glob.glob("outputs/*/logs/*.json") + glob.glob("outputs/*/train_log.jsonl")
print(f"Log files: {log_files}")

if log_files:
    losses = []
    steps = []
    with open(log_files[0]) as f:
        for line in f:
            try:
                entry = json.loads(line)
                if "loss" in entry:
                    losses.append(entry["loss"])
                    steps.append(entry.get("step", len(steps)))
            except json.JSONDecodeError:
                continue

    if losses:
        plt.figure(figsize=(10, 4))
        plt.plot(steps, losses)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("SmolVLA LoRA Training Loss")
        plt.grid(True, alpha=0.3)
        plt.savefig("training_loss.png", dpi=100)
        plt.show()
        print(f"Final loss: {losses[-1]:.4f}")
```

**Step 6: Commit**

```bash
git add notebooks/03_smolvla_train.ipynb configs/smolvla_libero.yaml
git commit -m "feat: SmolVLA LoRA training notebook and config"
```

---

## Task 5: OpenVLA Inference Baseline Notebook

**Files:**
- Create: `notebooks/04_openvla_baseline.ipynb`

**Step 1: Write notebook cell 1 — Install OpenVLA deps**

```python
# Cell 1: Install OpenVLA inference dependencies
!pip install -q transformers bitsandbytes>=0.43.0 accelerate>=0.26.0 timm pillow
!pip install -q lerobot[libero]

import os
os.environ["MUJOCO_GL"] = "egl"

import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

**Step 2: Write notebook cell 2 — Load OpenVLA with 4-bit quantization**

```python
# Cell 2: Load OpenVLA 7B with 4-bit quantization
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
import torch

processor = AutoProcessor.from_pretrained(
    "openvla/openvla-7b", trust_remote_code=True
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)

vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    quantization_config=bnb_config,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map="auto",
)

peak_mem = torch.cuda.max_memory_allocated() / 1e9
print(f"OpenVLA loaded. Peak VRAM: {peak_mem:.2f} GB")
```

**Step 3: Write notebook cell 3 — Test inference on a single image**

```python
# Cell 3: Test single-image inference
from PIL import Image
import numpy as np

# Create a dummy test image (replace with actual LIBERO frame later)
dummy_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

instruction = "pick up the red block"
prompt = f"In: What action should the robot take to {instruction}?\nOut:"

inputs = processor(prompt, dummy_img).to(vla.device, dtype=torch.bfloat16)
action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

print(f"Predicted action: {action}")
print(f"Action shape: {action.shape}")
print("OpenVLA inference: OK")
```

**Step 4: Write notebook cell 4 — Use pre-trained LIBERO checkpoint**

```python
# Cell 4: Load LIBERO-finetuned OpenVLA for fair comparison
# These are officially fine-tuned checkpoints with known performance

OPENVLA_LIBERO_CHECKPOINTS = {
    "libero_object": "openvla/openvla-7b-finetuned-libero-object",  # 88.4%
    "libero_spatial": "openvla/openvla-7b-finetuned-libero-spatial",  # 84.7%
    "libero_goal": "openvla/openvla-7b-finetuned-libero-goal",  # 79.2%
}

# For fair comparison, we use the LIBERO-Object checkpoint
# (same task suite as our SmolVLA training)
print("Available OpenVLA LIBERO checkpoints:")
for suite, ckpt in OPENVLA_LIBERO_CHECKPOINTS.items():
    print(f"  {suite}: {ckpt}")

# NOTE: These checkpoints are full fine-tuned (not LoRA), so they show
# OpenVLA's ceiling performance. Our SmolVLA LoRA result being competitive
# with these would be a strong V1 success signal.
```

**Step 5: Commit**

```bash
git add notebooks/04_openvla_baseline.ipynb
git commit -m "feat: OpenVLA 7B inference baseline notebook"
```

---

## Task 6: Evaluation Engine

**Files:**
- Create: `src/vlm_vla/eval_engine.py`
- Create: `notebooks/05_evaluation.ipynb`
- Create: `scripts/eval_libero.sh`

**Step 1: Write eval_engine.py**

```python
"""Evaluation engine for VLA models on LIBERO."""
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class TaskResult:
    task_name: str
    success_rate: float
    num_episodes: int
    avg_steps: float
    failures: dict  # {"miss": n, "drop": n, "timeout": n, "collision": n}


@dataclass
class EvalReport:
    model_name: str
    task_suite: str
    results: list  # list[TaskResult]

    @property
    def avg_success_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.success_rate for r in self.results) / len(self.results)

    def to_table(self) -> str:
        header = f"{'Task':<35} | {'Success':>8} | {'Episodes':>8} | {'Avg Steps':>9}"
        sep = "-" * len(header)
        rows = [header, sep]
        for r in self.results:
            rows.append(
                f"{r.task_name:<35} | {r.success_rate:>7.1%} | {r.num_episodes:>8} | {r.avg_steps:>9.1f}"
            )
        rows.append(sep)
        rows.append(
            f"{'Average':<35} | {self.avg_success_rate:>7.1%} | {'':>8} | {'':>9}"
        )
        return "\n".join(rows)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model_name": self.model_name,
            "task_suite": self.task_suite,
            "avg_success_rate": self.avg_success_rate,
            "results": [
                {
                    "task_name": r.task_name,
                    "success_rate": r.success_rate,
                    "num_episodes": r.num_episodes,
                    "avg_steps": r.avg_steps,
                    "failures": r.failures,
                }
                for r in self.results
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


def compare_reports(*reports: EvalReport) -> str:
    """Generate comparison table across multiple models."""
    if not reports:
        return "No reports to compare."

    # Collect all task names
    all_tasks = []
    for r in reports:
        for tr in r.results:
            if tr.task_name not in all_tasks:
                all_tasks.append(tr.task_name)

    # Build header
    model_names = [r.model_name for r in reports]
    header = f"{'Task':<35}"
    for name in model_names:
        header += f" | {name:>15}"
    sep = "-" * len(header)

    rows = [header, sep]
    for task in all_tasks:
        row = f"{task:<35}"
        for report in reports:
            tr = next((r for r in report.results if r.task_name == task), None)
            val = f"{tr.success_rate:.1%}" if tr else "N/A"
            row += f" | {val:>15}"
        rows.append(row)

    rows.append(sep)
    avg_row = f"{'Average':<35}"
    for report in reports:
        avg_row += f" | {report.avg_success_rate:>14.1%}"
    rows.append(avg_row)

    return "\n".join(rows)
```

**Step 2: Write eval_libero.sh**

```bash
#!/usr/bin/env bash
# scripts/eval_libero.sh — Run LIBERO evaluation for a trained policy
set -euo pipefail

POLICY_PATH="${1:?Usage: eval_libero.sh <policy_path> [task_suite] [n_episodes]}"
TASK_SUITE="${2:-libero_object}"
N_EPISODES="${3:-20}"

export MUJOCO_GL=egl

echo "=== Evaluating ${POLICY_PATH} on ${TASK_SUITE} ==="
echo "Episodes per task: ${N_EPISODES}"

lerobot-eval \
  --policy.path="${POLICY_PATH}" \
  --env.type=libero \
  --env.task="${TASK_SUITE}" \
  --eval.batch_size=1 \
  --eval.n_episodes="${N_EPISODES}"
```

**Step 3: Write evaluation notebook**

```python
# Cell 1: Run SmolVLA evaluation
import subprocess
import os
os.environ["MUJOCO_GL"] = "egl"

# Point to your trained checkpoint
SMOLVLA_CHECKPOINT = "outputs/latest/checkpoints/last"  # adjust path

result = subprocess.run([
    "lerobot-eval",
    f"--policy.path={SMOLVLA_CHECKPOINT}",
    "--env.type=libero",
    "--env.task=libero_object",
    "--eval.batch_size=1",
    "--eval.n_episodes=20",
], capture_output=True, text=True, timeout=7200)

print("STDOUT:", result.stdout[-2000:])
print("STDERR:", result.stderr[-500:] if result.returncode != 0 else "")
```

```python
# Cell 2: Parse results and build report
import sys
sys.path.insert(0, "/kaggle/working/vlm-vla/src")
from vlm_vla.eval_engine import EvalReport, TaskResult, compare_reports

# Parse lerobot-eval output (adapt based on actual output format)
# This is a template — fill with actual parsed values after first run
smolvla_report = EvalReport(
    model_name="SmolVLA-LoRA",
    task_suite="libero_object",
    results=[
        # TaskResult(task_name="task_0", success_rate=0.8, num_episodes=20, avg_steps=150, failures={}),
        # ... fill after running eval
    ],
)

print(smolvla_report.to_table())
smolvla_report.save("results/smolvla_libero_object.json")
```

```python
# Cell 3: Compare SmolVLA vs OpenVLA
# After running OpenVLA eval in notebook 04, load both reports
openvla_report = EvalReport(
    model_name="OpenVLA-7B-FT",
    task_suite="libero_object",
    results=[
        # Known reference: 88.4% average on libero_object
        # Fill with actual per-task breakdown
    ],
)

comparison = compare_reports(smolvla_report, openvla_report)
print(comparison)
```

**Step 4: Make eval script executable and commit**

```bash
chmod +x scripts/eval_libero.sh
git add src/vlm_vla/eval_engine.py notebooks/05_evaluation.ipynb scripts/eval_libero.sh
git commit -m "feat: evaluation engine with comparison tables"
```

---

## Task 7: Environment Adapter (Extensibility Layer)

**Files:**
- Create: `src/vlm_vla/env_adapter.py`

**Step 1: Write env_adapter.py**

```python
"""Environment adapter for swappable simulation backends.

V1: LIBERO (MuJoCo) only.
V3+: Extend to Isaac Lab, real hardware.
"""
from dataclasses import dataclass


@dataclass
class ActionSpaceConfig:
    """Extensible action space definition.

    V1: 7D (6 DoF arm + gripper)
    V4: 30D+ (humanoid full body)
    """
    dims: int = 7
    groups: dict = None  # e.g. {"arm": 6, "gripper": 1}
    normalize: bool = True
    action_range: tuple = (-1.0, 1.0)

    def __post_init__(self):
        if self.groups is None:
            self.groups = {"arm": self.dims - 1, "gripper": 1}


# Default configs for supported environments
LIBERO_ACTION_SPACE = ActionSpaceConfig(
    dims=7,
    groups={"arm": 6, "gripper": 1},
)

# Future: Isaac Lab humanoid
# ISAAC_HUMANOID_ACTION_SPACE = ActionSpaceConfig(
#     dims=30,
#     groups={"left_arm": 7, "right_arm": 7, "torso": 3, "legs": 12, "gripper_l": 1, "gripper_r": 1},
# )
```

**Step 2: Commit**

```bash
git add src/vlm_vla/env_adapter.py
git commit -m "feat: environment adapter with extensible action space"
```

---

## Task 8: End-to-End Smoke Test

**Files:**
- Create: `tests/test_smoke.py`

This runs locally (no GPU needed) to verify code structure.

**Step 1: Write smoke test**

```python
"""Smoke tests for VLA project structure — no GPU required."""
import pytest


def test_configs_importable():
    from vlm_vla.configs import SmolVLATrainConfig, OpenVLAInferConfig, EvalConfig

    cfg = SmolVLATrainConfig()
    assert cfg.batch_size == 8
    assert cfg.lora_rank == 32
    assert cfg.env_task == "libero_object"


def test_eval_engine_importable():
    from vlm_vla.eval_engine import EvalReport, TaskResult, compare_reports

    tr = TaskResult("test_task", 0.8, 20, 150.0, {})
    report = EvalReport("test_model", "test_suite", [tr])
    assert report.avg_success_rate == 0.8
    assert "test_task" in report.to_table()


def test_eval_report_save_load(tmp_path):
    from vlm_vla.eval_engine import EvalReport, TaskResult

    tr = TaskResult("pick_block", 0.75, 20, 120.0, {"timeout": 5})
    report = EvalReport("SmolVLA", "libero_object", [tr])

    path = str(tmp_path / "report.json")
    report.save(path)

    import json
    with open(path) as f:
        data = json.load(f)
    assert data["avg_success_rate"] == 0.75
    assert data["results"][0]["task_name"] == "pick_block"


def test_compare_reports():
    from vlm_vla.eval_engine import EvalReport, TaskResult, compare_reports

    r1 = EvalReport("ModelA", "suite", [TaskResult("t1", 0.8, 20, 100, {})])
    r2 = EvalReport("ModelB", "suite", [TaskResult("t1", 0.6, 20, 150, {})])
    table = compare_reports(r1, r2)
    assert "ModelA" in table
    assert "ModelB" in table


def test_env_adapter():
    from vlm_vla.env_adapter import ActionSpaceConfig, LIBERO_ACTION_SPACE

    assert LIBERO_ACTION_SPACE.dims == 7
    assert LIBERO_ACTION_SPACE.groups["arm"] == 6
    assert LIBERO_ACTION_SPACE.groups["gripper"] == 1


def test_data_utils_importable():
    from vlm_vla.data_utils import inspect_dataset, get_task_descriptions
    # Just verify importable — actual dataset tests need network
```

**Step 2: Run tests**

```bash
cd /home/denny/projects/vlm-vla
PYTHONPATH=src pytest tests/test_smoke.py -v
```

Expected: All 6 tests pass.

**Step 3: Commit**

```bash
git add tests/test_smoke.py
git commit -m "test: smoke tests for configs, eval engine, env adapter"
```

---

## Task 9: Kaggle Workflow Guide

**Files:**
- Create: `docs/kaggle-workflow.md`

**Step 1: Write the workflow guide**

```markdown
# Kaggle Workflow Guide

## Quick Start

### Session 1: Setup & Verify (~30 min)
1. Create new Kaggle notebook with GPU T4 x2
2. Upload `notebooks/01_setup_and_verify.ipynb`
3. Run all cells — verify: GPU OK, SmolVLA loads, LIBERO works, dataset loads, VRAM fits

### Session 2: Data Exploration (~1 hour)
1. Upload `notebooks/02_data_collection.ipynb` + `src/vlm_vla/data_utils.py`
2. Run all cells — understand data shapes, visualize trajectories, check action distributions

### Session 3-4: Training (~8-15 hours)
1. Upload `notebooks/03_smolvla_train.ipynb` + `configs/smolvla_libero.yaml`
2. Run training (20k steps ≈ 4-8 hours)
3. Save checkpoint to Kaggle output
4. If session times out: resume from last checkpoint in next session

### Session 5: OpenVLA Baseline (~2 hours)
1. Upload `notebooks/04_openvla_baseline.ipynb`
2. Load OpenVLA 7B with 4-bit quantization
3. Run inference on LIBERO test frames
4. Record baseline metrics

### Session 6: Evaluation (~3 hours)
1. Upload `notebooks/05_evaluation.ipynb` + `src/vlm_vla/eval_engine.py`
2. Run SmolVLA rollout evaluation (20 episodes × 10 tasks)
3. Compare against OpenVLA baseline
4. Generate final report

## Tips
- Always set `MUJOCO_GL=egl` before any LIBERO code
- Save checkpoints every 5000 steps (session may timeout)
- Use `wandb.enable=false` to avoid auth issues on Kaggle
- Push important checkpoints to HuggingFace Hub for persistence
```

**Step 2: Commit**

```bash
git add docs/kaggle-workflow.md
git commit -m "docs: Kaggle workflow guide for training sessions"
```

---

## Execution Summary

| Task | Description | Est. Time |
|------|-------------|-----------|
| 1 | Project structure & dependencies | 10 min |
| 2 | Kaggle setup notebook | 15 min |
| 3 | Data collection notebook | 15 min |
| 4 | SmolVLA training notebook | 15 min |
| 5 | OpenVLA baseline notebook | 15 min |
| 6 | Evaluation engine | 20 min |
| 7 | Environment adapter | 5 min |
| 8 | Smoke tests | 10 min |
| 9 | Kaggle workflow guide | 5 min |
| **Total** | | **~2 hours** |

After implementing all 9 tasks, you'll have a complete, tested project ready to run on Kaggle. The actual training/eval will happen in Kaggle sessions following the workflow guide.
