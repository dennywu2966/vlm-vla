# HuggingFace LeRobot -- Comprehensive Research Report

**Date:** 2026-02-21
**Purpose:** Evaluate LeRobot for VLM/VLA project integration and OpenVLA compatibility

---

## 1. Repository Current State

| Metric | Value |
|--------|-------|
| **GitHub Stars** | 21,700+ |
| **Forks** | 3,800+ |
| **Open Issues** | 727 |
| **Latest Version** | v0.4.4 (PyPI) |
| **Python** | >= 3.10 |
| **License** | Apache 2.0 |
| **Last Commit** | 2026-02-20 (daily active) |
| **Install** | `pip install lerobot` |

**Repository Structure:**
- `src/lerobot/policies/` -- All policy implementations (ACT, Diffusion, SmolVLA, Pi0, Gr00T, XVLA, etc.)
- `src/lerobot/robots/` -- Hardware drivers (SO100, Koch, LeKiwi, Reachy2, Unitree G1, etc.)
- `src/lerobot/envs/` -- Simulation environment integrations (LIBERO, MetaWorld, Aloha, PushT)
- `src/lerobot/datasets/` -- LeRobotDataset format, streaming, transforms
- `src/lerobot/processor/` -- Unified processor pipeline for policy-hardware bridging

**Key CLI tools:**
```
lerobot-train          # Train a policy
lerobot-eval           # Evaluate in sim or real
lerobot-record         # Record demonstrations
lerobot-replay         # Replay episodes
lerobot-teleoperate    # Teleoperate a robot
lerobot-calibrate      # Calibrate servos
lerobot-edit-dataset   # Edit/merge/split datasets
```

**Commit Activity (last 7 days of Feb 2026):** 15+ commits from multiple core contributors (Steven Palma, Caroline Pascal, Pepijn Kooijmans, community contributors). The project is under very active daily development.

---

## 2. Supported Models -- Full Inventory

### 2.1 Model Categories

| Category | Models | Status |
|----------|--------|--------|
| **Imitation Learning** | ACT, Diffusion Policy, VQ-BeT | Stable, mature |
| **Reinforcement Learning** | HIL-SERL, TDMPC | Stable; SAC for RL fine-tuning |
| **VLA (Vision-Language-Action)** | SmolVLA, Pi0Fast, Pi0.5, GR00T N1.5, XVLA, Wall-X, SARM | Active development |

### 2.2 Per-Model Details

#### SmolVLA (450M params) -- **MOST RELEVANT for T4**
- **Architecture:** SmolVLM2-500M backbone + action expert with cross-attention
- **Parameters:** ~450M total
- **Input:** Multi-camera images + robot state + natural language instruction
- **Output:** Action chunks (default chunk_size=50)
- **VLM backbone:** `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`
- **Training defaults:** freeze_vision_encoder=True, train_expert_only=True
- **Key config:** 16 VLM layers, expert_width_multiplier=0.75, cross-attention every 2 layers
- **PEFT/LoRA support:** YES, officially supported via `lerobot[peft]`
- **Pretrained:** `lerobot/smolvla_base` on HF Hub

#### ACT (Action Chunking with Transformers)
- **Architecture:** Conditional VAE with Transformer encoder-decoder
- **Parameters:** ~10-50M (lightweight)
- **Use case:** Fine manipulation, bimanual tasks
- **Sim benchmarks:** Aloha sim environments

#### Diffusion Policy
- **Architecture:** Denoising diffusion for action generation
- **Parameters:** ~50-100M
- **Use case:** Multi-modal action distributions
- **Sim benchmarks:** PushT, Aloha

#### VQ-BeT (Vector-Quantized Behavior Transformer)
- **Architecture:** VQ-VAE + Transformer for discretized action prediction
- **Parameters:** ~30-80M

#### Pi0Fast / Pi0.5
- **Architecture:** Based on Physical Intelligence Pi0 (3B VLM backbone)
- **Parameters:** ~3B
- **Status:** Requires custom transformers branch (`fix/lerobot_openpi`)
- **Note:** Conflicts with standard transformers; separate install needed

#### GR00T N1.5 (NVIDIA)
- **Architecture:** Eagle2.5-VL backbone + flow-matching action head
- **Parameters:** Large (multi-billion)
- **Requires:** flash-attn, specific installation
- **Status:** Integrated but requires manual setup

#### XVLA
- **Architecture:** Florence2 backbone + soft transformer + action hub
- **Parameters:** Medium-large
- **Status:** Active development, training from scratch possible

#### Wall-X
- **Architecture:** Qwen2.5-VL MoE backbone
- **Requires:** transformers==4.49.0 (conflicts with other VLAs)
- **Status:** Isolated install required

#### SARM
- **Architecture:** Uses Qwen-VL utilities
- **Status:** New addition

#### RTC (Real-Time Chunking)
- **Purpose:** Runtime optimization layer for any chunking policy
- **Features:** Action queue, latency tracking, debug visualization

### 2.3 T4 16GB Feasibility Matrix

| Model | Full Fine-tune | LoRA Fine-tune | Inference | Notes |
|-------|---------------|----------------|-----------|-------|
| **SmolVLA (450M)** | TIGHT (~22GB needed for bs=44) | **YES (~11-14GB)** | **YES (~4-6GB)** | Best candidate. Official LoRA support. |
| **ACT** | **YES (<8GB)** | N/A (small enough) | **YES (<4GB)** | Lightweight, no VLM component |
| **Diffusion Policy** | **YES (<10GB)** | N/A | **YES (<4GB)** | Lightweight |
| **VQ-BeT** | **YES (<8GB)** | N/A | **YES (<4GB)** | Lightweight |
| **Pi0Fast/Pi0.5 (3B)** | NO (>48GB) | MAYBE (~20-24GB with 4bit) | TIGHT (~14-16GB with 4bit) | Too large for reliable T4 use |
| **GR00T N1.5** | NO (>>48GB) | NO | NO | Requires A100+ |
| **XVLA** | NO (>24GB) | MAYBE | MAYBE | Depends on Florence2 variant |
| **TDMPC** | **YES (<10GB)** | N/A | **YES** | RL policy, lightweight |

**Bottom line for T4 16GB:**
- **Recommended:** SmolVLA with LoRA (officially supported), ACT, Diffusion Policy
- **SmolVLA LoRA training** uses ~11.5GB VRAM at batch_size=44, well within T4
- **SmolVLA full fine-tune** needs ~22GB (L4/A10 minimum), exceeds T4
- **Pi0/GR00T** family is out of scope for T4

---

## 3. Simulation Environments

### 3.1 Supported Environments

| Environment | Install Extra | Tasks | Pick-and-Place? | Benchmark Quality |
|-------------|--------------|-------|-----------------|-------------------|
| **LIBERO** | `pip install -e ".[libero]"` | 130+ tasks across 4 suites | **YES** (LIBERO-Object is specifically pick-and-place) | Production-grade benchmark |
| **MetaWorld** | `pip install -e ".[metaworld]"` | 50 distinct manipulation tasks | **YES** (pick-place, reach, push) | Well-established RL benchmark |
| **Aloha Sim** | `pip install -e ".[aloha]"` | Bimanual tasks (transfer cube, insertion) | Partial (transfer cube) | Good for bimanual |
| **PushT** | `pip install -e ".[pusht]"` | 2D pushing task | NO | Toy/tutorial benchmark |
| **IsaacLab Arena** | Hub-based | NVIDIA Isaac Sim tasks | YES | Advanced, GPU-heavy |

### 3.2 LIBERO Details (Best for Pick-and-Place)

LIBERO is the primary benchmark in LeRobot for manipulation evaluation:

- **LIBERO-Object:** 10 tasks, each introduces a novel object for pick-and-place. Best for evaluating generalization to new objects.
- **LIBERO-Spatial:** 10 tasks testing spatial reasoning (different positions/orientations).
- **LIBERO-Goal:** 10 tasks with different goals in similar scenes.
- **LIBERO-100:** 100 long-horizon tasks covering diverse household manipulation.

```bash
# Evaluate on LIBERO
lerobot-eval \
  --policy.path=lerobot/pi0_libero_finetuned \
  --env.type=libero \
  --env.task=libero_object \
  --eval.n_episodes=10
```

### 3.3 MetaWorld Details

MetaWorld v3.0 with 50 continuous-control manipulation tasks including:
- `pick-place-v3` (directly relevant)
- `reach-v3`
- `push-v3`
- `drawer-open-v3`
- `assembly-v3`

### 3.4 EnvHub (New Feature)

LeRobot now supports **EnvHub** -- a mechanism to distribute custom simulation environments via the HF Hub. Users can publish their own Gymnasium environments as HF repos, and others can load them directly:

```python
# Load environment from Hub
env = make_env("username/my_custom_env", trust_remote_code=True)
```

---

## 4. LeRobotDataset Format

### 4.1 Format Versions

| Version | Status | Key Feature |
|---------|--------|-------------|
| v1 | Deprecated | Original format |
| v2 / v2.1 | Legacy but still on Hub | One file per episode |
| **v3.0** | Current (lerobot >= 0.4.0) | Multi-episode files, streaming support |

### 4.2 v3.0 Structure

```
dataset_repo/
├── meta/
│   ├── info.json          # Schema, features, shapes, dtypes, FPS, path templates
│   ├── stats.json         # Global normalization stats (mean/std/min/max)
│   ├── tasks.jsonl        # Natural language task descriptions -> integer IDs
│   └── episodes/          # Per-episode metadata as chunked Parquet
│       └── file-0000.parquet
├── data/
│   └── file-0000.parquet  # Frame-by-frame state/action data (many episodes per file)
└── videos/
    └── observation.images.front/
        └── file-0000.mp4  # MP4 video shards (many episodes per file)
```

### 4.3 Key Design Principles

1. **Parquet for tabular data:** States, actions, timestamps stored efficiently
2. **MP4 for video:** Camera frames encoded as video for 15-40x compression vs raw images
3. **Metadata-driven episode boundaries:** Episodes are reconstructed via offsets, not file boundaries
4. **Hub-native streaming:** `StreamingLeRobotDataset` reads directly from HF Hub without download
5. **Delta timestamps:** Support for temporal windows (e.g., past 3 frames as context)

### 4.4 Usage API

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Load from Hub
dataset = LeRobotDataset("lerobot/aloha_mobile_cabinet")

# Each sample is a dict of tensors:
sample = dataset[0]
# {
#   'observation.state': tensor([...]),       # Joint positions
#   'action': tensor([...]),                  # Target actions
#   'observation.images.front': tensor([C, H, W]),  # Camera image
#   'timestamp': tensor(1.234),
#   'episode_index': tensor(0),
#   'task_index': tensor(0),
# }

# Create custom dataset
dataset = LeRobotDataset.create(
    repo_id="user/my_dataset",
    fps=30,
    features={...}
)
for frame in episode_data:
    dataset.add_frame(frame)
dataset.save_episode()
dataset.finalize()  # REQUIRED before push_to_hub
dataset.push_to_hub()
```

### 4.5 Dataset Scale on Hub

The HF Hub hosts **thousands of robotics datasets** in LeRobotDataset format under the `lerobot/` organization. Notable ones:
- `lerobot/aloha_mobile_cabinet` -- Mobile ALOHA bimanual
- `lerobot/svla_so100_pickplace` -- SmolVLA paper reference dataset (50 episodes, 5 cube positions)
- `HuggingFaceVLA/libero` -- Full LIBERO benchmark data
- Community-contributed datasets for SO-100, Koch, various tasks

---

## 5. Supported Real Robot Hardware

### 5.1 Officially Supported Robots

| Robot | Type | Servos | Cost Estimate | Status |
|-------|------|--------|---------------|--------|
| **SO-100 / SO-101** | Single 6-DOF arm | Feetech STS3215 | ~$100-114/arm | Primary, best documented |
| **Koch v1.1** | Single 6-DOF arm | Dynamixel XL430/XL330 | ~$300-400/arm | Original LeRobot arm |
| **LeKiwi** | Mobile base + SO-100 arm | Feetech | ~$250 | Mobile manipulation |
| **HopeJR** | Humanoid-style hand | Feetech | Varies | Dexterous manipulation |
| **OMX (ROBOTIS)** | Professional arm | Dynamixel | ~$500+ | Partner integration |
| **OpenARM** | Open-source arm | Damiao motors | Varies | Community contribution |
| **EarthRover Mini Plus** | Outdoor rover | Custom | Varies | Outdoor robotics |
| **Reachy2** | Full humanoid | Pollen Robotics | $15,000+ | Post-acquisition integration |
| **Unitree G1** | Humanoid | Unitree proprietary | $16,000+ | Locomotion + manipulation |

### 5.2 Teleoperation Devices

| Device | Type | Use Case |
|--------|------|----------|
| **Leader arm** (SO-100/Koch) | Kinematic mirroring | Primary method for data collection |
| **Gamepad** | Joystick control | Simple tasks |
| **Keyboard** | Discrete control | Debugging |
| **Phone** | 6-DOF tracking via HEBI | Advanced teleoperation |

### 5.3 Camera Support

- **OpenCV cameras** (webcam, USB cameras) -- primary
- **Intel RealSense** depth cameras -- optional extra
- **Reachy2 built-in cameras** -- specific to Reachy2

### 5.4 Motor Protocol Support

- **Feetech** (STS3215 etc.) -- SO-100, SO-101, LeKiwi, HopeJR
- **Dynamixel** (XL430, XL330 etc.) -- Koch, OMX
- **Damiao** (CAN bus motors) -- OpenARM
- **Unitree SDK2** -- Unitree G1

---

## 6. Fine-tuning on Kaggle T4 16GB -- Detailed Analysis

### 6.1 SmolVLA LoRA Fine-tuning (RECOMMENDED)

**Official PEFT support** added in LeRobot with the `lerobot[peft]` extra:

```bash
pip install -e ".[smolvla,peft]"

lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=your_data \
  --batch_size=8 \
  --steps=20000 \
  --peft.method_type=LORA \
  --peft.r=64
```

**Memory estimates for SmolVLA LoRA on T4 16GB:**

| Configuration | VRAM Estimate | Feasible? |
|--------------|---------------|-----------|
| LoRA r=64, bs=8 | ~10-12 GB | **YES** |
| LoRA r=64, bs=16 | ~12-14 GB | **YES (tight)** |
| LoRA r=64, bs=32 | ~14-16 GB | **BORDERLINE** |
| LoRA r=32, bs=32 | ~13-15 GB | **LIKELY YES** |
| Full fine-tune, bs=44 | ~22 GB | NO |
| Full fine-tune, bs=8 | ~16-18 GB | NO |

**Practical recommendations for Kaggle T4:**
1. Use `--peft.method_type=LORA --peft.r=32` or `--peft.r=64`
2. Start with `--batch_size=8`, increase if memory allows
3. Enable gradient checkpointing if available
4. Training time: ~4-8 hours for 20k steps depending on batch size
5. The LoRA targets `q_proj`, `v_proj` of the LM expert plus state/action projections by default

### 6.2 ACT / Diffusion Policy on T4

Both are lightweight enough for full fine-tuning on T4:

```bash
lerobot-train \
  --policy=act \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
  --batch_size=32
```

ACT typically uses <8GB VRAM even with batch_size=32.

### 6.3 Pi0 on T4 -- NOT Feasible

Pi0/Pi0.5 requires 48GB+ for fine-tuning. Even with LoRA + 4-bit quantization, inference barely fits on T4. A GitHub issue (#2216) confirms even an A6000 (48GB) hits OOM for Pi0.5 fine-tuning.

### 6.4 Kaggle-Specific Considerations

- **Session limit:** 12 hours max on Kaggle
- **Disk:** 73GB available (enough for datasets)
- **Network:** Can download from HF Hub directly
- **Recommended workflow:**
  1. Install lerobot with smolvla+peft extras
  2. Load dataset via `LeRobotDataset(repo_id)` -- auto-downloads and caches
  3. Train SmolVLA with LoRA for 10-20k steps
  4. Save checkpoints to Kaggle output or push to Hub

---

## 7. Community Activity

### 7.1 Growth Metrics

| Metric | Value |
|--------|-------|
| **GitHub Stars** | 21,700+ (crossed 15,000 in Aug 2025) |
| **Forks** | 3,800+ |
| **Open Issues** | 727 (very active issue tracker) |
| **Contributors** | 100+ |
| **Discord** | Active server with dedicated LeRobot channels |
| **HF Hub Models** | 27+ official models under lerobot/ |
| **HF Hub Datasets** | Thousands of community datasets |

### 7.2 Core Team (HuggingFace)

Active daily committers include:
- **Steven Palma** (imstevenpmwork) -- Lead maintainer, 5+ commits/week
- **Caroline Pascal** -- Dataset infrastructure, processor pipeline
- **Pepijn Kooijmans** (pkooij) -- Hardware/calibration
- **Remi Cadene** -- Original creator
- **Simon Alibert** -- Core infrastructure
- **Adil Zouitine**, **Dana Aubakirova**, **Martino Russi** -- Various features

### 7.3 Community Ecosystem

- **XLeRobot** -- Community-built dual-arm mobile robot ($660)
- **lerobot-ros** -- ROS2 interface wrapper
- **Phospho** -- LeRobot dataset documentation/tooling
- **Chinese tutorials** (Tongji University) -- Comprehensive SO-ARM101 guides
- **Multiple Kaggle notebooks** -- Training SmolVLA on Colab/Kaggle
- **Hackathon participation** -- Global robotics hackathons using LeRobot
- **Corporate adoption:** HuggingFace acquired Pollen Robotics (Reachy2 maker) in April 2025

### 7.4 Release Cadence

- v0.1.0 to v0.4.4 in ~18 months
- Major features added every 2-4 weeks
- Breaking changes managed via dataset versioning (v1 -> v2 -> v2.1 -> v3)

---

## 8. OpenVLA Compatibility Analysis

### 8.1 Architectural Comparison

| Aspect | OpenVLA | SmolVLA (LeRobot) |
|--------|---------|-------------------|
| **Parameters** | 7B | 450M |
| **VLM Backbone** | Llama 2 7B + SigLIP | SmolVLM2-500M |
| **Action Output** | Single-step discrete tokens | Action chunks (50 steps, continuous) |
| **Training Data** | Open X-Embodiment (970k demos) | LeRobot Community Datasets |
| **Fine-tune VRAM** | 24-48 GB (full), ~16GB (LoRA+4bit) | 22GB (full), ~11GB (LoRA) |
| **Inference VRAM** | ~14-16 GB | ~4-6 GB |
| **Framework** | Standalone (Prismatic VLMs) | LeRobot (HF ecosystem) |
| **Action tokenization** | 256 discrete bins per dim | Continuous flow matching |

### 8.2 Integration Possibilities

1. **Dataset bridge:** OpenVLA's training data (RLDS format) can be converted to LeRobotDataset format. Tools exist for porting datasets.

2. **Policy wrapper:** OpenVLA could be wrapped as a LeRobot policy by implementing the `PreTrainedConfig` and policy interface, but this is non-trivial.

3. **Complementary use:**
   - Use OpenVLA for zero-shot generalization (large model, broad training)
   - Use SmolVLA for task-specific fine-tuning (small model, fast iteration)
   - Both can output actions for the same robot hardware

4. **Key difference:** OpenVLA predicts single-step actions (autoregressive tokens), while SmolVLA predicts action chunks (50 steps via flow matching). This fundamentally affects control loop design.

### 8.3 Practical Recommendation

For a T4 16GB setup:
- **SmolVLA with LoRA** is the practical choice -- fits comfortably, officially supported, well-documented
- **OpenVLA** would need aggressive quantization (4-bit) just for inference on T4, and fine-tuning is borderline even with LoRA
- **Best approach:** Use LeRobot's SmolVLA as the primary model, with the option to evaluate against OpenVLA on a larger GPU later

---

## 9. Summary and Key Takeaways

1. **LeRobot is the most comprehensive open-source robotics ML framework** available, with 21k+ stars and daily active development by HuggingFace's core team.

2. **SmolVLA (450M) is the sweet spot for T4 16GB:** LoRA fine-tuning requires only ~11-14GB VRAM, well within T4 limits. Official PEFT support makes this straightforward.

3. **LIBERO is the best simulation benchmark for pick-and-place** evaluation, with dedicated object manipulation suites. MetaWorld also provides pick-place tasks.

4. **LeRobotDataset v3.0** is a mature, scalable format with Parquet+MP4 storage, streaming support, and Hub integration. It is becoming the de facto standard for robotics datasets.

5. **Hardware ecosystem is broad:** From $100 SO-100 arms to $16k Unitree G1 humanoids, all unified under the same Python API.

6. **ACT and Diffusion Policy** are lightweight alternatives that fully fit on T4 for both training and inference, but lack the vision-language understanding of VLA models.

7. **OpenVLA is complementary but not competitive on T4:** Its 7B size makes it impractical for fine-tuning on consumer GPUs. SmolVLA is the better choice for resource-constrained setups.

8. **Community is exceptionally active** with global hackathons, Chinese tutorials, ROS2 bridges, and thousands of shared datasets on HF Hub.
