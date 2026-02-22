# VLA Model Training Design

> Date: 2025-02-22
> Status: Approved

## Task Definition

### Goal
LoRA fine-tune **SmolVLA (450M)** from the LeRobot ecosystem on simulated **pick-and-place** tasks. Use **OpenVLA 7B** as inference-only baseline for comparison. Architecture and pipeline must support future scaling to more complex tasks, stronger models, RL, and real hardware (including humanoid robots).

### Constraints
| Item | Value |
|------|-------|
| Compute | Kaggle free T4 (16GB, ~30h/week) + Aliyun PAI ¥300 budget |
| Model | SmolVLA 450M (LoRA train) + OpenVLA 7B (4-bit inference only) |
| Experience | DL beginner (can read code, hasn't built training pipeline independently) |
| Hardware | Simulation only in V1; architecture must be real-hardware-ready |
| Speech | Deferred to V2 |
| RL | Deferred to V3 |

### Evolution Roadmap
| Phase | Content |
|-------|---------|
| V1 | SmolVLA LoRA + LIBERO pick-and-place (BC) |
| V1.5 | DAgger automatic data augmentation |
| V2 | Speech command integration |
| V3 | Isaac Lab + dual-arm + RL exploration |
| V3.5 | Humanoid full-body control (locomotion + manipulation) |
| V4 | Real hardware deployment + sim2real |

### Success Criteria (V1)
- Pick-and-place success rate ≥ 70% on LIBERO-Object (10 tasks, 20 rollouts each)
- Training completes within Kaggle T4 free quota (~30h)
- Evaluation pipeline produces reproducible metrics
- Code structure supports swapping model/env/action-space without rewriting core

---

## Section 1: Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      vlm-vla/                                │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌───────────┐    ┌────────────────┐    ┌────────────────┐   │
│  │  LIBERO    │───▶│ LeRobotDataset │───▶│  SmolVLA       │   │
│  │  (MuJoCo)  │    │ (Parquet+MP4)  │    │  LoRA Train    │   │
│  └───────────┘    └────────────────┘    └────────────────┘   │
│       │                                        │             │
│       │              ┌────────────────┐        │             │
│       └─────────────▶│  Eval Engine   │◀───────┘             │
│                      │  (rollout eval) │                     │
│                      └────────────────┘                      │
│                             │                                │
│                      ┌────────────────┐                      │
│                      │  OpenVLA 7B    │  (inference-only     │
│                      │  4-bit baseline│   comparison)        │
│                      └────────────────┘                      │
│                                                              │
│  Abstraction layers (future-proof):                          │
│  ├── EnvAdapter    ← swap LIBERO / Isaac Lab / real          │
│  ├── ActionSpace   ← 7D gripper → 30D humanoid              │
│  ├── ModelAdapter  ← SmolVLA / OpenVLA / future models       │
│  └── DataPipeline  ← LeRobotDataset v3.0 standard           │
└──────────────────────────────────────────────────────────────┘
```

### Key Design Decisions
- **SmolVLA as primary** (not OpenVLA): 450M fits T4 comfortably; OpenVLA 7B needs 4-bit quantization + many workarounds on T4, and is officially unsupported in that config
- **LeRobotDataset v3.0** as canonical data format: Parquet + MP4, HuggingFace native, used by both SmolVLA and can be adapted for OpenVLA
- **Modular adapters**: EnvAdapter, ActionSpace, ModelAdapter are thin interfaces so V2-V4 swaps don't require rewriting core training/eval code

---

## Section 2: Data Pipeline

### Data Flow

```
LIBERO simulation
      │
      ▼
Expert Demos (50 demos/task × 10 tasks = 500 trajectories)
      │
      ▼
LeRobotDataset v3.0
├── meta/
│   ├── info.json         ← dataset info, feature definitions
│   ├── episodes.jsonl    ← per-trajectory metadata
│   └── tasks.jsonl       ← task descriptions (natural language)
├── data/
│   └── train-*.parquet   ← states, actions, timestamps
└── videos/
    └── chunk-*/          ← MP4 videos (observation images)
```

### Data Fields

| Field | Type | Content | Future Extension |
|-------|------|---------|-----------------|
| `observation.image` | MP4 frames | 224×224 RGB | Add wrist cam, depth |
| `observation.state` | float[7] | joint positions | Extend to 30D |
| `action` | float[7] | delta joint commands | Extend to 30D |
| `language_instruction` | string | "pick up the red block" | Speech transcription |
| `timestamp` | float | seconds | Unchanged |

### Data Collection Strategy
1. Use LIBERO's built-in scripted expert policy for initial demos
2. Store in LeRobotDataset format for SmolVLA compatibility
3. For OpenVLA comparison: convert on-the-fly via adapter (LeRobot → RLDS bridge)

---

## Section 3: Training Pipeline

### Line 1: SmolVLA LoRA Fine-tune (Primary, Kaggle T4)

```
LeRobotDataset ──▶ SmolVLA 450M (pretrained weights)
                        │
                   LoRA Adapters (r=32~64)
                        │
                   LIBERO-Object 10 tasks
                        │
                   Output: LoRA weights (~50MB)
```

**Config:**
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Precision | fp16 | T4 supports fp16, not bf16 |
| Batch size | 4-8 | Fits in 16GB with LoRA |
| LoRA rank | 32-64 | Balance capacity vs memory |
| Learning rate | 2e-5 | SmolVLA recommended |
| Steps | 30,000-50,000 | ~10-20h on T4 |
| Image size | 224×224 | SmolVLA default |

### Line 2: OpenVLA 7B Inference-only Baseline

```
LeRobotDataset ──▶ OpenVLA 7B (4-bit quantized)
                        │
                   Zero-shot / few-shot inference
                        │
                   Output: success rate comparison
```

No training — just load pretrained weights, run rollouts, compare against fine-tuned SmolVLA.

### Kaggle Session Strategy
- Session 1-2: Environment setup + data collection (~2h)
- Session 3-5: SmolVLA LoRA training (~15h)
- Session 6: OpenVLA inference baseline (~3h)
- Session 7-8: Evaluation rollouts (~6h)
- Reserve: Aliyun PAI ¥300 for longer training runs or hyperparameter sweep

---

## Section 4: Evaluation & Verification

### Three-Layer Evaluation

```
Layer 1: Training Monitoring (every N steps)
├── training loss curve
├── learning rate schedule
└── gradient norm

Layer 2: Rollout Evaluation (every checkpoint)
├── success rate per task (20 rollouts × 10 tasks)
├── average steps to completion
└── failure mode classification (miss/drop/timeout/collision)

Layer 3: Comparative Analysis
├── SmolVLA fine-tuned vs SmolVLA zero-shot
├── SmolVLA fine-tuned vs OpenVLA 7B zero-shot
└── per-task breakdown + aggregated metrics
```

### Success Metric
- **Primary**: average success rate across 10 LIBERO-Object tasks
- **Target**: ≥ 70% (SOTA on LIBERO is ~90% with full resources)
- **Minimum viable**: ≥ 50% on at least 5 tasks

### Evaluation Script Output
```
Task                          | SmolVLA-FT | SmolVLA-ZS | OpenVLA-ZS
------------------------------|------------|------------|----------
pick_up_red_block             |     85%    |     20%    |     45%
place_on_plate                |     75%    |     15%    |     40%
...
------------------------------|------------|------------|----------
Average                       |     72%    |     18%    |     42%
```

### RL Future Path (V3)
| Phase | Method | When |
|-------|--------|------|
| V1 | Pure BC (supervised) | Now |
| V1.5 | DAgger (online data augmentation) | After V1 success |
| V3 | PPO/SAC on Isaac Lab | After V2 |

---

## Section 5: Scalability & Evolution

### Action Space Abstraction
```python
class ActionSpace:
    """V1: 7D gripper → V4: 30D humanoid"""
    def __init__(self, config):
        self.dims = config.action_dims
        self.groups = config.action_groups  # {"left_arm": 7, "right_arm": 7, ...}
        self.normalize = config.normalize
```

### Simulation Environment Evolution
| Phase | Simulator | Reason |
|-------|-----------|--------|
| V1-V2 | LIBERO (MuJoCo) | Mature, LeRobot native support, good docs |
| V3+ | Isaac Lab (NVIDIA) | Best humanoid support (H1/G1 models), massive parallel RL |

Both share MuJoCo physics engine — migration cost is low.

### Model Evolution
| Phase | Model | Size | Training |
|-------|-------|------|----------|
| V1 | SmolVLA | 450M | LoRA on T4 |
| V3 | Custom VLA (MoE) | 1-3B | Full fine-tune on PAI |
| V3.5 | Humanoid VLA | TBD | RL + BC on Isaac Lab |

---

## Section 6: Hardware Purchasing Guide

### Phase 1: Learning & Validation (¥2,000-5,000)
| Option | Price | Recommendation | Notes |
|--------|-------|---------------|-------|
| **SO-100 (LeRobot official)** | ~¥800-1,500 | ★★★★★ | 5 DoF + gripper, LeRobot native, fastest to start |
| Koch v1.1 | ~¥2,000-3,000 | ★★★★ | 6 DoF, LeRobot supported, higher precision |
| Used WidowX 250 | ~¥3,000-5,000 | ★★★ | 6 DoF, OpenVLA's official arm |

**Recommendation: SO-100** — cheap, LeRobot native driver, active community.

### Phase 2: Capability Upgrade (¥5,000-20,000)
| Option | Price | Notes |
|--------|-------|-------|
| Moss v1 (dual-arm) | ~¥5,000-8,000 | Dual SO-100 + teleoperation |
| ALOHA 2 open-source replica | ~¥10,000-20,000 | Dual-arm + teleoperation, Mobile ALOHA path |

### Phase 3: Humanoid (¥30,000+)
| Option | Price | Notes |
|--------|-------|-------|
| Unitree G1 (education) | ~¥100,000+ | 23 DoF, Isaac Lab ecosystem |
| Fourier GR-1 | ~¥200,000+ | 40 DoF, domestic ecosystem |
| Self-assembled (open-source) | ~¥30,000-50,000 | Stompy, Berkeley Humanoid |

### Accessories (All Phases)
| Equipment | Price | Purpose |
|-----------|-------|---------|
| USB cameras ×2 (720p+) | ~¥200 | Fixed view + wrist view |
| 3D-printed workspace | ~¥100-300 | Standardized workspace |
| Object kit | ~¥50-100 | Blocks, cups for pick-and-place |

### Purchasing Advice
1. **Don't buy hardware during V1-V2** — pure simulation, save money and time
2. **Buy SO-100 when validating sim2real in V3** (~¥1,000) — minimum viable hardware
3. **Upgrade to dual-arm or humanoid** only after proving sim2real works
4. **Domestic channels**: Taobao search "SO-100 机械臂" or "LeRobot 套件"
