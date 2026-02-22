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
