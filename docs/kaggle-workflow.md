# Kaggle Workflow Guide

## Status

| Session | Status | Output |
|---------|--------|--------|
| 1: Setup & Verify | ✅ Done | GPU OK, LIBERO works |
| 2: Data Exploration | ✅ Done | Dataset: 273k frames, 8D state, 7D action |
| 3-4: Training | ✅ Done | 20k steps, loss 0.134, HF Hub: `dennywu2966/smolvla-libero-object-lora` |
| 5: Evaluation | 🔴 **NEXT** | Notebook fixed, ready to run |

---

## Session 5: Evaluation (NEXT — ~3-4 hours, T4 x2)

### Setup
1. New Kaggle notebook, **GPU T4 x2**, internet ON
2. Upload files:
   - `notebooks/05_evaluation.ipynb`
   - `src/vlm_vla/eval_engine.py`

### Run
Run all cells in order. Cell 2 takes ~2-3h (200 rollouts).

### What to expect
- Cell 0: Install + GPU check (~5 min)
- Cell 1: Verify checkpoint integrity (< 1 min)
- Cell 2: SmolVLA-LoRA eval (~2-3h for 10 tasks × 20 episodes)
- Cell 3: Official SmolVLA eval (~2-3h)
- Cell 4-6: Parse + save results

### After completion
1. Download `eval_comparison_v2.json` from Kaggle output
2. Copy to `results/eval_comparison_v2.json` locally
3. Update `results/eval_report.md` with actual numbers

---

## Key Findings (Verified Locally)

### Why previous eval failed
Most likely causes (priority order):
1. **LIBERO not installed** — eval ran in a fresh session without `lerobot[libero]` extra
2. **Local checkpoint path** — original notebook used `outputs/latest/checkpoints/last` which doesn't persist across Kaggle sessions
3. **Missing `--policy.use_peft=true`** — less likely since `use_peft=True` is in `config.json`

### Checkpoint facts
- `adapter_config.json`: `base_model_name_or_path = "lerobot/smolvla_base"`, r=32
- `config.json`: `use_peft=True` (read automatically by lerobot-eval, no CLI flag needed)
- Normalizer: 8D state stats (eef_pos 3 + eef_axisangle 3 + gripper_qpos 2)
- rename_map embedded in `policy_preprocessor.json`
- All preprocessing is automatic from checkpoint

### Correct eval command
```bash
lerobot-eval \
  --policy.path=dennywu2966/smolvla-libero-object-lora \
  --policy.device=cuda \
  --env.type=libero \
  --env.task=libero_object \
  --eval.n_episodes=20 \
  --eval.batch_size=1
```

---

## Success Criteria (V1)

| Metric | Minimum | Target |
|--------|---------|--------|
| SmolVLA-LoRA success rate | ≥ 50% | ≥ 70% |
| vs zero-shot (0%) | 50x improvement | 70x improvement |
| vs OpenVLA-7B (88.4%) | competitive | competitive |

---

## Tips
- Set `MUJOCO_GL=egl` before any LIBERO code
- If session times out mid-eval, check `outputs/eval/*/eval_info.json` for partial results
- Each task checkpoint is saved independently by lerobot-eval
- Use `--eval.batch_size=2` to halve wall time if VRAM allows (T4 x2 = 32GB)
