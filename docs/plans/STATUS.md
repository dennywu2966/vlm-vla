# VLA V1 Project Status

> Updated: 2026-02-25

## Summary

Training complete. Root cause of eval failure identified from source analysis. Notebook fixed. Ready for Kaggle eval session.

---

## ✅ Completed

| Task | Details |
|------|---------|
| Project structure (9 tasks) | All notebooks, src, configs, tests created |
| SmolVLA LoRA training | 20k steps, final loss ~0.134 |
| HF Hub push | `dennywu2966/smolvla-libero-object-lora` |
| Base zero-shot eval | 0% on all 10 LIBERO-Object tasks (expected) |
| Root cause analysis | Verified from lerobot 0.4.3 source + HF Hub checkpoint |
| Notebook 05 rewritten | Final version with correct approach |

---

## Root Cause Analysis (Verified)

Investigated `lerobot/scripts/lerobot_eval.py` and `lerobot/policies/factory.py`:

1. **`use_peft=True` already in `config.json`** — lerobot-eval reads it automatically via `PreTrainedConfig.from_pretrained()`. No extra CLI flag needed.

2. **Actual eval failure causes** (most likely):
   - LIBERO not installed in the eval Kaggle session (fresh session, missing `lerobot[libero]`)
   - Original notebook used local path `outputs/latest/checkpoints/last` — not persistent across sessions

3. **Checkpoint is valid**:
   - `adapter_config.json`: correct `base_model_name_or_path = "lerobot/smolvla_base"`, r=32
   - Normalizer stats: **8D state** (matches LiberoProcessorStep output: eef_pos 3 + axisangle 3 + gripper 2)
   - `policy_preprocessor.json`: embeds rename_map (image→camera1, image2→camera2)
   - All preprocessing loaded automatically from HF Hub

---

## 🔴 Next: Kaggle Evaluation Session (~3-4h)

Upload to new Kaggle T4 x2 notebook (internet ON):
- `notebooks/05_evaluation.ipynb`
- `src/vlm_vla/eval_engine.py`

Eval command (embedded in notebook Cell 2):
```bash
lerobot-eval \
  --policy.path=dennywu2966/smolvla-libero-object-lora \
  --policy.device=cuda \
  --env.type=libero \
  --env.task=libero_object \
  --eval.n_episodes=20 \
  --eval.batch_size=1
```

After results: copy `eval_comparison_v2.json` to `results/`, update `results/eval_report.md`.

---

## Reference Numbers

| Model | LIBERO-Object | Notes |
|-------|--------------|-------|
| SmolVLA base (zero-shot) | 0% | confirmed |
| SmolVLA-LoRA 20k steps | ❓ TBD | next session |
| SmolVLA official FT | ~90%+ | HuggingFaceVLA reference |
| OpenVLA-7B official FT | 88.4% | Hejna et al. 2024 |
