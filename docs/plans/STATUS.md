# VLA V1 Project Status

> Updated: 2026-02-25

## Summary

Training complete. Evaluation blocker identified and fixed. Ready for next Kaggle session.

---

## ✅ Completed

| Task | Details |
|------|---------|
| Project structure (9 tasks) | All notebooks, src, configs, tests created |
| SmolVLA LoRA training | 20k steps, final loss ~0.134, checkpoint on HF Hub |
| HF Hub push | `dennywu2966/smolvla-libero-object-lora` |
| Base zero-shot eval | 0% on all 10 LIBERO-Object tasks (expected) |
| OpenVLA reference noted | 88.4% (published, HuggingFaceVLA official) |
| **Eval blocker fixed** | Root cause found, notebook 05 corrected |

---

## Root Cause of Previous Eval Failure

`lerobot-eval` was called **without `--policy.use_peft=true`**.

`lerobot/policies/factory.py:make_policy()` has two branches:
- `use_peft=false` → load full model weights from pretrained_path
- `use_peft=true` → load `adapter_config.json`, read `base_model_name_or_path`, load base policy, apply `PeftModel.from_pretrained`

Without the flag, lerobot tried to load the LoRA adapter repo as a full model → crash.

---

## 📋 Next Kaggle Session: Evaluation (2-4 hours, T4 x2)

Upload to Kaggle:
- `notebooks/05_evaluation.ipynb`
- `src/vlm_vla/eval_engine.py`

**Cell 2**: Eval custom LoRA — `lerobot-eval --policy.use_peft=true --policy.path=dennywu2966/smolvla-libero-object-lora`
**Cell 3**: Eval official model — `lerobot-eval --policy.path=HuggingFaceVLA/smolvla_libero` (upper bound)
**Cell 4-5**: Parse JSON results → comparison table

### Success Criteria
- SmolVLA-LoRA fine-tuned ≥ 50% on LIBERO-Object (minimum viable)
- SmolVLA-LoRA fine-tuned ≥ 70% (V1 target)
- Comparison table: SmolVLA-FT vs SmolVLA-ZS (0%) vs OpenVLA-7B (88.4%) vs Official FT

### If <50% after fix
Options (in order):
1. Check `adapter_config.json` has correct `base_model_name_or_path` (Cell 1 validates this)
2. Train longer (40k steps) from checkpoint — re-run notebook 03
3. Verify camera rename_map was applied correctly during training (key mismatch corrupts signal)

---

## Reference Numbers

| Model | LIBERO-Object | Source |
|-------|--------------|--------|
| SmolVLA base (zero-shot) | 0% | our eval |
| SmolVLA-LoRA fine-tuned | ❓ TBD | next session |
| OpenVLA-7B official FT | 88.4% | Hejna et al. 2024 |
| SmolVLA official FT (HuggingFaceVLA) | ~90%+ | published |
