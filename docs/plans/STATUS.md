# VLA V1 Project Status

> Updated: 2026-02-25

## Summary

Training complete, evaluation blocked. Need to fix LoRA eval approach.

---

## ✅ Completed

| Task | Details |
|------|---------|
| Project structure (9 tasks) | All notebooks, src, configs, tests created |
| SmolVLA LoRA training | 20k steps, final loss ~0.134, checkpoint on HF Hub |
| HF Hub push | `dennywu2966/smolvla-libero-object-lora` |
| Base zero-shot eval | 0% on all 10 LIBERO-Object tasks (expected) |
| OpenVLA reference noted | 88.4% (published, HuggingFaceVLA official) |

---

## ❌ Blocker: LoRA Eval Fails

**Root cause**: `lerobot-eval --policy.path=<lora_checkpoint>` loads only adapter weights; the CLI doesn't know to merge them with the base model first. Crashes silently or times out.

Tested:
- `dennywu2966/smolvla-libero-object-lora` → failed
- `HuggingFaceVLA/smolvla_libero` → also failed (same CLI issue)

**Fix**: Merge LoRA into base model first (PEFT `merge_and_unload()`), save merged weights, then eval. Notebook 05 needs rewriting.

---

## 📋 Execution Plan (Next Kaggle Session)

### Session: Fix Evaluation (2-4 hours, T4 x2)

Upload to Kaggle:
- `notebooks/05_evaluation.ipynb` (after fix below)
- `src/vlm_vla/eval_engine.py`

**Step 1**: Merge LoRA → full model, save locally
**Step 2**: Run `lerobot-eval` on merged model
**Step 3**: Parse results → build comparison table
**Step 4**: Push merged model to HF Hub as `dennywu2966/smolvla-libero-object-merged`

### Success Criteria
- SmolVLA fine-tuned ≥ 50% on LIBERO-Object (minimum viable)
- SmolVLA fine-tuned ≥ 70% (V1 target)
- Comparison table: SmolVLA-FT vs SmolVLA-ZS (0%) vs OpenVLA-7B (88.4%)

### If <50% after fix
Options:
1. Train longer (40k steps) — re-run notebook 03 from checkpoint
2. Lower LoRA rank to r=16 and retrain (less overfitting risk)
3. Check if rename_map was applied correctly during training (camera key mismatch could corrupt training signal)

---

## Reference Numbers

| Model | LIBERO-Object | Source |
|-------|--------------|--------|
| SmolVLA base (zero-shot) | 0% | our eval |
| SmolVLA-LoRA fine-tuned | ❓ TBD | blocked |
| OpenVLA-7B official FT | 88.4% | Hejna et al. 2024 |
| SmolVLA official FT (HuggingFaceVLA) | ~90%+ | published |
