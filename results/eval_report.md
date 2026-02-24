# SmolVLA LIBERO-Object Evaluation Results

## Summary

This report presents the evaluation results of SmolVLA on the LIBERO-Object benchmark.

### Models Evaluated

| Model | Description | Success Rate |
|-------|-------------|--------------|
| smolvla_base_zeroshot | SmolVLA base model (no fine-tuning) | 0.0% |
| openvla_7b_published | OpenVLA-7B (from Hejna et al. 2024) | 88.4% |

### Results Details

**SmolVLA Base (Zero-shot)**
- All 10 LIBERO-Object tasks: 0% success rate
- This is expected as the base model was not fine-tuned on LIBERO

**Fine-tuned Models (Evaluation Failed)**
- Custom LoRA adapter (dennywu2966/smolvla-libero-object-lora): Failed to produce results
- Official HuggingFaceVLA/smolvla_libero: Also failed to produce results

### Technical Issues

The lerobot-eval command appears to have compatibility issues with LoRA adapter checkpoints. Both:
1. Our custom-trained LoRA adapter (5.8MB, r=32)
2. The official HuggingFaceVLA/smolvla_libero fine-tuned model

failed to produce evaluation results. The eval process either:
- Crashes silently before producing output
- Times out before loading the model

### Recommendations

1. **For custom LoRA evaluation**: May need to merge LoRA weights into base model before evaluation
2. **For official models**: Check if lerobot-eval supports the specific model format
3. **Alternative approach**: Use the lerobot API directly for inference instead of lerobot-eval CLI

### Training Details

**Custom LoRA Training (Completed)**
- Base model: lerobot/smolvla_base
- Dataset: HuggingFaceVLA/libero (LIBERO-Object subset)
- Training steps: 20,000
- LoRA rank: r=32
- Final training loss: ~0.134
- Checkpoint: https://huggingface.co/dennywu2966/smolvla-libero-object-lora

---

*Generated: 2026-02-24*
