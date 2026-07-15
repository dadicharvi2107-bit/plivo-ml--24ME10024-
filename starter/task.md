# Task List - LLM Speedrun

- `[x]` Run 1: Enable Weight Tying (saves parameter capacity).
- `[x]` Run 2: Add Cosine Learning Rate schedule with Linear Warmup.
- `[x]` Run 3: Implement Weight Decay splitting (exclude biases and 1D params).
- `[x]` Run 4-6: Scale up model dimensions under cap (6 layers, 8 heads, 160 embd, 256 block size) and tune learning rate (6e-4). Achieved dev BPB 2.1984.
- `[x]` Run 7: Adjust model capacity baseline to fit BPE (n_embd=144, dev BPB 2.9228 at 400 steps).
- `[x]` Run 8: Swapped in optimized BPE Tokenizer (vocab=2000, dev BPB 2.4545 at 400 steps, showing massive gain).
- `[x]` Run 9: Train final BPE candidate for 2000 steps (dev BPB = 2.0383).
- `[x]` Run 10: Tune AdamW betas to (0.9, 0.95) (dev BPB = 2.0461).
- `[x]` Verify lossless BPE roundtrip & evaluate final model.
- `[x]` Restore the best performing checkpoint (Run 9's 2.0383 dev BPB) as `ckpt.pt`.
- `[x]` Commit all checkpoints, merges, logs, and push to GitHub master branch.
