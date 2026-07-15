# RUNLOG

## Run 0 - Baseline (Adam)

Hypothesis:
Measure the baseline performance using standard Adam optimizer and no gradient clipping.

Changes:
- Baseline model with standard Adam optimizer.

Training:
- 2000 optimizer steps
- 1,339,840 parameters
- flat learning rate of 3e-4
- batch size 8, block size 128

Results:
- Dev BPB: 2.3718

Conclusion:
Baseline established.

## Run 1 - AdamW + Gradient Clipping

Hypothesis:
Changing the optimizer to AdamW and introducing gradient clipping (max norm 1.0) will stabilize training and improve the final bits-per-byte (bpb) score.

Changes:
- Switched optimizer from Adam to AdamW (weight decay 0.01).
- Added gradient norm clipping with max norm 1.0.

Training:
- 2000 optimizer steps
- 1,339,840 parameters
- flat learning rate of 3e-4
- batch size 8, block size 128

Results:
- Dev BPB: 2.3531
- Train Loss: 1.7141

Conclusion:
Successfully reduced dev bpb by 0.0187. The combination of AdamW weight decay and gradient clipping stabilizes gradient updates.

## Run 2 - Weight Tying Only

Hypothesis:
Enabling weight tying between input embeddings and the output head projection will reduce parameter count and serve as a baseline for scaling up the model size later. On its own, it might slightly degrade performance because it restricts parameter capacity.

Changes:
- Set `tie_weights = True` in `Config`.

Training:
- 2000 optimizer steps
- 1,298,880 parameters (saved 40,960 parameters)
- flat learning rate of 3e-4
- batch size 8, block size 128

Results:
- Dev BPB: 2.3962
- Train Loss: 1.7522

Conclusion:
As expected, weight tying alone slightly degrades performance (from 2.3531 to 2.3962 bpb) because it restricts model capacity. However, it successfully reduces parameter count by 40.9K, which will be utilized to build a deeper and wider model in subsequent runs.

## Run 3 - Weight Tying + Cosine LR Schedule

Hypothesis:
Adding linear warmup and cosine learning rate decay will stabilize training and help convergence, leading to a better dev bpb.

Changes:
- Added `get_lr` helper implementing 200 steps linear warmup and cosine decay to 10% of peak learning rate.
- Peak learning rate kept at baseline `3e-4`.

Training:
- 2000 optimizer steps
- 1,298,880 parameters
- Cosine decay learning rate (peak 3e-4)
- batch size 8, block size 128

Results:
- Dev BPB: 2.6100
- Train Loss: 1.9104

Conclusion:
Dev bpb degraded from 2.3962 to 2.6100. This is a very valuable finding: because of warmup and cosine decay, the average learning rate throughout training was much lower than the baseline's flat 3e-4. Small models need higher learning rates to learn effectively. A peak LR of 3e-4 was too small when decayed, leading to severe underfitting. We need to increase the peak learning rate in subsequent runs.
