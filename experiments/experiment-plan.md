# Fixed compute experiment plan

Compare six model architectures at the estimated forward-and-backward compute budget of the completed 13.49M-parameter baseline. Select the best allocation by final loss on the same fixed validation sample.

## Compute budget

The baseline processed **100,007,936 training tokens**. Its measured cost is **66,846,720 FLOPs per token**.

```text
C = 66,846,720 × 100,007,936
  = 6,685,202,495,569,920 FLOPs
  ≈ 6.6852 × 10^15 FLOPs
```

Counts cover forward, loss calculation, and backward; they exclude optimizer updates, validation, and generation. These are PyTorch formula-based estimates, not measured hardware instructions or equal wall-clock budgets.

Measurements used PyTorch 2.14.0+cu130 on an A10G, BF16, SDPA, context length 256, and batch size 1. All six FLOPs/token values matched exactly between the [2-update measurements](flops-20261122-192407/summary.json) and [10-update measurements](flops-10steps-20261122-192555/summary.json), using code revision `47d26d9`.

## Shared training configuration

- Vocabulary size: 8,192, using the same existing tokenizer.
- Sequence length: 256.
- Head dimension: 64.
- BF16 and SDPA.
- Effective batch: 64 sequences, initially microbatch 64 × accumulation 1.
- Tokens per optimizer update: **16,384**.
- Learning-rate schedule ends at each run's planned total updates.

If microbatch 64 does not fit, use 32 × 2 or 16 × 4 to retain the same effective batch and planned update count. Use the same validation data and evaluation procedure for every model.

## Model configurations and budgets

```text
steps = floor(C / FLOPs_per_token / 16,384)
actual_tokens = steps × 16,384
estimated_compute = actual_tokens × FLOPs_per_token
```

| Model | Feature width | Layers | Heads | Parameters | FLOPs/token | Steps | Actual training tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Smallest | 128 | 2 | 2 | 2,533,888 | 9,568,256 | 42,644 | 698,679,296 |
| Small | 256 | 3 | 4 | 6,634,752 | 29,491,200 | 13,835 | 226,672,640 |
| Baseline | 384 | 4 | 6 | 13,490,432 | 66,846,720 | 6,104 | 100,007,936 |
| Medium | 512 | 6 | 8 | 27,430,912 | 149,422,080 | 2,730 | 44,728,320 |
| Large | 640 | 8 | 10 | 50,026,752 | 285,736,960 | 1,428 | 23,396,352 |
| Largest | 768 | 12 | 12 | 97,806,848 | 580,386,816 | 703 | 11,517,952 |

All planned compute totals are at or below C and within 0.027% of it after rounding. For a single uninterrupted run, set both `total_steps` and `job_steps` to the table value. For split jobs, keep `total_steps` fixed and allocate `job_steps` across invocations.

