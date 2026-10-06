# Results summary

Model 122→500 (sigmoid)→24 (softmax), SGD lr=0.5, batch 1000, weight_decay=1e-4, 10 epochs, seeds [0, 1, 2, 3, 4].

## Mean ± std over seeds vs paper

| Metric | Ours (mean ± std) | Paper |
|---|---|---|
| Train multiclass accuracy | 98.02% ± 0.19% | ≈ 99.5% |
| Test multiclass accuracy | 68.76% ± 0.04% | ≈ 70–71.5% |
| Train binary (normal vs abnormal) | 98.25% ± 0.21% | not reported |
| Test binary (normal vs abnormal) | 75.38% ± 0.51% | ≈ 80% (up to ~85%) |

## Per seed

| Seed | Train multi | Test multi | Train binary | Test binary |
|---|---|---|---|---|
| 0 | 97.89% | 68.71% | 98.08% | 75.05% |
| 1 | 98.13% | 68.77% | 98.41% | 76.03% |
| 2 | 97.75% | 68.74% | 97.97% | 74.78% |
| 3 | 98.13% | 68.81% | 98.35% | 75.77% |
| 4 | 98.20% | 68.75% | 98.42% | 75.28% |

## Figure 2 run (50 epochs, seed 0)

Final train multiclass 99.15%, final test multiclass 70.03%. See `fig2_accuracy_vs_epoch.png`.

Std is the sample standard deviation (ddof=1).
