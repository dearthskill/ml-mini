# Using Neural Networks for Security Attack Detection — NSL-KDD Reproduction

Faithful reproduction of the CS 229 final report *"Using Neural Network for Security Attacks Detection"* on the NSL-KDD dataset (UE24CS352A ML mini-project). No extra models, tuning or architectural changes beyond the paper.

## Problem statement

Network intrusion detection systems must tell normal traffic apart from attacks, and ideally identify which attack is happening. Given 41 features describing a network connection (protocol, service, byte counts, error rates, host-level statistics, …), we train a single-hidden-layer neural network to classify each connection as `normal` or one of the specific attack types seen in training. Performance is measured both as multiclass accuracy and as binary normal-vs-abnormal accuracy. The test set includes attack types never seen during training, which tests how well the model generalises.

## Dataset

**NSL-KDD** is a cleaned version of KDD Cup '99 with duplicate records removed.
- `KDDTrain+.txt`: 125,973 connections, 22 attack types plus normal
- `KDDTest+.txt`: 22,544 connections, including 17 attack types that appear only in the test set
- Each row has 41 features, the label and a difficulty score. The difficulty score is dropped.

Download from the [UNB CIC NSL-KDD page](https://www.unb.ca/cic/datasets/nsl.html), Kaggle ("NSL-KDD"), or the mirror at <https://github.com/defcom17/NSL_KDD>.

## Setup

```bash
pip install -r requirements.txt
```
Put the two files at `data/KDDTrain+.txt` and `data/KDDTest+.txt`. The `data/` directory is gitignored.

## How to run

```bash
python src/main.py                       # 5 seeds × 10 epochs + 50-epoch Figure 2 run
python src/main.py --epochs 10 --seeds 0 1 2 3 4
```
**Expected runtime:** about 20 s on a laptop CPU (Apple M-series), of which each 10-epoch run takes about 1 s. CUDA is used automatically if it's available.

Outputs in `results/`: `metrics.csv`, `summary.md`, `fig2_accuracy_vs_epoch.png`, `confusion_matrix.png`, `model_seed0.pt`.

## Method (as in the paper)

- **Preprocessing:** One-hot encode `protocol_type`, `service` and `flag`, fitting on the training set only; unseen test values become all zeros. This gives 122 inputs: 38 numeric + 3 + 70 + 11. All 122 columns are then standardized with a `StandardScaler` fit on the training set.
- **Labels:** 24 classes. Index 0 is `normal`, indices 1–22 are the 22 training attacks sorted alphabetically, and index 23 is `other`, which covers every test-only attack.
- **Model:** 122 → 500 (sigmoid) → 24 (softmax via `CrossEntropyLoss`).
- **Training:** Plain mini-batch SGD, learning rate 0.5, no momentum, batch size 1000, reshuffled every epoch, 10 epochs, L2 regularisation.

## Decisions not specified in the paper

- **Weight initialization:** Xavier/Glorot uniform for both layers; biases set to 0.
- **L2 regularization:** `weight_decay = 1e-4` in SGD.
- **Random seeds:** 5 seeds (0–4), each reported along with mean ± std (sample std, ddof=1). Figure 2 uses a separate 50-epoch run with seed 0.
- **Device:** CPU, or CUDA if available.
- **Confusion matrix** (not in the paper; for the presentation): test labels grouped into the 5 Table 2 categories (seed 0).

## Results (5 seeds, 10 epochs)

| Metric | Ours (mean ± std) | Paper |
|---|---|---|
| Train multiclass accuracy | 98.02% ± 0.19% | ≈ 99.5% |
| Test multiclass accuracy | 68.76% ± 0.04% | ≈ 70–71.5% |
| Train binary accuracy | 98.25% ± 0.21% | not reported |
| Test binary accuracy | 75.38% ± 0.51% | ≈ 80% (up to ~85%) |

In the 50-epoch Figure 2 run, accuracy reaches 99.15% on train and 70.03% on test, which matches the paper's numbers more closely. See `results/summary.md` for per-seed numbers.

**Observations:**
- Our results vary much less between seeds than the paper's did. We see no runs above 85%.
- The `other` class never appears in training, so the model never predicts it. Every test-only attack, about 17% of the test set, is therefore misclassified in the multiclass metric.
- Most of the binary errors are R2L and U2R attacks being predicted as normal.

## Repository structure

```
src/data.py      loading + preprocessing + Table 3 sanity check
src/model.py     network definition
src/train.py     training loop
src/evaluate.py  metrics + plots
src/main.py      entry point
```
