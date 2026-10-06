# Task: Reproduce "Using Neural Network for Security Attacks Detection" (CS 229 Final Report) on NSL-KDD

This is a **faithful reproduction** of an existing paper for a university ML mini-project (UE24CS352A). Do **not** add improvements, extra models, tuning, or architectural changes beyond what is specified here. Where the paper is silent, use the values in the "Decisions not specified in the paper" section and nothing else.

Allowed libraries: PyTorch (model + training), scikit-learn (preprocessing / metrics), pandas, numpy, matplotlib.

---

## 1. Dataset

- NSL-KDD: `KDDTrain+.txt` and `KDDTest+.txt` (CSV, no header).
- Each row has 43 columns: 41 features, then the `label` (attack name or `normal`), then a `difficulty` score. **Drop `difficulty`.**
- Expected file location: `data/KDDTrain+.txt`, `data/KDDTest+.txt`. If the files are missing, print a clear message telling the user where to get them (Kaggle / UNB CIC NSL-KDD page) and exit.
- Column names (in order): duration, protocol_type, service, flag, src_bytes, dst_bytes, land, wrong_fragment, urgent, hot, num_failed_logins, logged_in, num_compromised, root_shell, su_attempted, num_root, num_file_creations, num_shells, num_access_files, num_outbound_cmds, is_host_login, is_guest_login, count, srv_count, serror_rate, srv_serror_rate, rerror_rate, srv_rerror_rate, same_srv_rate, diff_srv_rate, srv_diff_host_rate, dst_host_count, dst_host_srv_count, dst_host_same_srv_rate, dst_host_diff_srv_rate, dst_host_same_src_port_rate, dst_host_srv_diff_host_rate, dst_host_serror_rate, dst_host_srv_serror_rate, dst_host_rerror_rate, dst_host_srv_rerror_rate, label, difficulty

### Attack categories (paper Table 2; `*` = test-only)
- **DoS:** back, land, neptune, pod, smurf, teardrop, mailbomb\*, apache2\*, processtable\*, udpstorm\*
- **Probe:** ipsweep, nmap, portsweep, satan, mscan\*, saint\*
- **R2L:** ftp_write, guess_passwd, imap, multihop, phf, spy, warezclient, warezmaster, sendmail\*, named\*, snmpgetattack\*, snmpguess\*, xlock\*, xsnoop\*, worm\*
- **U2R:** buffer_overflow, loadmodule, perl, rootkit, httptunnel\*, ps\*, sqlattack\*, xterm\*

## 2. Preprocessing (paper §4.1)

1. Separate features (X) from `label` (Y).
2. One-hot encode `protocol_type`, `service`, `flag`. Fit the encoder on the **training set** only; unseen test categories → all-zeros (`handle_unknown="ignore"`).
   - The result must be **122 input columns** (38 numeric + 3 protocol + 70 service + 11 flag). Assert this.
3. Standardize all 122 columns to zero mean / unit variance (`StandardScaler` fit on train, applied to test).
4. Labels → **24 classes**:
   - index 0: `normal`
   - indices 1–22: the 22 attack names present in the training set (sorted alphabetically)
   - index 23: `other` — every test-set attack not seen in training maps here
   - Assert there are exactly 22 distinct attack names in training.
   - Note: `other` never occurs in training data; this is faithful to the paper, do not change it.
5. Print a sanity check matching paper Table 3 (counts per category):
   - Train: Normal 67343, DoS 45927, Probe 11656, R2L 995, U2R 52
   - Test:  Normal 9710, DoS 7458, R2L 2754, Probe 2421, U2R 200
   - (Test totals may differ slightly from these if a few test-only attack names aren't in the category list above; report any unmapped names.)

## 3. Model (paper §4.2, Figure 1)

- Fully connected: **122 → 500 → 24**
- Hidden activation: **sigmoid**
- Output: **softmax** (in PyTorch: output logits + `CrossEntropyLoss`; minimizing cross-entropy = the paper's gradient *ascent* on log-likelihood)

## 4. Training (paper §4.2)

- Optimizer: plain mini-batch **SGD**, **learning rate 0.5**, no momentum
- **Batch size 1000**, shuffled each epoch
- **10 epochs** (main result)
- Regularized (L2) — see decisions below

## 5. Decisions not specified in the paper (our choices — document them in the README and write-up)

- **Weight initialization:** Xavier/Glorot uniform for both layers (suits sigmoid); biases = 0
- **L2 regularization:** `weight_decay = 1e-4` in SGD
- **Random seeds:** paper reports results vary a lot between runs (~70% typical, sometimes >85%). Run **5 seeds (0–4)** and report each plus mean ± std.
- Device: CPU is fine; use CUDA if available.

## 6. Evaluation (paper §4.2 end, §5.2)

Compute on both train and test:
1. **Multiclass accuracy** over the 24 classes.
2. **Binary (normal vs abnormal) accuracy:** predicted class == `normal` vs not, compared with true label == `normal` vs not.

Paper's reported numbers to compare against:
- Train multiclass ≈ 99.5% (near 100%)
- Test multiclass ≈ 70–71.5%
- Test binary ≈ 80% (up to ~85% on some runs)

## 7. Figures

Reproduce **Figure 2**: a separate **50-epoch** run (same settings, seed 0) recording multiclass accuracy per epoch → two plots: (a) train accuracy vs epoch, (b) test accuracy vs epoch. Save as `results/fig2_accuracy_vs_epoch.png`.

Also save a confusion matrix for the 5 broad categories (Normal/DoS/Probe/R2L/U2R) on the test set for the presentation: `results/confusion_matrix.png`.

## 8. Outputs

- `results/metrics.csv` — one row per seed: train_multi, test_multi, train_binary, test_binary
- `results/summary.md` — table of mean ± std vs the paper's numbers
- Figures as above
- Save the seed-0 model weights to `results/model_seed0.pt`

## 9. Repository structure

```
.
├── README.md
├── requirements.txt
├── data/              # NSL-KDD files go here (gitignored)
├── src/
│   ├── data.py        # loading + preprocessing
│   ├── model.py       # network definition
│   ├── train.py       # training loop
│   ├── evaluate.py    # metrics + plots
│   └── main.py        # entry point: runs everything
└── results/
```

- `python src/main.py` runs the whole pipeline end to end.
- Add a `--epochs` and `--seeds` CLI option (defaults 10 and 0–4).
- Add `data/` to `.gitignore`.

## 10. README must include

- Project title, problem statement (one paragraph), dataset description + where to download it
- Setup: `pip install -r requirements.txt`, where to place data files
- How to run, and expected runtime
- The "Decisions not specified in the paper" list
- Results table vs the paper

## 11. Out of scope

No hyperparameter search, no other models, no class balancing, no feature selection, no extra datasets. Reproduction only.
