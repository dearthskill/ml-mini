"""Entry point: runs the full NSL-KDD reproduction pipeline."""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch

from data import load_and_preprocess
from evaluate import evaluate, plot_category_confusion, plot_fig2
from train import train

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIG2_EPOCHS = 50

PAPER = {"train_multi": "≈ 99.5%", "test_multi": "≈ 70–71.5%",
         "train_binary": "not reported", "test_binary": "≈ 80% (up to ~85%)"}
METRICS = ["train_multi", "test_multi", "train_binary", "test_binary"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=10, help="epochs for the main runs")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    return p.parse_args()


def write_summary(df, epochs, fig2_final, path):
    lines = ["# Results summary", "",
             f"Model 122→500 (sigmoid)→24 (softmax), SGD lr=0.5, batch 1000, "
             f"weight_decay=1e-4, {epochs} epochs, seeds {list(df.seed)}.", "",
             "## Mean ± std over seeds vs paper", "",
             "| Metric | Ours (mean ± std) | Paper |", "|---|---|---|"]
    names = {"train_multi": "Train multiclass accuracy", "test_multi": "Test multiclass accuracy",
             "train_binary": "Train binary (normal vs abnormal)",
             "test_binary": "Test binary (normal vs abnormal)"}
    for m in METRICS:
        lines.append(f"| {names[m]} | {df[m].mean()*100:.2f}% ± {df[m].std(ddof=1)*100:.2f}% "
                     f"| {PAPER[m]} |")
    lines += ["", "## Per seed", "", "| Seed | Train multi | Test multi | Train binary | Test binary |",
              "|---|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(f"| {int(r.seed)} | " + " | ".join(f"{r[m]*100:.2f}%" for m in METRICS) + " |")
    if fig2_final is not None:
        lines += ["", f"## Figure 2 run ({fig2_final[0]} epochs, seed 0)", "",
                  f"Final train multiclass {fig2_final[1]*100:.2f}%, "
                  f"final test multiclass {fig2_final[2]*100:.2f}%. "
                  "See `fig2_accuracy_vs_epoch.png`."]
    lines += ["", "Std is the sample standard deviation (ddof=1)."]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    X_train, y_train, X_test, y_test, class_names, _, y_test_raw = load_and_preprocess()

    # Main result: one run per seed
    rows = []
    for seed in args.seeds:
        t0 = time.time()
        model = train(X_train, y_train, args.epochs, seed, device)
        tr_multi, tr_bin, _ = evaluate(model, X_train, y_train, device)
        te_multi, te_bin, te_pred = evaluate(model, X_test, y_test, device)
        rows.append({"seed": seed, "train_multi": tr_multi, "test_multi": te_multi,
                     "train_binary": tr_bin, "test_binary": te_bin})
        print(f"seed {seed}: train multi {tr_multi:.4f} | test multi {te_multi:.4f} | "
              f"train bin {tr_bin:.4f} | test bin {te_bin:.4f}  ({time.time()-t0:.1f}s)")
        if seed == 0:
            torch.save(model.state_dict(), os.path.join(RESULTS, "model_seed0.pt"))
            plot_category_confusion(te_pred, y_test_raw, class_names,
                                    os.path.join(RESULTS, "confusion_matrix.png"))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "metrics.csv"), index=False)
    print("\nMean ± std:")
    for m in METRICS:
        print(f"  {m:13s} {df[m].mean():.4f} ± {df[m].std(ddof=1):.4f}")

    # Figure 2: separate 50-epoch run, seed 0, accuracy recorded each epoch
    train_acc, test_acc = [], []

    def record(epoch, model, loss):
        train_acc.append(evaluate(model, X_train, y_train, device)[0])
        test_acc.append(evaluate(model, X_test, y_test, device)[0])
        print(f"  fig2 epoch {epoch:2d}: loss {loss:.4f} train {train_acc[-1]:.4f} "
              f"test {test_acc[-1]:.4f}")

    print(f"\nFigure 2 run ({FIG2_EPOCHS} epochs, seed 0):")
    train(X_train, y_train, FIG2_EPOCHS, 0, device, on_epoch_end=record)
    plot_fig2(train_acc, test_acc, os.path.join(RESULTS, "fig2_accuracy_vs_epoch.png"))

    write_summary(df, args.epochs, (FIG2_EPOCHS, train_acc[-1], test_acc[-1]),
                  os.path.join(RESULTS, "summary.md"))
    print(f"\nWrote results to {RESULTS}")


if __name__ == "__main__":
    main()
