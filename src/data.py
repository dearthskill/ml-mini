"""Loading and preprocessing of NSL-KDD (paper §4.1)."""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
TRAIN_FILE = os.path.join(DATA_DIR, "KDDTrain+.txt")
TEST_FILE = os.path.join(DATA_DIR, "KDDTest+.txt")

COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty",
]
CATEGORICAL = ["protocol_type", "service", "flag"]

# Paper Table 2 (* = test-only attacks are included here too)
CATEGORIES = {
    "DoS": ["back", "land", "neptune", "pod", "smurf", "teardrop",
            "mailbomb", "apache2", "processtable", "udpstorm"],
    "Probe": ["ipsweep", "nmap", "portsweep", "satan", "mscan", "saint"],
    "R2L": ["ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy", "warezclient",
            "warezmaster", "sendmail", "named", "snmpgetattack", "snmpguess", "xlock",
            "xsnoop", "worm"],
    "U2R": ["buffer_overflow", "loadmodule", "perl", "rootkit",
            "httptunnel", "ps", "sqlattack", "xterm"],
}
ATTACK_TO_CATEGORY = {a: c for c, attacks in CATEGORIES.items() for a in attacks}
CATEGORY_NAMES = ["Normal", "DoS", "Probe", "R2L", "U2R"]

PAPER_TABLE3 = {
    "train": {"Normal": 67343, "DoS": 45927, "Probe": 11656, "R2L": 995, "U2R": 52},
    "test": {"Normal": 9710, "DoS": 7458, "Probe": 2421, "R2L": 2754, "U2R": 200},
}


def to_category(label):
    """Map a raw label to its broad category ('Unmapped' if not in Table 2)."""
    if label == "normal":
        return "Normal"
    return ATTACK_TO_CATEGORY.get(label, "Unmapped")


def load_raw():
    for path in (TRAIN_FILE, TEST_FILE):
        if not os.path.exists(path):
            print(f"ERROR: missing {path}\n"
                  "Download KDDTrain+.txt and KDDTest+.txt from the UNB CIC NSL-KDD page\n"
                  "(https://www.unb.ca/cic/datasets/nsl.html) or Kaggle, and place them in data/.")
            sys.exit(1)
    train = pd.read_csv(TRAIN_FILE, header=None, names=COLUMNS).drop(columns="difficulty")
    test = pd.read_csv(TEST_FILE, header=None, names=COLUMNS).drop(columns="difficulty")
    return train, test


def print_category_counts(train_labels, test_labels):
    print("\nSanity check vs paper Table 3 (counts per category):")
    ok = True
    for split, labels in (("train", train_labels), ("test", test_labels)):
        cats = labels.map(to_category)
        counts = cats.value_counts()
        for name in CATEGORY_NAMES:
            got, exp = int(counts.get(name, 0)), PAPER_TABLE3[split][name]
            match = "OK" if got == exp else "MISMATCH"
            ok &= got == exp
            print(f"  {split:5s} {name:6s} got {got:6d}  paper {exp:6d}  {match}")
        unmapped = sorted(labels[cats == "Unmapped"].unique())
        if unmapped:
            ok = False
            print(f"  {split}: unmapped attack names: {unmapped} "
                  f"({int((cats == 'Unmapped').sum())} rows)")
    print(f"  Table 3 check: {'PASSED' if ok else 'see mismatches above'}\n")
    return ok


def load_and_preprocess():
    """Return standardized X_train, y_train, X_test, y_test, class names, raw labels."""
    train, test = load_raw()

    # 1. Separate features and labels
    y_train_raw, y_test_raw = train.pop("label"), test.pop("label")

    # 2. One-hot encode categorical columns (fit on train only)
    encoder = OneHotEncoder(handle_unknown="ignore")
    encoder.fit(train[CATEGORICAL])
    numeric_cols = [c for c in train.columns if c not in CATEGORICAL]

    def encode(df):
        onehot = encoder.transform(df[CATEGORICAL]).toarray()
        return np.hstack([df[numeric_cols].to_numpy(dtype=np.float64), onehot])

    X_train, X_test = encode(train), encode(test)
    n_cat = [len(c) for c in encoder.categories_]
    print(f"Input columns: {len(numeric_cols)} numeric + {n_cat[0]} protocol + "
          f"{n_cat[1]} service + {n_cat[2]} flag = {X_train.shape[1]}")
    assert X_train.shape[1] == 122, f"expected 122 input columns, got {X_train.shape[1]}"
    assert X_test.shape[1] == 122

    # 3. Standardize (fit on train only)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    # 4. Labels -> 24 classes: normal, 22 training attacks (sorted), other
    train_attacks = sorted(set(y_train_raw) - {"normal"})
    print(f"Distinct attack names in training: {len(train_attacks)}")
    assert len(train_attacks) == 22, f"expected 22 training attacks, got {len(train_attacks)}"
    class_names = ["normal"] + train_attacks + ["other"]
    assert len(class_names) == 24
    index = {name: i for i, name in enumerate(class_names)}
    other = index["other"]
    y_train = y_train_raw.map(lambda l: index.get(l, other)).to_numpy(dtype=np.int64)
    y_test = y_test_raw.map(lambda l: index.get(l, other)).to_numpy(dtype=np.int64)
    test_only = sorted(set(y_test_raw) - set(y_train_raw))
    print(f"Classes: {len(class_names)}; test-only attacks mapped to 'other' "
          f"({int((y_test == other).sum())} rows): {test_only}")

    # 5. Sanity check against Table 3
    print_category_counts(y_train_raw, y_test_raw)

    return X_train, y_train, X_test, y_test, class_names, y_train_raw, y_test_raw
