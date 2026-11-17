"""Train the nose on REAL sensor recordings, and test it honestly (README Part 3, step 5).

  python train_real.py

The honest test is leave-one-session-out: train on every session except one, test
on the one it never saw (a different day, a different jar setup). That number is the
result you can quote. The "same-session" number printed for comparison is inflated:
neighbouring readings are near-copies of each other, so a random split just tests
memory. It is printed only to show you the gap.
"""
import argparse
import glob
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from hw import CLEAN_AIR, GAS_COLS

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "nose_real.joblib")
SMOOTH = 5  # the live demo averages this many consecutive readings

parser = argparse.ArgumentParser()
parser.add_argument("--data", default=os.path.join(HERE, "data", "real"))
parser.add_argument("--skip-seconds", type=float, default=120,
                    help="drop the first N seconds of each smell recording (sensor still settling)")
parser.add_argument("--min-rows", type=int, default=30)
args = parser.parse_args()


def load(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not files:
        raise SystemExit(f"No CSV files in {folder}. Record some first with record.py.")
    frames = []
    for i, fp in enumerate(files):
        d = pd.read_csv(fp)
        d["recording"] = os.path.basename(fp)
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def build(df, skip_s):
    """Per session: divide by that session's clean-air level. Returns X, y, session, recording, order."""
    X, y, sess, rec = [], [], [], []
    clean_scores = []
    for s, ds in df.groupby("session"):
        clean = ds[ds["label"] == CLEAN_AIR]
        if len(clean) < 5:
            raise SystemExit(f"Session '{s}' has no clean_air recording. Record one "
                             f"(python record.py --session {s} --label clean_air --minutes 5).")
        base = clean[GAS_COLS].median().to_numpy()
        lc = np.log(clean[GAS_COLS].to_numpy() / base)
        clean_scores.extend(np.abs(lc).max(axis=1))
        for r, dr in ds[ds["label"] != CLEAN_AIR].groupby("recording"):
            dr = dr[dr["t_s"] >= skip_s]
            if len(dr) == 0:
                continue
            X.append(np.log(dr[GAS_COLS].to_numpy() / base))
            y.extend(dr["label"])
            sess.extend([s] * len(dr))
            rec.extend([r] * len(dr))
    if not X:
        raise SystemExit("No smell rows left after skipping the settling time. Record longer, or lower --skip-seconds.")
    clean_thr = max(0.03, 1.5 * float(np.percentile(clean_scores, 99)))
    return np.vstack(X), np.array(y), np.array(sess), np.array(rec), clean_thr


def smooth_predict(clf, X, rec, k=SMOOTH):
    """Average predict_proba over blocks of k consecutive rows in the same recording; return one pred per row."""
    P = clf.predict_proba(X)
    out = np.empty(len(X), dtype=object)
    for r in np.unique(rec):
        idx = np.where(rec == r)[0]
        for a in range(0, len(idx), k):
            blk = idx[a:a + k]
            out[blk] = clf.classes_[P[blk].mean(axis=0).argmax()]
    return out


def make_clf():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=0)


df = load(args.data)
X, y, sess, rec, clean_thr = build(df, args.skip_seconds)
classes = sorted(str(c) for c in set(y))
sessions = sorted(str(x) for x in set(sess))
print(f"{len(X)} readings | smells: {classes} | sessions: {sessions}")
for c in classes:
    print(f"   {c:15s} {np.sum(y == c):5d} readings in {len(set(sess[y == c]))} session(s)")
if len(classes) < 2:
    raise SystemExit("Need at least 2 different smells to train.")

print("\n=== HONEST TEST: train on other sessions, test on the one never seen ===")
if len(sessions) >= 2:
    accs, accs_s, all_true, all_pred = [], [], [], []
    for held in sessions:
        tr, te = sess != held, sess == held
        te_classes = set(y[te])
        clf = make_clf().fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        pred_s = smooth_predict(clf, X[te], rec[te])
        a, a_s = accuracy_score(y[te], pred), accuracy_score(y[te], pred_s)
        accs.append(a)
        accs_s.append(a_s)
        all_true.extend(y[te])
        all_pred.extend(pred)
        print(f"held-out session {held:10s}: {a:6.1%} per reading, {a_s:6.1%} averaged over {SMOOTH} readings"
              f"  ({len(te_classes)} smells in this session)")
    print(f"\nMEAN over held-out sessions: {np.mean(accs):.1%} per reading, {np.mean(accs_s):.1%} smoothed")
    print("confusion matrix, all held-out rows (rows = truth, cols = guess):", classes)
    print(confusion_matrix(all_true, all_pred, labels=classes))
    print(f"(chance level for {len(classes)} smells = {1 / len(classes):.0%})")
else:
    print("WARNING: only ONE session recorded, so there is no honest test yet.")
    print("Record at least one more session on a different day. Meanwhile, a time-ordered split")
    print("(train on the first 70% of each recording, test on the last 30%) as a weak sanity check:")
    tr = np.zeros(len(X), bool)
    for r in np.unique(rec):
        idx = np.where(rec == r)[0]
        cut = int(len(idx) * 0.7)
        tr[idx[:max(0, cut - 5)]] = True  # 5-row gap between train and test
    te = np.zeros(len(X), bool)
    for r in np.unique(rec):
        idx = np.where(rec == r)[0]
        te[idx[int(len(idx) * 0.7):]] = True
    clf = make_clf().fit(X[tr], y[tr])
    print(f"   time-ordered split accuracy: {accuracy_score(y[te], clf.predict(X[te])):.1%}  (still optimistic)")

print("\n=== for comparison only: random split WITHIN the same sessions (inflated, do not quote) ===")
Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
print(f"   {accuracy_score(yb, make_clf().fit(Xa, ya).predict(Xb)):.1%}")

clf = make_clf().fit(X, y)
joblib.dump({"clf": clf, "classes": [str(c) for c in clf.classes_], "gas_cols": GAS_COLS,
             "clean_thr": clean_thr, "skip_seconds": args.skip_seconds,
             "sessions": sessions}, MODEL_PATH)
print(f"\nfinal model (trained on all sessions) saved -> {MODEL_PATH}")

try:  # a picture of each smell's fingerprint: great for explaining the project
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    w = 0.8 / len(classes)
    for i, c in enumerate(classes):
        m = X[y == c].mean(axis=0)
        s = X[y == c].std(axis=0)
        ax.bar(np.arange(4) + i * w, m, w, yerr=s, label=c, capsize=2)
    ax.set_xticks(np.arange(4) + 0.4 - w / 2)
    ax.set_xticklabels([f"{t} °C" for t in (200, 250, 300, 350)])
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("log(gas resistance / clean air)")
    ax.set_title("Fingerprint of each smell at each heater temperature")
    ax.legend()
    fig.tight_layout()
    png = os.path.join(args.data, "fingerprints.png")
    fig.savefig(png, dpi=150)
    print(f"fingerprint chart -> {png}")
except Exception as e:  # plotting is optional
    print("(skipped chart:", e, ")")
