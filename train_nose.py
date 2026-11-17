"""Train the nose: sniff every smell 200 times (random noise + humidity), then
train a random forest to name the smell from the 8 sensor readings.

Run:  python train_nose.py
"""
import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from sensor import SimulatedNose, build_smell_vectors, load_samples

HERE = os.path.dirname(os.path.abspath(__file__))
NOSE_PATH = os.path.join(HERE, "nose_model.joblib")
SNIFFS_PER_SMELL = 200


def make_dataset(seed=0):
    nose = SimulatedNose()
    smells = build_smell_vectors(load_samples())
    rng = np.random.default_rng(seed)
    X, y = [], []
    for key, smiles in smells.items():
        for _ in range(SNIFFS_PER_SMELL):
            X.append(nose.sniff(smiles, rng))
            y.append(key)
    return np.array(X), np.array(y)


if __name__ == "__main__":
    X, y = make_dataset()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0).fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = accuracy_score(yte, pred)
    print(f"{len(X)} sniffs ({SNIFFS_PER_SMELL} per smell), tested on {len(Xte)} new sniffs")
    print(f"nose accuracy: {acc:.1%}")
    labels = [str(c) for c in clf.classes_]
    print("confusion matrix (rows = true, cols = guess):", labels)
    print(confusion_matrix(yte, pred, labels=labels))
    print("\nNOTE: this is tested on the same simulator it learned from, so the number\n"
          "only shows the code works. A real sensor is harder (README Part 3).")
    joblib.dump(clf, NOSE_PATH)
    print(f"saved -> {NOSE_PATH}")
