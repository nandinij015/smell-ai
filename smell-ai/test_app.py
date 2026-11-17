"""Checks that everything works without opening a browser.

Run:  python test_app.py
"""
import sys

import joblib
import numpy as np
from streamlit.testing.v1 import AppTest

from sensor import SimulatedNose, build_smell_vectors, load_samples
from smell_model import MODEL_PATH, SmellModel
from train_nose import NOSE_PATH

failures = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        failures.append(name)


# --- smell model ---------------------------------------------------------
model = SmellModel.load()
check("smell model has 138 words", len(model.words) == 138)
top = model.describe(["COc1cc(C=O)ccc1O"], 5)  # vanillin
check("vanillin sounds like vanilla", "vanilla" in top, str(top))
top = model.describe(["CC(C)C1CCC(C)CC1O"], 5)  # menthol
check("menthol sounds like mint", "mint" in top, str(top))
try:
    model.describe(["not a molecule"])
    check("invalid SMILES rejected", False)
except ValueError:
    check("invalid SMILES rejected", True)

# --- sensor + nose -------------------------------------------------------
nose, clf = SimulatedNose(), joblib.load(NOSE_PATH)
smells = build_smell_vectors(load_samples())
check("six samples loaded", len(smells) == 6)
reading = nose.sniff(next(iter(smells.values())), np.random.default_rng(1))
check("sensor gives 8 readings", reading.shape == (8,))
rng = np.random.default_rng(123)  # sniffs the nose was not trained on
hits = sum(clf.predict(nose.sniff(s, rng).reshape(1, -1))[0] == k
           for k, s in smells.items() for _ in range(20))
acc = hits / (20 * len(smells))
check("nose accuracy > 85% on fresh sniffs", acc > 0.85, f"({acc:.0%})")

# --- the app itself, headless -------------------------------------------
at = AppTest.from_file("app.py", default_timeout=60).run()
check("app loads without error", not at.exception, str(at.exception))
at.selectbox[0].select("coffee")
at.button[0].click()
at.run()
check("Sniff runs without error", not at.exception, str(at.exception))
text = " ".join(m.value for m in at.markdown) + " ".join(s.value for s in at.subheader)
check("app shows the nose and smell-model sections",
      "The nose guesses" in text and "The smell model describes it" in text)

print()
if failures:
    print(f"{len(failures)} check(s) failed: {failures}")
    sys.exit(1)
print("All checks passed.")
