"""Smell model: molecule structure -> smell words.

Turns each molecule into a fingerprint of its chemical fragments (Morgan /
ECFP, radius 2) and learns which fragments go with which of the 138 smell
words, using the OpenPOM curated dataset (4,983 molecules labelled by
perfume experts, merged GoodScents + Leffingwell).

Run:   python smell_model.py
Use:   from smell_model import SmellModel; SmellModel.load().describe([...])
"""
import os
import sys
import time
import urllib.request
import warnings

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "data", "curated_GS_LF_merged_4983.csv")
MODEL_PATH = os.path.join(HERE, "smell_model.joblib")
DATA_URL = (
    "https://raw.githubusercontent.com/BioMachineLearning/openpom/main/"
    "openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv"
)

N_BITS = 2048
RADIUS = 2
_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)


def canonical(smiles):
    """RDKit mol from SMILES (stereo ignored, like the dataset). None if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.RemoveStereochemistry(mol)
    return mol


def fingerprint(mol):
    """2048-bit fingerprint of a molecule's chemical fragments."""
    return np.array(_GEN.GetFingerprintAsNumPy(mol), dtype=np.uint8)


def ensure_dataset():
    """Use the bundled OpenPOM CSV; download it if it is missing."""
    if not os.path.exists(DATA_PATH):
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        print("Downloading dataset from OpenPOM ...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    return DATA_PATH


def load_data():
    df = pd.read_csv(ensure_dataset())
    words = [c for c in df.columns if c not in ("nonStereoSMILES", "descriptors")]
    X, Y, keep = [], [], []
    for i, smi in enumerate(df["nonStereoSMILES"]):
        mol = canonical(smi)
        if mol is None:
            continue
        X.append(fingerprint(mol))
        keep.append(i)
    Y = df.loc[keep, words].to_numpy(dtype=np.uint8)
    return np.array(X), Y, words


def _proba(model, X):
    """Smell-word probabilities -> (n_samples, n_words)."""
    return model.predict_proba(X.astype(np.float32))


class SmellModel:
    def __init__(self, net, words):
        self.net = net
        self.words = list(words)

    # ---- training -------------------------------------------------------
    @classmethod
    def train(cls, test_size=0.2, seed=0):
        X, Y, words = load_data()
        print(f"{len(X)} molecules, {len(words)} smell words")
        Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=test_size, random_state=seed)

        t0 = time.time()
        # one small neural net (2048 -> 512 -> 138); sigmoid outputs = multi-label
        net = MLPClassifier(hidden_layer_sizes=(512,), alpha=0.1, max_iter=60,
                            random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            net.fit(Xtr.astype(np.float32), Ytr)
        print(f"trained on {len(Xtr)} molecules in {time.time() - t0:.0f}s")

        P = _proba(net, Xte)
        aucs = [roc_auc_score(Yte[:, j], P[:, j])
                for j in range(len(words)) if 0 < Yte[:, j].sum() < len(Yte)]
        auc = float(np.mean(aucs))
        print(f"ROC-AUC on {len(Xte)} molecules it never saw: {auc:.3f} "
              f"(0.5 = guessing, 1 = perfect; averaged over {len(aucs)} smell words)")
        return cls(net, words), auc

    def save(self, path=MODEL_PATH):
        joblib.dump({"net": self.net, "words": self.words}, path, compress=3)

    @classmethod
    def load(cls, path=MODEL_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found - run `python smell_model.py` first.")
        d = joblib.load(path)
        return cls(d["net"], d["words"])

    # ---- prediction -----------------------------------------------------
    def predict(self, smiles_list):
        """Average smell-word probabilities over valid molecules. Returns (words->prob Series, n_used)."""
        fps = []
        for smi in smiles_list:
            mol = canonical(smi)
            if mol is not None:
                fps.append(fingerprint(mol))
        if not fps:
            raise ValueError("No valid SMILES given")
        probs = _proba(self.net, np.array(fps)).mean(axis=0)
        return pd.Series(probs, index=self.words).sort_values(ascending=False), len(fps)

    def describe(self, smiles_list, top_n=5):
        """Top smell words for a molecule or a mixture of molecules."""
        probs, _ = self.predict(smiles_list)
        return list(probs.index[:top_n])


if __name__ == "__main__":
    model, auc = SmellModel.train()
    model.save()
    print(f"saved -> {MODEL_PATH}")
    print("quick check:")
    for name, smi in [("ethanol", "CCO"), ("vanillin", "COc1cc(C=O)ccc1O"),
                      ("menthol", "CC(C)C1CCC(C)CC1O"), ("limonene", "CC(=C)C1CCC(C)=CC1")]:
        print(f"  {name:9s} -> {', '.join(model.describe([smi]))}")
    sys.exit(0)
