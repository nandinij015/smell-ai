"""Smell AI: a real e-nose (ESP32 + BME688) plus a molecule-to-smell model.

Run:  streamlit run app.py

Tabs
  1. Live nose        - replay a real recording, or sniff live with the sensor plugged in
  2. Honest results   - leave-one-session-out accuracy, confusion matrix, smell fingerprints
  3. Molecule explorer- type any molecule (SMILES), see its structure and predicted smell words
  4. How it works     - pipeline and limitations
"""
import glob
import json
import os
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from hw import CLEAN_AIR, GAS_COLS, features, open_serial, stream

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("SMELL_DATA", os.path.join(HERE, "data", "real"))
NOSE_REAL = os.path.join(HERE, "nose_real.joblib")
HEATERS = [200, 250, 300, 350]
SKIP_S = 120  # settling time dropped from each smell recording (same as train_real.py)

st.set_page_config(page_title="Smell AI", page_icon="👃", layout="wide")


# ---------------------------------------------------------------- loading ----
@st.cache_resource
def load_nose():
    return joblib.load(NOSE_REAL) if os.path.exists(NOSE_REAL) else None


@st.cache_resource
def load_smell_model():
    try:
        from smell_model import SmellModel
        return SmellModel.load()
    except Exception:
        return None


@st.cache_data
def load_samples():
    with open(os.path.join(HERE, "samples.json")) as f:
        return json.load(f)


@st.cache_data
def load_recordings(folder, _stamp):
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    frames = []
    for fp in files:
        d = pd.read_csv(fp)
        d["recording"] = os.path.basename(fp)
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def data_stamp():
    return tuple(sorted((p, os.path.getmtime(p)) for p in glob.glob(os.path.join(DATA, "*.csv"))))


def build_features(df):
    """Per session: divide by that session's clean-air median, log. Mirrors train_real.py."""
    X, y, sess, rec = [], [], [], []
    for s, ds in df.groupby("session"):
        clean = ds[ds["label"] == CLEAN_AIR]
        if len(clean) < 5:
            continue
        base = clean[GAS_COLS].median().to_numpy()
        for r, dr in ds[ds["label"] != CLEAN_AIR].groupby("recording"):
            dr = dr[dr["t_s"] >= SKIP_S]
            if len(dr):
                X.append(np.log(dr[GAS_COLS].to_numpy() / base))
                y.extend(dr["label"])
                sess.extend([s] * len(dr))
                rec.extend([r] * len(dr))
    if not X:
        return None
    return np.vstack(X), np.array(y), np.array(sess), np.array(rec)


@st.cache_data
def honest_eval(_stamp):
    """Leave-one-session-out: train on other sessions, test on the one never seen."""
    df = load_recordings(DATA, _stamp)
    built = build_features(df) if len(df) else None
    if built is None:
        return None
    X, y, sess, rec = built
    classes = sorted(set(y))
    sessions = sorted(set(sess))
    out = {"X": X, "y": y, "classes": classes, "sessions": sessions, "rows": len(X)}
    if len(sessions) >= 2:
        per, true_all, pred_all = [], [], []
        for held in sessions:
            tr, te = sess != held, sess == held
            clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=3, n_jobs=-1,
                                         random_state=0).fit(X[tr], y[tr])
            pred = clf.predict(X[te])
            per.append((held, accuracy_score(y[te], pred)))
            true_all.extend(y[te])
            pred_all.extend(pred)
        out["per_session"] = per
        out["mean_acc"] = float(np.mean([a for _, a in per]))
        out["cm"] = confusion_matrix(true_all, pred_all, labels=classes)
    return out


def pretty(label):
    return label.replace("_", " ").title()


# ------------------------------------------------------------ live engine ----
def clean_threshold(clean_rows, baseline):
    """How far from clean air a reading must be to count as 'a smell': 1.5x the clean recording's own
    99th-percentile wobble, never below 0.15. (The threshold saved in nose_real.joblib is pooled over
    clean recordings that drifted a lot, so it is far too high for live use.)"""
    wobble = np.abs(np.log(np.asarray(clean_rows) / baseline)).max(axis=1)
    return max(0.15, 1.5 * float(np.percentile(wobble, 99)))


def sniff_loop(rows, baseline, thr, nose, smell_model, samples, max_seconds=None, delay=0.0):
    """Feed gas rows through the nose and keep the dashboard updated."""
    clf, classes = nose["clf"], nose["classes"]
    c1, c2 = st.columns([3, 2])
    status = st.empty()
    chart_ph = c1.empty()
    proba_ph = c2.empty()
    words_ph = st.empty()
    history, recent = [], []
    t0 = time.time()
    for g in rows:
        f = features(g, baseline)
        history.append(f)
        chart_ph.line_chart(pd.DataFrame(history[-80:], columns=[f"{t} °C" for t in HEATERS]),
                            height=280)
        if np.abs(f).max() < thr:
            recent.clear()
            status.info("Clean air: nothing detected")
            proba_ph.empty()
            words_ph.empty()
        else:
            recent.append(clf.predict_proba(f.reshape(1, -1))[0])
            recent[:] = recent[-5:]
            p = np.mean(recent, axis=0)
            top = int(np.argmax(p))
            label = classes[top]
            status.success(f"Nose says **{pretty(label)}** ({p[top]:.0%} confident)")
            proba_ph.bar_chart(pd.Series(p, index=[pretty(c) for c in classes]), height=280)
            if smell_model is not None and label in samples:
                words = smell_model.describe(list(samples[label]["molecules"].values()), 5)
                words_ph.write("Smell model describes it as: **" + ", ".join(words) + "**")
        if delay:
            time.sleep(delay)
        if max_seconds and time.time() - t0 > max_seconds:
            break


def tab_live():
    nose = load_nose()
    if nose is None:
        st.error("`nose_real.joblib` not found. Run `python train_real.py` first.")
        return
    smell_model, samples = load_smell_model(), load_samples()
    df = load_recordings(DATA, data_stamp())

    try:
        from hw import find_port
        port = find_port()
    except Exception:
        port = None

    modes = ["Replay a real recording"] + (["Live sensor"] if port else [])
    mode = st.radio("Source", modes, horizontal=True)
    if not port:
        st.caption("No sensor detected, so only replay is available. Replays use real BME688 "
                   "data recorded on the project's ESP32.")

    if mode.startswith("Replay"):
        if df.empty:
            st.error("No recordings in data/real.")
            return
        smells = df[df["label"] != CLEAN_AIR]["recording"].unique().tolist()
        rec = st.selectbox("Recording", smells)
        session = df.loc[df["recording"] == rec, "session"].iloc[0]
        cleans = df[(df["session"] == session) & (df["label"] == CLEAN_AIR)]["recording"].unique().tolist()
        base_rec = st.selectbox("Clean-air baseline recording", cleans,
                                help="Sensor drift is large, so use the clean-air recording taken "
                                     "just before this smell.")
        speed = st.slider("Seconds between readings", 0.0, 0.5, 0.05, 0.05)
        n_rows = st.slider("Readings to replay", 30, 360, 150, 10,
                           help="One reading is about 2.5 s. Weak smells fade after a few minutes.")
        if st.button("▶ Run replay", type="primary"):
            clean = df[df["recording"] == base_rec][GAS_COLS].to_numpy()
            baseline = np.median(clean, axis=0)
            thr = clean_threshold(clean, baseline)
            rows = df[df["recording"] == rec][GAS_COLS].to_numpy()[:n_rows]
            true = df.loc[df["recording"] == rec, "label"].iloc[0]
            st.caption(f"True label of this recording: **{pretty(true)}**. "
                       f"Readings within {thr:.2f} (log scale) of the baseline count as clean air.")
            sniff_loop(rows, baseline, thr, nose, smell_model, samples, delay=speed)
    else:
        st.success(f"Sensor found on {port}")
        st.caption("Close the Arduino Serial Monitor first. Keep the sensor in clean air for the "
                   "baseline, then bring it over a jar.")
        secs = st.slider("Sniff for (seconds)", 30, 300, 120, 10)
        if st.button("● Start live sniff", type="primary"):
            ser = open_serial(port)
            try:
                gen = (r["gas"] for r in stream(ser, show_info=False))
                bar, base_rows, t0 = st.progress(0.0, text="Setting clean-air baseline (60 s) ..."), [], time.time()
                for g in gen:
                    base_rows.append(g)
                    el = time.time() - t0
                    bar.progress(min(el / 60, 1.0), text="Setting clean-air baseline (60 s) ...")
                    if el >= 60:
                        break
                bar.empty()
                baseline = np.median(np.array(base_rows), axis=0)
                thr = clean_threshold(base_rows, baseline)
                st.write("Baseline set. **Bring the sensor to a smell now.**")
                sniff_loop(gen, baseline, thr, nose, smell_model, samples, max_seconds=secs)
            finally:
                ser.close()


def tab_results():
    res = honest_eval(data_stamp())
    if res is None:
        st.warning("No usable recordings yet (each session needs a clean_air recording).")
        return
    n_s, n_c = len(res["sessions"]), len(res["classes"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Smells", n_c)
    c2.metric("Sessions (days)", n_s)
    c3.metric("Labelled readings", res["rows"])

    if n_s >= 2:
        st.metric("Accuracy on a day the model never saw", f"{res['mean_acc']:.0%}",
                  help=f"Mean over held-out sessions. Chance for {n_c} smells = {1 / n_c:.0%}.")
        st.dataframe(pd.DataFrame(res["per_session"], columns=["Held-out session", "Accuracy"])
                     .assign(Accuracy=lambda d: d["Accuracy"].map("{:.0%}".format)), hide_index=True)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        ax.imshow(res["cm"], cmap="Blues")
        ax.set_xticks(range(n_c), [pretty(c) for c in res["classes"]], rotation=30)
        ax.set_yticks(range(n_c), [pretty(c) for c in res["classes"]])
        for i in range(n_c):
            for j in range(n_c):
                ax.text(j, i, res["cm"][i, j], ha="center", va="center")
        ax.set_xlabel("Nose guessed")
        ax.set_ylabel("Truth")
        fig.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("Only one session is recorded, so there is **no honest test yet**: any "
                   "accuracy on one day is inflated, because neighbouring readings are near-copies. "
                   "Record 2 more days (`python record.py --session day2 ...`) and this page "
                   "will show accuracy on a day the model has never seen.")

    st.subheader("Smell fingerprints")
    st.caption("Average log(gas resistance / clean air) at each heater temperature. "
               "Error bars are one standard deviation.")
    X, y = res["X"], res["y"]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    w = 0.8 / n_c
    for i, c in enumerate(res["classes"]):
        ax.bar(np.arange(4) + i * w, X[y == c].mean(axis=0), w, yerr=X[y == c].std(axis=0),
               capsize=2, label=pretty(c))
    ax.set_xticks(np.arange(4) + 0.4 - w / 2, [f"{t} °C" for t in HEATERS])
    ax.axhline(0, color="k", lw=0.5)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)


EXAMPLES = {
    "Limonene (citrus)": "CC(=C)C1CCC(C)=CC1",
    "Menthol (mint)": "CC(C)C1CCC(C)CC1O",
    "Vanillin (vanilla)": "COc1cc(C=O)ccc1O",
    "Isoamyl acetate (banana)": "CC(C)CCOC(C)=O",
    "Linalool (floral)": "CC(C)=CCCC(C)(O)C=C",
    "Ethanol": "CCO",
}


def tab_molecule():
    smell_model = load_smell_model()
    if smell_model is None:
        st.error("`smell_model.joblib` not found. Run `python smell_model.py` first.")
        return
    pick = st.selectbox("Try an example", ["(type your own)"] + list(EXAMPLES))
    smi = st.text_input("SMILES string", EXAMPLES.get(pick, "CC(C)CCOC(C)=O"))
    if not smi:
        return
    try:
        probs, _ = smell_model.predict([smi])
    except ValueError:
        st.error("That is not a valid SMILES string.")
        return
    left, right = st.columns([1, 2])
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        left.image(Draw.MolToImage(Chem.MolFromSmiles(smi), size=(320, 260)))
    except Exception:
        pass
    right.write("Predicted smell: **" + ", ".join(probs.index[:5]) + "**")
    right.bar_chart(probs.head(10))


def tab_how():
    st.markdown("""
**Pipeline**

1. **Sensor.** A BME688 gas sensor on an ESP32. The firmware steps the heater through
   200, 250, 300 and 350 °C and reads gas resistance at each, giving 4 numbers per cycle (about every 2.5 s).
2. **Features.** Each reading is divided by that session's clean-air level and logged, which removes the
   sensor's personal offset.
3. **Nose.** A random forest names the smell. The live view averages 5 readings before guessing.
4. **Smell model.** A separate neural net, trained on 4,983 expert-labelled molecules (OpenPOM dataset),
   describes the guessed smell in words from its known aroma molecules.

**What it does not do**

- It does not identify molecules from the sensor. It matches a 4-number pattern to smells it was
  trained on. The words come from the separate molecule model.
- Accuracy is only meaningful on a day the model has not seen (see *Honest results*).
- Humidity, temperature and sensor drift change readings, so setup must be kept the same.
""")


st.title("👃 Smell AI")
st.caption("A real electronic nose (ESP32 + BME688) joined to a molecule-to-smell model.")
t1, t2, t3, t4 = st.tabs(["Live nose", "Honest results", "Molecule explorer", "How it works"])
with t1:
    tab_live()
with t2:
    tab_results()
with t3:
    tab_molecule()
with t4:
    tab_how()