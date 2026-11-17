"""The app: pick a smell, press Sniff.

  1. the 8 (simulated) sensor channels react
  2. the nose guesses what the smell is, with its confidence
  3. the smell model describes it in words

Run:  streamlit run app.py
"""
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sensor import (CHANNEL_NAMES, HARDWARE_CHANNELS, HardwareNose, SimulatedNose,
                    find_port, load_samples)
from smell_model import MODEL_PATH, SmellModel
from train_nose import NOSE_PATH

st.set_page_config(page_title="AI Nose", page_icon="👃", layout="centered")


@st.cache_resource
def load_everything():
    return SmellModel.load(), joblib.load(NOSE_PATH), SimulatedNose(), load_samples()


st.title("👃 AI Nose")
try:
    import serial  # noqa: F401
except ModuleNotFoundError:
    st.warning("Hardware is not available yet: `pyserial` is not installed. Run `pip install pyserial` or reinstall from requirements.txt.")
    serial_available = False
else:
    serial_available = True

real_port = find_port() if serial_available else None
hardware_available = bool(real_port)

if hardware_available:
    st.success(f"Hardware sensor detected on {real_port}. Using the live ESP32 + BME688 path.")
else:
    st.caption("A simulated 8-sensor electronic nose + a molecule-to-smell model. "
               "The sensor is simulated in code; see README Part 3 for a real one.")

if not (os.path.exists(MODEL_PATH) and os.path.exists(NOSE_PATH)):
    st.error("Models not trained yet. Run `python smell_model.py` and then "
             "`python train_nose.py`, then reload this page.")
    st.stop()

smell_model, nose_clf, nose, samples = load_everything()
keys = list(samples)
labels = {k: samples[k]["label"] for k in keys}

sensor_mode = st.sidebar.radio(
    "Sensor input",
    ["Live hardware" if hardware_available else "Simulation", "Simulation"] if hardware_available else ["Simulation"],
    index=0 if hardware_available else 0,
)

if hardware_available and sensor_mode == "Live hardware":
    try:
        hw = HardwareNose(port=real_port)
    except Exception as exc:
        st.error(f"Could not open the hardware sensor: {exc}")
        st.stop()
    reading = hw.sniff()
    st.subheader("Live sensor readings")
    st.bar_chart(pd.DataFrame({"reading": reading}, index=HARDWARE_CHANNELS))
    st.caption("This is the real BME688 sensor stream from the ESP32; the trained nose model is not available until `nose_real.joblib` is created.")
    if os.path.exists("nose_real.joblib"):
        real_model = joblib.load("nose_real.joblib")
        st.subheader("Real nose guess")
        probs = real_model["clf"].predict_proba(reading.reshape(1, -1))[0]
        order = np.argsort(probs)[::-1]
        label = real_model["classes"][order[0]]
        st.metric("Guess", label, f"{probs[order[0]]:.0%} confident")
    else:
        st.info("Hardware is connected, but no real training model exists yet. Run `python train_real.py` after recording sessions.")
    st.stop()

col1, col2 = st.columns([2, 1])
choice = col1.selectbox("Pick a smell", keys, format_func=lambda k: labels[k])
humidity = col2.slider("Air humidity", 0.0, 1.0, 0.5, 0.05)

with st.expander("What's in this smell?"):
    st.table(pd.DataFrame({"molecule": list(samples[choice]["molecules"]),
                           "SMILES": list(samples[choice]["molecules"].values())}))

if st.button("👃 Sniff", type="primary"):
    smiles = list(samples[choice]["molecules"].values())

    # 1. sensors react
    reading = nose.sniff(smiles, humidity=humidity)
    st.subheader("1. The sensors react")
    st.bar_chart(pd.DataFrame({"reading (1.0 = clean air)": reading}, index=CHANNEL_NAMES))

    # 2. the nose guesses
    proba = nose_clf.predict_proba(reading.reshape(1, -1))[0]
    order = np.argsort(proba)[::-1]
    guess = nose_clf.classes_[order[0]]
    st.subheader("2. The nose guesses")
    st.metric("Nose says", labels[guess], f"{proba[order[0]]:.0%} confident")
    st.progress(float(proba[order[0]]))
    st.dataframe(pd.DataFrame({"smell": [labels[nose_clf.classes_[i]] for i in order],
                               "confidence": [f"{proba[i]:.0%}" for i in order]}),
                 hide_index=True)
    if guess == choice:
        st.success("Correct.")
    else:
        st.warning(f"Wrong - it was {labels[choice]}. Simulated sensors fool the nose sometimes, too.")

    # 3. the smell model describes it
    st.subheader("3. The smell model describes it")
    probs, n_used = smell_model.predict(list(samples[guess]["molecules"].values()))
    st.write("Based on the nose's guess, it smells: **" + ", ".join(probs.index[:5]) + "**")
    st.bar_chart(probs.head(10))
    st.caption(f"Averaged over the {n_used} known molecules of '{labels[guess]}'.")

st.divider()
st.subheader("Or ask the smell model about any molecule")
smi = st.text_input("SMILES string", "CC(C)CCOC(C)=O",
                    help="e.g. CCO (ethanol), COc1cc(C=O)ccc1O (vanillin)")
if smi:
    try:
        p, _ = smell_model.predict([smi])
        st.write("Smells: **" + ", ".join(p.index[:5]) + "**")
    except ValueError:
        st.error("That is not a valid SMILES string.")
