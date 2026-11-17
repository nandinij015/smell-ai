"""Live demo: the real nose sniffs, guesses the smell, and the smell model describes it.

  python live_demo.py                      # live, with the ESP32 plugged in
  python live_demo.py --replay data/real/day2__coffee.csv --baseline-file data/real/day2__clean_air.csv
                                           # no hardware: replays a recording (backup for interviews)

Live steps: keep the sensor in clean air for the baseline period, then bring it
over a jar. Stop with Ctrl-C.
"""
import argparse
import collections
import json
import os
import time

import joblib
import numpy as np
import pandas as pd

from hw import GAS_COLS, features, open_serial, stream

HERE = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument("--port", default=None)
parser.add_argument("--model", default=os.path.join(HERE, "nose_real.joblib"))
parser.add_argument("--baseline-seconds", type=float, default=60)
parser.add_argument("--smooth", type=int, default=5, help="average this many readings before guessing")
parser.add_argument("--replay", default=None, help="CSV recording to replay instead of using the ESP32")
parser.add_argument("--baseline-file", default=None, help="clean-air CSV for --replay")
parser.add_argument("--speed", type=float, default=0.3, help="seconds between replayed readings")
args = parser.parse_args()

if not os.path.exists(args.model):
    raise SystemExit(f"{args.model} not found. Run `python train_real.py` first.")
M = joblib.load(args.model)
clf, classes = M["clf"], M["classes"]

# optional: describe the guessed smell in words with the molecule-to-smell model
describe = None
try:
    from smell_model import SmellModel
    sm = SmellModel.load()
    with open(os.path.join(HERE, "samples.json")) as f:
        samples = json.load(f)

    def describe(label):
        if label not in samples:
            return None
        return ", ".join(sm.describe(list(samples[label]["molecules"].values()), 5))
except Exception:
    pass  # smell_model.joblib missing -> just skip the words


def rows_live():
    ser = open_serial(args.port)
    yield from (r["gas"] for r in stream(ser))


def rows_replay():
    d = pd.read_csv(args.replay)
    for g in d[GAS_COLS].to_numpy():
        time.sleep(args.speed)
        yield list(g)


if args.replay:
    if not args.baseline_file:
        raise SystemExit("--replay needs --baseline-file (a clean_air CSV from the same session).")
    _clean = pd.read_csv(args.baseline_file)[GAS_COLS].to_numpy()
    baseline = np.median(_clean, axis=0)
    clean_thr = max(0.15, 1.5 * float(np.percentile(np.abs(np.log(_clean / baseline)).max(axis=1), 99)))
    source = rows_replay()
    print("replaying", args.replay)
else:
    source = rows_live()
    print(f"Keep the sensor in CLEAN AIR for {args.baseline_seconds:.0f} s to set the baseline ...")
    base_rows, t0 = [], None
    for g in source:
        t0 = t0 or time.time()
        base_rows.append(g)
        if time.time() - t0 >= args.baseline_seconds:
            break
    baseline = np.median(np.array(base_rows), axis=0)
    # threshold from THIS baseline's own wobble (the one saved in the model is pooled over drifting recordings)
    clean_thr = max(0.15, 1.5 * float(np.percentile(np.abs(np.log(np.array(base_rows) / baseline)).max(axis=1), 99)))
    print("baseline set. Bring the sensor to a smell now.\n")

recent = collections.deque(maxlen=args.smooth)
last_label = None
try:
    for g in source:
        f = features(g, baseline)
        if np.abs(f).max() < clean_thr:
            recent.clear()
            print("clean air                          ", end="\r")
            last_label = None
            continue
        recent.append(clf.predict_proba(f.reshape(1, -1))[0])
        p = np.mean(recent, axis=0)
        order = np.argsort(p)[::-1]
        label = classes[order[0]]
        print(f"Nose says: {label.upper():15s} {p[order[0]]:4.0%}   "
              f"(2nd: {classes[order[1]]} {p[order[1]]:.0%})" if len(classes) > 1 else f"Nose says: {label}",
              " " * 10, end="\r")
        if len(recent) == args.smooth and label != last_label:
            last_label = label
            words = describe(label) if describe else None
            print()
            print(f"  -> {label}: smells {words}" if words else f"  -> {label}")
except KeyboardInterrupt:
    print("\nbye")