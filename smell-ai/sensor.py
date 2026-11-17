"""Simulated electronic nose: an array of 8 gas sensors.

Each sensor channel reacts differently to four properties of a molecule:
  - size               (molecular weight)
  - oil-loving-ness    (logP, how much it prefers fat to water)
  - oxygen content     (number of oxygen atoms)
  - sulfur content     (number of sulfur atoms)

A sniff of a smell is the mix of its molecules' responses, plus the things that
make real sensors annoying: random noise, humidity (which pushes every channel
a bit), slow drift, and the fact that you never catch every molecule evenly.

THIS IS A SIMULATION. It shows how the pipeline fits together; it says nothing
about how well a real sensor would do (see README Part 3).
"""
import json
import os
import time

import numpy as np
from rdkit.Chem import Crippen, Descriptors

HARDWARE_CHANNELS = ["g200", "g250", "g300", "g350"]


def parse_line(line):
    """Parse one 'D,ms,g200,g250,g300,g350,tempC,hum,pres' line."""
    line = line.strip()
    if not line.startswith("D,"):
        return None
    parts = line.split(",")
    if len(parts) != 9:
        return None
    try:
        ms = int(parts[1])
        gas = [float(x) for x in parts[2:6]]
        temp_c, hum, pres = (float(x) for x in parts[6:9])
    except ValueError:
        return None
    if any(g <= 0 for g in gas):
        return None
    return {"ms": ms, "gas": gas, "temp_c": temp_c, "hum": hum, "pres_hpa": pres}


def find_port():
    """Best guess for the ESP32 serial port, or None."""
    try:
        import serial.tools.list_ports as list_ports
    except ImportError:
        return None
    ports = list(list_ports.comports())
    for p in ports:
        text = f"{p.description} {p.manufacturer or ''}".lower()
        if any(k in text for k in ("cp210", "ch340", "ch910", "silicon labs", "usb serial", "uart")):
            return p.device
    for p in ports:
        if "usb" in p.device.lower() or "acm" in p.device.lower():
            return p.device
    return ports[0].device if ports else None


def open_serial(port=None, baud=115200):
    import serial
    port = port or find_port()
    if not port:
        raise RuntimeError("No serial port found. Plug in the ESP32 and make sure the USB cable carries data.")
    return serial.Serial(port, baud, timeout=3)


class HardwareNose:
    def __init__(self, port=None, baud=115200):
        self.port = port or find_port()
        self.baud = baud
        self.serial = None
        self.baseline = None
        self.connected = False
        self._connect()

    def _connect(self):
        if not self.port:
            return
        try:
            import serial
            self.serial = serial.Serial(self.port, self.baud, timeout=3)
            self.connected = self.serial is not None and getattr(self.serial, "is_open", True)
        except Exception:
            self.serial = None
            self.connected = False

    def read_once(self):
        if self.serial is None or not getattr(self.serial, "is_open", True):
            raise RuntimeError("Hardware sensor is not connected. Plug in the ESP32 and USB cable, then restart the app.")
        while True:
            raw = self.serial.readline()
            if not raw:
                continue
            line = raw.decode("utf-8", errors="ignore").strip()
            if line.startswith("#"):
                continue
            row = parse_line(line)
            if row is not None:
                reading = np.asarray(row["gas"], dtype=float)
                return reading

    def collect_baseline(self, seconds=60):
        if self.serial is None or not getattr(self.serial, "is_open", True):
            raise RuntimeError("Hardware sensor is not connected. Plug in the ESP32 and USB cable, then restart the app.")
        readings = []
        t0 = time.time()
        while time.time() - t0 < seconds:
            r = self.read_once()
            readings.append(r)
        self.baseline = np.median(np.asarray(readings), axis=0)
        return self.baseline.copy()

    def sniff(self, baseline=None):
        reading = self.read_once()
        base = np.asarray(baseline if baseline is not None else self.baseline, dtype=float)
        if base is None or np.all(base == 0):
            base = reading.copy()
        base = np.maximum(base, 1e-9)
        return np.log(np.maximum(reading, 1e-9) / base)

from smell_model import canonical

HERE = os.path.dirname(os.path.abspath(__file__))
N_CHANNELS = 8
CHANNEL_NAMES = [f"S{i + 1}" for i in range(N_CHANNELS)]


def load_samples(path=os.path.join(HERE, "samples.json")):
    with open(path) as f:
        return json.load(f)


def molecule_properties(smiles):
    """[size, oil-loving-ness, oxygen, sulfur], each roughly scaled to ~0-1."""
    mol = canonical(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    n_o = sum(a.GetSymbol() == "O" for a in mol.GetAtoms())
    n_s = sum(a.GetSymbol() == "S" for a in mol.GetAtoms())
    return np.array([
        Descriptors.MolWt(mol) / 200.0,
        Crippen.MolLogP(mol) / 3.0,
        n_o / 2.0,
        float(n_s),
    ])


class SimulatedNose:
    def __init__(self, seed=3):
        rng = np.random.default_rng(seed)  # fixed seed = the same "chip" every time
        self.weights = rng.normal(0, 1.5, size=(N_CHANNELS, 4))
        self.bias = rng.normal(0, 0.5, size=N_CHANNELS)
        self.humidity_gain = rng.uniform(0.02, 0.10, size=N_CHANNELS)

    def response(self, smiles):
        """Noise-free response of every channel to one molecule (0..1)."""
        z = self.weights @ molecule_properties(smiles) + self.bias
        return 1.0 / (1.0 + np.exp(-z))

    def sniff(self, smiles_list, rng=None, noise=0.01, humidity=None, drift=0.005):
        """One sniff of a smell made of several molecules -> 8 channel readings.

        Readings are relative to clean air (1.0 = no change), like the real-sensor
        recipe in the README.
        """
        rng = rng or np.random.default_rng()
        # you rarely catch every molecule, and never in exactly equal amounts
        n = len(smiles_list)
        keep = rng.random(n) < 0.95
        if not keep.any():
            keep[rng.integers(n)] = True
        amounts = rng.dirichlet(np.full(keep.sum(), 80.0))  # mild variation in mix
        responses = np.array([self.response(s) for s, k in zip(smiles_list, keep) if k])
        signal = amounts @ responses

        if humidity is None:
            humidity = rng.uniform(0.2, 0.8)
        reading = (1.0 + signal
                   + self.humidity_gain * humidity
                   + rng.normal(0, drift)                 # slow drift shared by all channels
                   + rng.normal(0, noise, N_CHANNELS))    # per-channel electrical noise
        return reading


def build_smell_vectors(samples):
    """{smell key: list of SMILES}."""
    return {k: list(v["molecules"].values()) for k, v in samples.items()}


if __name__ == "__main__":
    nose = SimulatedNose()
    samples = load_samples()
    rng = np.random.default_rng(0)
    for key, info in samples.items():
        r = nose.sniff(list(info["molecules"].values()), rng)
        print(f"{info['label']:15s}", np.round(r, 2))
