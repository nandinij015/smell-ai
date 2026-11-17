# Build an AI that can smell

A smell model that reads a molecule and predicts what it smells like, joined to an
electronic nose that tells smells apart. Runs on any laptop in about a minute, no GPU,
no hardware.

It uses the dataset from the open-source **OpenPOM** project (4,983 molecules labelled
by perfume experts, 138 smell words). The dataset is bundled in `data/`.

> **The nose is simulated in code.** It shows how the whole pipeline fits together.
> Part 3 explains how to swap in a real gas sensor.

## Part 1: run it

Python 3.10 or newer.

```
cd smell-ai
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python smell_model.py              # trains the smell model (~30 s)
python train_nose.py               # trains the nose (~15 s)
streamlit run app.py               # opens the app in your browser
```

Pick a smell, press **Sniff**, and watch:
the 8 sensor channels react, the nose guess what it is (with its confidence), and the
smell model describe it in words.

To check everything works without a browser: `python test_app.py`

## Part 2: how it works

| File | What it does |
| --- | --- |
| `smell_model.py` | Turns each molecule into a fingerprint of its chemical fragments (Morgan/ECFP), then a small neural net learns which fragments go with which of 138 smell words |
| `samples.json` | Six smells (coffee, vanilla candle, mint, lemon, banana, wood smoke), each a list of real aroma molecules |
| `sensor.py` | A simulated array of 8 gas sensors; each reacts differently to a molecule's size, oil-loving-ness, oxygen and sulfur |
| `train_nose.py` | Sniffs every smell 200 times with random noise and humidity, then trains a random forest to name the smell |
| `app.py` | The Streamlit app |
| `test_app.py` | Headless check of the whole thing |
| `data/` | OpenPOM's curated dataset (MIT licence) |

The smell model reads molecule structures, but a real gas sensor only gives electrical
readings. So the nose names the sample first, and that sample's list of known molecules
tells the smell model what to describe. **The nose never predicts molecules from raw
sensor data**; that link is the hard research problem, and this project does not solve it.

### Results (yours may differ a little)

- Smell model: **~0.84 ROC-AUC** on 997 molecules it never saw (0.5 = guessing, 1 = perfect).
  Split is random; OpenPOM's own 0.89 uses a graph neural network, stratified 5-fold CV and
  an ensemble of 10 models, so the numbers are not directly comparable.
- Nose: **~95%** on new sniffs.

**Read the nose number carefully.** It is tested on the same simulator it learned from,
so it only proves the code works. It also depends on how much noise the simulator adds
(`noise`, `drift` and the mixing variation in `sensor.py`); we lowered them until the
nose worked, so raising them shows how fast accuracy falls. A real sensor is harder,
which is what Part 3 is for. Don't quote this number as a result.

## Part 3: give it a real nose

**Full step-by-step guide: [`HARDWARE.md`](HARDWARE.md).** The code for it is included: `firmware/bme688_logger/` (ESP32),
`record.py`, `train_real.py`, `live_demo.py`. The summary below explains the idea.

A **BME688** gas sensor (around ₹1,400) can step its heater through different
temperatures, so one chip gives several readings per cycle.

1. Wire the BME688 to an ESP32 over I2C: power, ground, SCL, SDA. (Check that your
   breakout board is supported by the Adafruit library before ordering.)
2. Leave it powered for 24 hours before trusting any reading.
3. With the Adafruit BME680 library, loop through a few heater temperatures (for example
   200, 250, 300 and 350 °C). Read gas resistance at each and print to your laptop.
4. Record each smell in its own jar for 30 minutes, with clean air in between. Do at
   least 3 sessions on different days.
5. Use those readings in place of `sensor.py`: one row per heater cycle, divided by the
   clean-air reading. Train the same random forest, and **test it on a session it never saw**.
   That honest number (expect something like 60-85%) is the real result.

Two traps:
- Hot coffee mostly teaches a sensor to detect steam. Use room-temperature grounds.
- Very similar smells are hard to tell apart. Start with very different ones.

## Part 4: go further

Swap in **OpenPOM** itself, a graph neural network (message passing over atoms and bonds,
256-dimensional "principal odor map", 0.89 ROC-AUC in its authors' tests). It needs
Python 3.10, PyTorch 2.4, DGL and DeepChem, and a GPU helps, so use Google Colab. The
pretrained weights are Git LFS files: `git clone` the repo with Git LFS installed, because
a GitHub zip download contains only 133-byte placeholders.

## Credits

- Research: Lee et al., *A principal odor map unifies diverse tasks in olfactory
  perception*, Science (2023).
- Dataset: OpenPOM (MIT licence), built on the GoodScents and Leffingwell odour data.
  See `data/OPENPOM_LICENSE`.
