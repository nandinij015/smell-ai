# Real nose: step-by-step (ESP32 + BME688)

Parts: ESP32 38-pin board, BME688 nano breakout, 4 female-to-female jumper wires,
micro-USB **data** cable. Everything else (jars, samples) is from home.

## 1. Check the sensor first
Look at the BME688 board. It has 4 holes/pins labelled **3V3, GND, SDA, SCL**.
- Pins soldered on (metal pins sticking out)? Good, go to step 2.
- Loose header strip in the bag? It must be **soldered** on (any electronics repair
  shop can do 4 pins). Do not just push wires into the holes.

## 2. Wire it (USB cable UNPLUGGED while wiring)

| BME688 | ESP32 pin label |
| --- | --- |
| 3V3 | 3V3 (some boards print it as 3.3V) |
| GND | GND |
| SDA | G21 (GPIO21) |
| SCL | G22 (GPIO22) |

Never connect the sensor to 5V / VIN. Use any colour wires, just keep the 4 straight.

## 3. Put the program on the ESP32
1. Install **Arduino IDE 2** (arduino.cc/en/software).
2. File > Preferences > "Additional boards manager URLs", add:
   `https://espressif.github.io/arduino-esp32/package_esp32_index.json`
3. Tools > Board > Boards Manager, search **esp32**, install "esp32 by Espressif Systems".
4. Sketch > Include Library > Manage Libraries: install **Adafruit BME680 Library**
   (click "Install all" for its dependencies, which include Adafruit Unified Sensor).
5. Open `firmware/bme688_logger/bme688_logger.ino`.
6. Tools > Board: **ESP32 Dev Module**. Plug in the ESP32. Tools > Port: pick the new port.
   - No port appears? Try another USB cable (it must carry data) and install the
     CP210x or CH340 driver (Windows/Mac; Linux usually needs nothing).
7. Click Upload. If it says "Connecting......", hold the **BOOT** button on the ESP32
   until upload starts.
8. Tools > Serial Monitor, **115200 baud**. You should see:
   - `# I2C scan:` then `#   device at 0x76` (or 0x77)
   - `# BME688 found.`
   - lines starting with `D,` about every 2 seconds.

| Problem | Fix |
| --- | --- |
| "NOTHING found" in the I2C scan | SDA/SCL swapped, a wire loose, or the pins aren't soldered |
| "BME688 not found" but the scan found a device | Library may not accept your board: tell me the address and the exact message |
| Gibberish text | Serial Monitor baud must be 115200 |
| Upload fails | Different cable, hold BOOT while uploading |

If you see `D,` lines, the hardware works. **Close the Serial Monitor** before using the Python
scripts below: only one program can use the port at a time.

## 4. Burn-in (README step 2)
Leave the ESP32 plugged in and running for **24 hours** before recording anything you
will use. Readings drift a lot in the first day. (Use a laptop, or any USB phone charger:
the sketch runs without a computer.) Keep it in a room, away from cooking, perfume and
cleaning sprays.

## 5. Install the laptop scripts
```
cd smell-ai
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python smell_model.py            # only needed once (smell model, ~30 s)
```

## 6. Record sessions
One **session = one day**. Use `day1`, `day2`, `day3`. The model is tested by training on
some days and testing on a day it has never seen, so do at least 2 days (3 is better).

Smells (very different from each other): `coffee`, `mint`, `lemon`. Labels must be spelled
exactly like this so the demo can describe them in words. Others the app knows:
`banana`, `vanilla_candle`, `wood_smoke`.

Each day, in this order (about 1 hour):
```
python record.py --session day1 --label clean_air --minutes 5
python record.py --session day1 --label coffee    --minutes 15
python record.py --session day1 --label clean_air --minutes 5
python record.py --session day1 --label mint      --minutes 15
python record.py --session day1 --label clean_air --minutes 5
python record.py --session day1 --label lemon     --minutes 15
```
(The README says 30 minutes per smell. 15 is enough for a demo: the first 2 minutes are
dropped while the sensor settles. More days help far more than more minutes.)

How to expose the sensor, **the same way every time**:
- Put a small amount of the sample in a jar: room-temperature ground coffee, a few mint
  leaves, fresh lemon peel. **Never hot** (hot coffee mostly teaches it to detect steam).
- Hold or tape the sensor board 1-2 cm above the sample with the lid loosely over it.
  The sensor must never touch liquid or the sample.
- Between smells, ventilate the room and let the readings return to the clean-air level
  before starting the next recording.
- Don't change the room (fan, window, cooking, perfume) in the middle of a session.

## 7. Train and test honestly
```
python train_real.py
```
Read the line `MEAN over held-out sessions`. **That** is your real result. It will be lower
than the "same session" number, which is inflated and printed only for comparison. Expect
something like 60-85%. Chance level for 3 smells is 33%.
It also saves `data/real/fingerprints.png`: the signature of each smell at each heater
temperature. Great for explaining the project.

## 8. The demo
```
python live_demo.py
```
Keep the sensor in clean air for 60 s (it sets the baseline), then bring it over a jar and
wait 1-2 minutes. It prints the nose's guess and the smell model's description in words.

**Backup for the interview:** record a screen video of a successful run. If you have no
hardware at hand, replay a recording:
```
python live_demo.py --replay data/real/day3__coffee.csv --baseline-file data/real/day3__clean_air.csv
```
(Say clearly that it is a replay.)

## What to say, and not say
- Say: "real BME688 data, 3 smells, trained on days 1-2, tested on day 3: X%."
- Don't say it identifies molecules. It matches patterns of 4 heater-temperature readings
  to smells it was trained on, and the words come from the separate smell model.
- Don't quote the simulated nose's 95%. Don't quote the "same session" number.
