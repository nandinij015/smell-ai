"""Shared helpers for the real BME688 nose: serial reading, parsing, features."""
import time

import numpy as np

GAS_COLS = ["g200", "g250", "g300", "g350"]  # gas resistance (ohms) at each heater temperature
CSV_COLS = ["t_s", "label", "session"] + GAS_COLS + ["temp_c", "hum", "pres_hpa"]
CLEAN_AIR = "clean_air"
BAUD = 115200


def parse_line(line):
    """Parse one 'D,ms,g200,g250,g300,g350,tempC,hum,pres' line. None if it isn't a good data line."""
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
    if any(g <= 0 for g in gas):  # a 0 means the gas reading failed
        return None
    return {"ms": ms, "gas": gas, "temp_c": temp_c, "hum": hum, "pres_hpa": pres}


def features(gas, baseline):
    """The nose's input: log of (gas resistance / clean-air resistance), one number per heater temperature.

    Dividing by clean air removes the sensor's personal offset; the log makes the
    large resistance swings comparable. Accepts one row (4,) or many (n, 4).
    """
    return np.log(np.asarray(gas, dtype=float) / np.asarray(baseline, dtype=float))


def find_port():
    """Best guess for the ESP32's serial port, or None."""
    from serial.tools import list_ports
    ports = list(list_ports.comports())
    for p in ports:
        text = f"{p.description} {p.manufacturer or ''}".lower()
        if any(k in text for k in ("cp210", "ch340", "ch910", "silicon labs", "usb serial", "uart")):
            return p.device
    for p in ports:
        if "usb" in p.device.lower() or "acm" in p.device.lower():
            return p.device
    return ports[0].device if ports else None


def open_serial(port=None, baud=BAUD):
    import serial
    port = port or find_port()
    if not port:
        raise SystemExit("No serial port found. Is the ESP32 plugged in with a DATA cable? "
                         "On Windows you may need the CP210x/CH340 driver. Use --port COM3 (or /dev/ttyUSB0).")
    print(f"opening {port} at {baud} baud ...")
    return serial.Serial(port, baud, timeout=3)


def stream(ser, show_info=True):
    """Yield parsed data rows forever, printing the ESP32's '#' info lines."""
    import serial
    while True:
        try:
            raw = ser.readline()
        except serial.SerialException as e:
            print(f"\nSerial connection lost ({e.__class__.__name__}). Was the USB cable unplugged "
                  "or did the ESP32 reset? Replug it and run again; what was recorded is saved.")
            return
        if not raw:
            print("(no data for 3 s - is the ESP32 running the logger sketch?)")
            continue
        line = raw.decode("utf-8", errors="ignore").strip()
        if line.startswith("#"):
            if show_info:
                print(line)
            continue
        row = parse_line(line)
        if row is not None:
            row["wall"] = time.time()
            yield row
