"""Record one jar session from the ESP32 to a CSV file (README Part 3, step 4).

Examples:
  python record.py --session day1 --label clean_air --minutes 5
  python record.py --session day1 --label coffee    --minutes 30
  python record.py --session day1 --label clean_air --minutes 5      # clean air again afterwards

Rules that make the data honest:
  * Use ONE --session name per day/setup (day1, day2, day3 ...). Never mix days in one session.
  * Record clean_air in every session: the nose divides by it.
  * Labels for smells should be: coffee, mint, lemon, banana, vanilla_candle, wood_smoke
    (so the demo can describe them in words).
  * Let the sensor breathe clean air between smells until readings return to normal.
Stop early any time with Ctrl-C; what you recorded so far is kept.
"""
import argparse
import csv
import os
import time

from hw import CSV_COLS, open_serial, stream

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--label", required=True, help="coffee / mint / lemon / banana / ... or clean_air")
parser.add_argument("--session", required=True, help="one name per day, e.g. day1")
parser.add_argument("--minutes", type=float, default=30)
parser.add_argument("--port", default=None, help="e.g. COM3 or /dev/ttyUSB0 (auto-detected if omitted)")
parser.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "real"))
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)
path = os.path.join(args.out, f"{args.session}__{args.label}.csv")
if os.path.exists(path):
    n = 2
    while os.path.exists(path.replace(".csv", f"_{n}.csv")):
        n += 1
    path = path.replace(".csv", f"_{n}.csv")  # never overwrite a recording
    print(f"(file exists - saving as {os.path.basename(path)})")

ser = open_serial(args.port)
print(f"recording '{args.label}' for session '{args.session}' for {args.minutes:g} min -> {path}")
print("Ctrl-C to stop early.\n")

t_first = None
n_rows = 0
deadline = time.time() + args.minutes * 60
try:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLS)
        for row in stream(ser):
            if t_first is None:
                t_first = row["ms"]
            t_s = (row["ms"] - t_first) / 1000.0
            w.writerow([f"{t_s:.2f}", args.label, args.session,
                        *[f"{g:.0f}" for g in row["gas"]],
                        f"{row['temp_c']:.2f}", f"{row['hum']:.2f}", f"{row['pres_hpa']:.2f}"])
            f.flush()
            n_rows += 1
            if n_rows % 10 == 1:
                g = row["gas"]
                print(f"{t_s:7.0f}s  gas(kOhm) 200C={g[0]/1000:7.1f} 250C={g[1]/1000:7.1f} "
                      f"300C={g[2]/1000:7.1f} 350C={g[3]/1000:7.1f}  T={row['temp_c']:.1f}C RH={row['hum']:.0f}%")
            if time.time() >= deadline:
                break
except KeyboardInterrupt:
    print("\nstopped early")
print(f"saved {n_rows} rows -> {path}")
