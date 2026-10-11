#!/usr/bin/env python3
# MIT License. Original esp32-ai (c) 2026 Viacheslav Sierbov. Changes (c) 2026 Mark Bartlett / InsiderLLM.
"""Reset an ESP32-S3 over native USB-Serial/JTAG (RTS pulse) and capture serial output.
usage: capture_serial.py PORT SECONDS OUTFILE [STOP_REGEX]
Stops after SECONDS, or as soon as a line matching STOP_REGEX has been received.
Exit status 0 if STOP_REGEX matched (or none given), 3 if it timed out without a match."""
import re, sys, time, serial
port, secs, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
stop = re.compile(sys.argv[4].encode()) if len(sys.argv) > 4 else None
s = serial.Serial(port, 115200, timeout=0.1)
# USB-Serial/JTAG: DTR=0, RTS=1 -> chip reset; release RTS -> normal boot (not download mode)
s.dtr = False; s.rts = True; time.sleep(0.1); s.rts = False
t0 = time.time()
buf = b""
matched = stop is None
with open(out, "wb") as f:
    f.write(f"# capture {port} {secs}s start {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n".encode())
    while time.time() - t0 < secs:
        try:
            b = s.read(4096)
        except serial.SerialException:
            # port re-enumerates after reset; reopen
            time.sleep(0.2)
            try: s.close(); s = serial.Serial(port, 115200, timeout=0.1)
            except Exception: pass
            continue
        if not b:
            continue
        f.write(b); f.flush()
        if stop:
            buf = (buf + b)[-8192:]
            if any(stop.search(line) for line in buf.split(b"\n")[:-1]):
                matched = True
                break
    f.write(f"\n# capture end {time.time() - t0:.1f}s matched={matched}\n".encode())
sys.exit(0 if matched else 3)
