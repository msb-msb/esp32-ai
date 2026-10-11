#!/usr/bin/env python3
# MIT License. Original esp32-ai (c) 2026 Viacheslav Sierbov. Changes (c) 2026 Mark Bartlett / InsiderLLM.
"""Parse esp32_tinystories serial captures.
usage: parse_run.py LOG [LOG ...]       -> one JSON object per log on stdout
Fields: file, fp, throughput_tok_s, ms_token, profile{input,attn,ffn,ple,head},
non_head, text_sha256, text_len, logit_hash_all, logit_hashes[], simd_selftest, head_path."""
import hashlib, json, re, sys

PROFILE = re.compile(rb"^profile ms/token: input ([\d.]+) \| attn ([\d.]+) \| ffn ([\d.]+) \| ple ([\d.]+) \| head ([\d.]+)", re.M)
THR = re.compile(rb"^throughput: ([\d.]+) tok/s\s+\(([\d.]+) ms/token\)", re.M)

def last(pat, t):
    m = list((pat if hasattr(pat, "finditer") else re.compile(pat, re.M)).finditer(t))
    return m[-1] if m else None

def parse(path):
    t = open(path, "rb").read()
    # Use only the last boot in the file: a capture can start with the tail of the
    # boot that followed the upload, before the reset.
    boots = [m.start() for m in re.finditer(rb"=== ESP32-S3 PLE TinyLM ===", t)]
    if boots:
        t = t[boots[-1]:]
    r = {"file": path}
    m = last(rb"build: bytes=\d+ fp=([0-9a-f]+)", t)
    r["fp"] = m.group(1).decode() if m else None
    m = last(THR, t)
    r["throughput_tok_s"], r["ms_token"] = (float(m.group(1)), float(m.group(2))) if m else (None, None)
    m = last(PROFILE, t)
    if m:
        p = dict(zip(["input", "attn", "ffn", "ple", "head"], map(float, m.groups())))
        r["profile"] = p
        r["non_head"] = round(p["input"] + p["attn"] + p["ffn"] + p["ple"], 1)
    a, b = t.find(b">>> "), t.find(b"\n\n--- ")
    text = t[a:b] if a >= 0 and b > a else b""
    r["text_sha256"] = hashlib.sha256(text).hexdigest()
    r["text_len"] = len(text)
    r["logit_hashes"] = [h.decode() for h in re.findall(rb"^logit_hash\[\d+\] ([0-9a-f]{16})", t, re.M)]
    m = last(rb"^logit_hash_all ([0-9a-f]{16})", t)
    r["logit_hash_all"] = m.group(1).decode() if m else None
    m = last(rb"^simd head selftest: (.*)$", t)
    r["simd_selftest"] = m.group(1).decode().strip() if m else None
    r["head_path"] = ("pie-int4" if b"head   -> PIE int4" in t else
                      "pie" if b"head   -> PIE int8" in t else "scalar")
    r["fatal"] = [l.decode(errors="replace") for l in re.findall(rb"^FATAL.*$", t, re.M)]
    return r

for f in sys.argv[1:]:
    print(json.dumps(parse(f)))
