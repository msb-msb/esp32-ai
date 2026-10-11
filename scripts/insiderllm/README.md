<!-- MIT License. Original esp32-ai (c) 2026 Viacheslav Sierbov. Changes (c) 2026 Mark Bartlett / InsiderLLM. -->
# scripts/insiderllm

Host tools used for the InsiderLLM measurements of the TinyStories sketch (see the top of the repo README). They live
only on the `tools` branch, together with the `LOGIT_CHECKSUM` gate in the sketch.

| file | what it does |
|---|---|
| `build_variant.sh` | Compile and flash the firmware only (not the model) with `DEFINES` and FQBN overrides (`CPUFreq=160`), one build directory and `BUILDINFO.txt` per configuration under `build/insiderllm/`. |
| `capture_serial.py` | Reset the board over native USB (RTS pulse) and capture serial for N seconds, or until a line matches a regex. |
| `parse_run.py` | Turn a capture into JSON: fingerprint, ms/token, stage profile, sha256 of the generated text, logit hashes, self-test result. |
| `chat.py` | Chat client and TTFT/decode bench for `INTERACTIVE=1` firmware. |

The correctness gate, as used for every speed change:

```bash
PORT=/dev/ttyACM0 DEFINES="-DUSE_DISPLAY=0 -DLOGIT_CHECKSUM=1" scripts/insiderllm/build_variant.sh
python3 scripts/insiderllm/capture_serial.py /dev/ttyACM0 120 gate.txt '^profile ms/token|FATAL'
python3 scripts/insiderllm/parse_run.py gate.txt    # logit_hash_all 4400a1cbf4a5137d, text_sha256 b9abf55b...
```

`capture_serial.py` and `chat.py` need pyserial (`uv run --with pyserial ...` or `pip install pyserial`); `chat.py`
also needs `tokenizers==0.23.1` and the tokenizer that `scripts/fetch_model.sh tinystories` installs.
