## InsiderLLM fork

This is a fork of [slvDev/esp32-ai](https://github.com/slvDev/esp32-ai), and upstream is the canonical project. The
model, the firmware and the method are slvDev's; everything below the line is upstream's README, unchanged.

This fork adds speed patches for the TinyStories sketch on an ESP32-S3 N16R8:
- The output changes in no build. The logits are bit-identical (FNV-1a over the first 20 tokens' raw logits:
  `4400a1cbf4a5137d`), and the 200-token text is byte-identical (sha256 `b9abf55b…c467`).
- There are also an interactive serial mode and an optional stop at end of story.
- These changes are offered upstream in [slvDev/esp32-ai#26](https://github.com/slvDev/esp32-ai/issues/26). Nothing
  here is endorsed by upstream.

### Branches

Each branch is one change, stacked on the one before it. Every change sits behind a compile-time flag. The defaults
reproduce upstream behaviour except where the change itself is the point.

| branch | flag (default) | change |
|---|---|---|
| `no-display-flag` | `USE_DISPLAY` (1) | Display as a build flag. `scripts/deploy.sh` passes `DEFINES` to the compiler. |
| `eot-stop` | `STOP_AT_EOT` (0) | `generate_vocab.py` writes `VOCAB_EOT_ID`. With 1, generation stops at `<|endoftext|>`. |
| `simd-head` | `HEAD_SIMD` (1) | Output head on the S3 PIE int8 MAC (`head_dot96_s8.S`). |
| `int4-head` | `HEAD_INT4` (1) | Head streamed as packed int4, unpacked exactly in PIE. |
| `simd-core` | `CORE_SIMD` (1) | All core int8 matvecs on PIE (dot96 / dot128 / zero-padded dot80). |
| `logits-sram` | `LOGITS_SRAM` (1) | Logits buffer in internal SRAM instead of PSRAM. |
| `int4-core` | `CORE_INT4` (1) | Core weights streamed as packed int4 (dot96_s4 / dot128_s4, down padded to 96). |
| `interactive` | `INTERACTIVE` (0) | Line protocol on native USB serial, for a host chat client. |
| `prefill-skip` | `PREFILL_SKIP` (1) | The head is skipped on prompt positions before the last. |
| **`insiderllm`** | | All of the above, plus this section. The default branch. |
| `tools` | `LOGIT_CHECKSUM` (0) | `insiderllm` plus the logit-hash gate and the host scripts in `scripts/insiderllm/`. |

All the PIE kernels are checked at boot against the scalar arithmetic, on every row of every tensor. A failed
self-test stops the boot.

### Build progression

ESP32-S3 N16R8 (rev v0.2, 16 MB QIO flash and 8 MB octal PSRAM, both at 80 MHz), CPU at 240 MHz, Arduino-ESP32 core
3.3.10, `-O3`, release `model.bin`, no display attached. Each row is 3 resets × 200 greedy tokens from the compiled
prompt "Once upon a time". All three runs printed identical values.

| build | head ms | non-head ms | ms/token | tok/s (end to end) | logit hash |
|---|---:|---:|---:|---:|---|
| upstream with `USE_DISPLAY=0` (scalar head) | 59.1 | 35.5 | 94.7 | 10.34 | `4400a1cbf4a5137d` |
| + `simd-head` | 43.6 | 35.5 | 79.1 | 12.31 | `4400a1cbf4a5137d` |
| + `int4-head` | 24.1 | 35.5 | 59.6 | 16.21 | `4400a1cbf4a5137d` |
| + `simd-core` | 24.1 | 26.4 | 50.4 | 19.05 | `4400a1cbf4a5137d` |
| + `logits-sram` | 21.2 | 26.3 | 47.5 | 20.62 | `4400a1cbf4a5137d` |
| **+ `int4-core`** | **21.2** | **22.0** | **43.3** | **22.59** | **`4400a1cbf4a5137d`** |
| `insiderllm` branch, fresh clone, built with the commands below | 21.2 | 22.0 | 43.2 | 22.61 | `4400a1cbf4a5137d` |

- **Where the rows come from:** the first six were measured as patches on a copy of upstream (the bench
  records). The last row is this branch built from a fresh clone, with its profile and output identical to the
  `int4-core` row. The 0.1 ms difference was not investigated.
- **Upstream's figure:** 94.9 ms/token of compute and 9.88 tok/s, labelled "attached serial" in
  `firmware/esp32_tinystories/README.md` ("9.88 tok/s attached" in RESULTS.md).
- **The other three branches** don't change these numbers: `eot-stop` and `interactive` are off by default, and
  `prefill-skip` only touches prompt positions, which are outside the timed loop.
- **Other clocks:** at 160 and 80 MHz, the `int4-core` build measured 49.5 and 69.5 ms/token.
- **What remains:** the head and the core now sit at roughly 62 MB/s of PSRAM traffic, so what's left is memory
  traffic, not compute.

### Build and flash (Linux)

You need the following:
- `arduino-cli` with the ESP32 core 3.3.10;
- [`uv`](https://docs.astral.sh/uv/), which `deploy.sh` uses for its header tools;
- `python3`, `cc`;
- the board on its native USB port (`/dev/ttyACM0`).

```bash
arduino-cli config add board_manager.additional_urls https://espressif.github.io/arduino-esp32/package_esp32_index.json
arduino-cli core update-index
arduino-cli core install esp32:esp32@3.3.10

git clone -b insiderllm https://github.com/msb-msb/esp32-ai.git
cd esp32-ai
scripts/fetch_model.sh tinystories
PORT=/dev/ttyACM0 DEFINES="-DUSE_DISPLAY=0" scripts/deploy.sh tinystories
arduino-cli monitor -p /dev/ttyACM0 --config baudrate=115200
```

- **What `deploy.sh` does:** it regenerates `generated/vocab.h` (now with `VOCAB_EOT_ID`), runs upstream's host
  gates, compiles, and flashes both the model and the firmware.
- **The FQBN** is upstream's, unchanged:
  `esp32:esp32:esp32s3:UploadSpeed=921600,USBMode=hwcdc,CDCOnBoot=cdc,UploadMode=default,CPUFreq=240,FlashMode=qio,FlashSize=16M,PartitionScheme=custom,PSRAM=opi,DebugLevel=info`.
- **Defines:** drop `-DUSE_DISPLAY=0` if the SH1106 OLED is wired up. Without a panel, the display code's I2C errors
  cut end-to-end speed from 10.34 to 5.43 tok/s.
- **Defaults:** every other flag in the table defaults to the measured configuration. Override any of them in
  `DEFINES`, for example `DEFINES="-DUSE_DISPLAY=0 -DCORE_INT4=0"`.

After the self-test lines (`simd core int4 selftest: … ok`, `simd head selftest: 25353/25353 rows ok`), the end of
the boot log should read:

```
--- 200 tokens in 8.85 s ---
throughput: 22.61 tok/s   (43.2 ms/token)
profile ms/token: input 1.0 | attn 16.1 | ffn 2.4 | ple 2.5 | head 21.2
```

**Checking the logits yourself.** On the `tools` branch, use
`DEFINES="-DUSE_DISPLAY=0 -DLOGIT_CHECKSUM=1"`. The run then prints per-token hashes and
`logit_hash_all 4400a1cbf4a5137d (n=20)`.

### Chat with the board

Build the interactive firmware, then run the host client from the `tools` branch:

```bash
git checkout tools
PORT=/dev/ttyACM0 DEFINES="-DUSE_DISPLAY=0 -DINTERACTIVE=1" scripts/deploy.sh tinystories
uv run --no-project --with pyserial --with 'tokenizers==0.23.1' \
  python scripts/insiderllm/chat.py --port /dev/ttyACM0 --temp 0.8 --top-p 0.9
```

Type a prompt per line, and each reply streams back as it is generated.
- **Prompts are one-shot:** the board clears its KV cache for every prompt.
- **Greedy and its limits:** `--temp 0` is the benchmark's greedy argmax. `--eot 1` (the default) stops at end of
  story.
- **Speed:** greedy decode runs at 43.33 ms/token. Sampling adds about 32 ms per token outside the forward pass, so
  sampled chat comes out near 13 tok/s.
- **It is still TinyStories:** it writes children's stories, and it likes Lily.

### Measurements

The public records hold logs, patch hashes, build info and the pre-registered predictions. They are in
[msb-msb/insiderllm-bench-results](https://github.com/msb-msb/insiderllm-bench-results):
- [reproduction of upstream](https://github.com/msb-msb/insiderllm-bench-results/tree/main/miu-esp32-s3-repro-2026-10-10/)
- [CPU-clock sweep](https://github.com/msb-msb/insiderllm-bench-results/tree/main/miu-cpu-clock-01-2026-10-10/)
- [SIMD head](https://github.com/msb-msb/insiderllm-bench-results/tree/main/miu-simd-head-01-2026-10-10/)
- [int4 head](https://github.com/msb-msb/insiderllm-bench-results/tree/main/miu-int4-head-01-2026-10-10/)
- [SIMD core](https://github.com/msb-msb/insiderllm-bench-results/tree/main/miu-simd-core-01-2026-10-10/)
- [training reproduction](https://github.com/msb-msb/insiderllm-bench-results/tree/main/tamanna-ts-repro-2026-10-10/)
- [story-boundary decoding test](https://github.com/msb-msb/insiderllm-bench-results/tree/main/tamanna-decode-01-2026-10-10/)

The records for `eot-stop`, `logits-sram`, `int4-core`, `interactive` and `prefill-skip` (measured 2026-10-11) will
be published to the same repository.

Changes (c) 2026 Mark Bartlett / InsiderLLM, under the same MIT License as upstream (`LICENSE`, unchanged).

---

# Running a 28.9M parameter LLM on a microcontroller

<p align="center">
  <a href="https://x.com/slvDev">𝕏 slvDev</a> &nbsp;·&nbsp;
  <a href="https://www.linkedin.com/in/slvdev/">LinkedIn</a>
</p>

![28.9M-parameter LLM running on an ESP32-S3](media/esp32-ple-demo.gif)

This is a 28.9 million parameter language model that generates text on an ESP32-S3
microcontroller. It runs on the chip itself, with nothing sent to a server, and it
displays generated text at 9.88 tokens per second on a small screen wired to the
chip. It fits because most of the model lives in flash instead of RAM, using
Per-Layer Embeddings, an idea from Google's Gemma 3n.

The same chip also runs a port of part of the fruit fly connectome: 48,311 neurons
and 9.46 million connections from a real fly brain, escaping a spider on screen.
See [Porting part of the fruit fly connectome to ESP32](#porting-part-of-the-fruit-fly-connectome-to-esp32).

## The numbers

|              |                                                    |
| ------------ | -------------------------------------------------- |
| Parameters   | 28.9M stored (25M of them in a flash lookup table) |
| Chip         | ESP32-S3, 512KB SRAM, 8MB PSRAM and 16MB flash     |
| Speed        | 9.88 tok/s end to end, 94.9 ms/token of compute    |
| Connectivity | none, everything runs on the device                |
| Model size   | 14.9MB at 4-bit                                    |

## Why it is hard, and how it fits anyway

A microcontroller has very little fast memory. The ESP32-S3 gives you 512KB of
SRAM, and only the values touched many times per token can live there:
activations and norm weights. The dense core and output head, scanned once per
position, sit in PSRAM. What is left is the embedding tables, and their size is
what normally decides how big a model can be.

In this model, most parameters sit in an embedding table, which the model reads
from rather than computes on. So that 25-million-parameter table stays in slow
flash, and only the few rows each token needs are pulled from it, about 450
bytes. Most of the model is therefore never loaded to run it: it sits in flash
and is sampled a little at a time.

That idea is Google's Per-Layer Embeddings, from
[Gemma 3n](https://ai.google.dev/gemma/docs/gemma-3n). Here it runs
on the memory layout of a microcontroller instead of a phone or a GPU.

Each tier holds whatever is read at its own frequency:

```
  SRAM  (fast, tiny)   activations and norm weights, touched many times a token
  PSRAM (medium)       the core and output head, read once per position
  FLASH (huge, slow)   the 25M-param table, about 6 rows read per token (~450 B)
```

## What it does, and what it does not

The model was trained on TinyStories, so it writes short, simple stories and mostly
keeps them coherent. It will not answer questions, follow instructions, write code,
or know facts. That limit comes from the small part of the model that does the
reasoning, and the memory trick does not change it. What is interesting here is the
architecture, fitting a large model onto a tiny chip, rather than what a 28.9 million
parameter model can say.

# Porting part of the fruit fly connectome to ESP32

This is a port of part of a real brain to the same board. It holds 48,311 neurons and
9,462,135 connections from the male fruit fly connectome, MaleCNS v1.0, with the
synaptic contact count of every connection kept exactly. The graph is 13.5MiB
compressed in flash, and every simulation step goes through all of its
connections, on both cores with the chip's SIMD instructions.

It is only part of the brain because the whole nervous system does not fit: the
smallest lossless encoding we found is 24.49MiB, against about 14MiB of flash left
for it. [Why only part of it](docs/fly-connectome/README.md#why-only-part-of-the-nervous-system).

## The numbers

|             |                                                            |
| ----------- | ---------------------------------------------------------- |
| Neurons     | 48,311: central brain, visual projection and descending   |
| Connections | 9,462,135, carrying 49,481,754 synaptic contacts          |
| Graph size  | 13.5MiB compressed in flash, blocks cached in PSRAM       |
| Decisions   | a new decision about every 1.7 s                          |
| Trained     | nothing: the demo reads the fly's own escape neurons      |

## Escaping a spider, with nothing trained

To show that it computes something, it escapes a spider on the OLED. The
spider's growing size drives the fly's own looming detectors, LC4 and LPLC2, on
the side it comes from. When the escape neurons on that side get active enough,
including DNp01, the giant fiber that makes a real fly jump, the fly leaps away
from that side.

No model was trained for this. The port ships a small text file naming those
input and output neurons and one threshold; the behavior comes from the fly's
own wiring. I checked it first: stimulating the looming detectors activates
those escape neurons 30 to 100 times more than random visual neurons do, and
ranking all 1,316 descending neurons by specificity puts the known escape
neurons on top.

It is a selected part of the nervous system, not the whole fly: what is ported,
what is left out and how it runs is in
[`docs/fly-connectome/`](docs/fly-connectome/README.md).

# Run it yourself

## Models

- [Barista](https://huggingface.co/slvDev/esp32-ai-barista) - espresso question answering
- [TinyStories](https://huggingface.co/slvDev/esp32-ai-tinystories) - story generation
- [Fly](https://huggingface.co/slvDev/esp32-ai-fly) - fruit fly connectome escaping a spider

## Commands

Download and deployment are separate operations: one reaches the network, the
other touches the board.

```bash
scripts/fetch_model.sh barista   # download, verify, install into artifacts/
scripts/deploy.sh barista        # generate headers, run gates, compile, flash
```

`tinystories` and `fly` take the same two commands. Each requires the model to
be named, because the board holds one at a time and deploying replaces it.

`fetch_model.sh` checks the inference assets against a SHA-256 and byte size
pinned in the script, and cross-checks the release's own `metadata.json` against
those same pins. It installs nothing unless every check passes, so a failed
download leaves what you already have untouched. `deploy.sh` downloads no model:
it works from whatever is already in `artifacts/<model>/`. It does run two of its
header tools through `uv`, which fetches one pinned wheel the first time on a
machine that has never cached it.

The firmware details and the boot output to expect live in
[`firmware/esp32_barista/README.md`](firmware/esp32_barista/README.md),
[`firmware/esp32_tinystories/README.md`](firmware/esp32_tinystories/README.md) and
[`firmware/esp32_fly/README.md`](firmware/esp32_fly/README.md). The reusable
architecture is in `src/`; the training, ablation and quantization code that
reproduces the published numbers is in `research/tinystories/`, and the tools
that find a circuit in the fly graph and export it are in `research/fly/`. The
language model's full method, its ablations and its on-chip measurements are
written up in [`RESULTS.md`](RESULTS.md); the fly's measurements are in
[`firmware/esp32_fly/README.md`](firmware/esp32_fly/README.md#measured).

## Credit

TinyStories is the dataset this trains on: short synthetic stories simple enough
that a small model can still learn to write coherently (Ronen Eldan and Yuanzhi Li,
Microsoft Research, [arXiv:2305.07759](https://arxiv.org/abs/2305.07759)). The other
half is Per-Layer Embeddings, Google's design from Gemma 3n, which is what
lets a big model fit on a small chip.

Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c) is the
reference for training a small language model and running it in plain C.

The fly graph comes from MaleCNS v1.0, the male Drosophila central nervous system
connectome by the FlyEM Project at HHMI Janelia, the University of Cambridge, the
MRC Laboratory of Molecular Biology and Google Research, used under CC BY 4.0.
What was selected and changed is in
[`docs/fly-connectome/ATTRIBUTION.md`](docs/fly-connectome/ATTRIBUTION.md).

## Measurements

Detailed measurements and ablations are documented in `RESULTS.md`.
