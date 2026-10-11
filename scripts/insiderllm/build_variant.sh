#!/usr/bin/env bash
# MIT License. Original esp32-ai (c) 2026 Viacheslav Sierbov. Changes (c) 2026 Mark Bartlett / InsiderLLM.
#
# Build and flash the esp32_tinystories firmware only (not the model), with
# extra defines and FQBN overrides, into a separate build directory per
# configuration. This is the fork version of the tool used for the
# measurements: there the changes were patches applied to a copy of upstream;
# here they are commits, so it builds the checked-out tree as it is.
#
# usage: PORT=/dev/ttyACM0 [DEFINES="-DX=1 ..."] [NO_UPLOAD=1] \
#          scripts/insiderllm/build_variant.sh [KEY=VAL[,KEY=VAL...]]
#   KEY=VAL  board options replacing those in deploy.sh's FQBN, e.g. CPUFreq=160.
#            Only keys already in the base FQBN are accepted.
#
# The model partition is NOT rewritten: run scripts/deploy.sh tinystories once
# first (it also generates generated/vocab.h) and check the boot "fp=" line.
# DEFINES goes to compiler.cpp.extra_flags (empty in core 3.3.10's platform.txt).
# Output: build/insiderllm/<slug>/ with BUILDINFO.txt (commit, defines, FQBN).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SKETCH="$REPO/firmware/esp32_tinystories"
BASE_FQBN='esp32:esp32:esp32s3:UploadSpeed=921600,USBMode=hwcdc,CDCOnBoot=cdc,UploadMode=default,CPUFreq=240,FlashMode=qio,FlashSize=16M,PartitionScheme=custom,PSRAM=opi,DebugLevel=info'
OPT_FLAGS='-O3'   # same as scripts/deploy.sh for tinystories

[ $# -le 1 ] || { sed -n '4,18p' "$0" >&2; exit 2; }
OVERRIDES=${1:-}
DEFINES=${DEFINES:-}
[ -f "$SKETCH/generated/vocab.h" ] || {
  echo "missing generated/vocab.h; run scripts/deploy.sh tinystories first" >&2; exit 1; }
if [ -z "${NO_UPLOAD:-}" ]; then
  [ -n "${PORT:-}" ] || { echo "set PORT=... (native USB node)" >&2; exit 1; }
fi

# --- FQBN: replace option values, reject unknown keys --------------------------
prefix=${BASE_FQBN%%:UploadSpeed*}            # esp32:esp32:esp32s3
IFS=, read -r -a opts <<< "${BASE_FQBN#"$prefix":}"
if [ -n "$OVERRIDES" ]; then
  IFS=, read -r -a ovs <<< "$OVERRIDES"
  for ov in "${ovs[@]}"; do
    k=${ov%%=*}; hit=0
    for i in "${!opts[@]}"; do
      if [ "${opts[$i]%%=*}" = "$k" ]; then opts[$i]=$ov; hit=1; fi
    done
    [ $hit -eq 1 ] || { echo "unknown FQBN key: $k" >&2; exit 1; }
  done
fi
FQBN="$prefix:$(IFS=,; echo "${opts[*]}")"
SLUG=${OVERRIDES:-base}
[ -n "$DEFINES" ] && SLUG="$SLUG ${DEFINES//-D/}"
SLUG=${SLUG//[=, ]/_}

BUILD="$REPO/build/insiderllm/$SLUG"
mkdir -p "$BUILD"
{
  echo "date:    $(date -Iseconds)"
  echo "commit:  $(git -C "$REPO" rev-parse HEAD)$(git -C "$REPO" diff --quiet || echo ' (dirty)')"
  echo "branch:  $(git -C "$REPO" rev-parse --abbrev-ref HEAD)"
  echo "defines: ${DEFINES:-(none)}"
  echo "fqbn:    $FQBN"
  echo "opt:     $OPT_FLAGS"
  echo "ino:     $(sha256sum "$SKETCH/esp32_tinystories.ino" | cut -d' ' -f1)"
  echo "vocab:   $(sha256sum "$SKETCH/generated/vocab.h" | cut -d' ' -f1)"
} > "$BUILD/BUILDINFO.txt"
cat "$BUILD/BUILDINFO.txt"

echo "=== compile ==="
arduino-cli compile --fqbn "$FQBN" \
  --build-property "compiler.optimization_flags=$OPT_FLAGS" \
  --build-property "compiler.cpp.extra_flags=$DEFINES" \
  --build-path "$BUILD" "$SKETCH" 2>&1 | tail -n 2

[ -n "${NO_UPLOAD:-}" ] && { echo "NO_UPLOAD set; not flashing"; exit 0; }
echo "=== upload firmware -> $PORT ==="
arduino-cli upload -p "$PORT" --fqbn "$FQBN" --input-dir "$BUILD" "$SKETCH" 2>&1 | tail -n 1
