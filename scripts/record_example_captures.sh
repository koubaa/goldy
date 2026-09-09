#!/usr/bin/env bash
# Dump every windowed example as packed RGBA frames, then stitch a looping WebM.
#
# Examples render into a leased colour target and withdraw the pixels. This script
# points GOLDY_EXAMPLE_CAPTURE at a raw file and runs ffmpeg over it. Nothing is
# grabbed from a desktop window — no Xvfb, no x11grab.
#
# Usage:
#   scripts/record_example_captures.sh                 # every example
#   scripts/record_example_captures.sh triangle plasma # a subset
#
# Environment:
#   GOLDY_BACKEND   backend to record with (default: crate default, Vulkan)
#   FRAMES          frames per clip (default: 75)
#   FPS             virtual / output frame rate (default: 15)
#   WIDTH HEIGHT    capture size (default: 640x480; multi_window is 960x320)
#   OUT_DIR         output directory (default: docs/src/assets/examples)
#
# Rebuild the whole set after changing an example's visuals; the book embeds
# these clips directly.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

GOLDY_BACKEND="${GOLDY_BACKEND:-}"
FRAMES="${FRAMES:-75}"
FPS="${FPS:-15}"
WIDTH="${WIDTH:-640}"
HEIGHT="${HEIGHT:-480}"
OUT_DIR="${OUT_DIR:-docs/src/assets/examples}"

case "$GOLDY_BACKEND" in
    webgpu | wgpu) CARGO_ARGS=(--no-default-features --features webgpu,examples) ;;
    cuda) CARGO_ARGS=(--features examples,cuda) ;;
    *) CARGO_ARGS=(--features examples) ;;
esac

mapfile -t ALL_EXAMPLES < <(grep -A1 '\[\[example\]\]' Cargo.toml | grep 'name = ' | sed 's/.*name = "\([^"]*\)".*/\1/')

if [[ $# -gt 0 ]]; then
    EXAMPLES=("$@")
else
    EXAMPLES=("${ALL_EXAMPLES[@]}")
fi

echo "Building examples${GOLDY_BACKEND:+ for $GOLDY_BACKEND}..."
cargo build --release "${CARGO_ARGS[@]}" --examples

mkdir -p "$OUT_DIR"
RAW_DIR="$(mktemp -d)"
trap 'rm -rf "$RAW_DIR"' EXIT

capture_size_for() {
    case "$1" in
        multi_window) echo "960 320" ;;
        *) echo "$WIDTH $HEIGHT" ;;
    esac
}

for name in "${EXAMPLES[@]}"; do
    binary="target/release/examples/$name"
    if [[ ! -x "$binary" ]]; then
        echo "skip  $name (not built)"
        continue
    fi

    read -r cap_w cap_h < <(capture_size_for "$name")
    raw="$RAW_DIR/$name.rgba"
    log="/tmp/goldy-capture-$name.log"

    echo "dump  $name (${cap_w}x${cap_h}, ${FRAMES} frames @ ${FPS} fps)"
    set +e
    env ${GOLDY_BACKEND:+GOLDY_BACKEND="$GOLDY_BACKEND"} \
        GOLDY_EXAMPLE_CAPTURE="$raw" \
        GOLDY_EXAMPLE_CAPTURE_FRAMES="$FRAMES" \
        GOLDY_EXAMPLE_CAPTURE_FPS="$FPS" \
        GOLDY_EXAMPLE_CAPTURE_WIDTH="$cap_w" \
        GOLDY_EXAMPLE_CAPTURE_HEIGHT="$cap_h" \
        "$binary" >"$log" 2>&1
    status=$?
    set -e

    if [[ "$status" -ne 0 ]]; then
        echo "      failed (exit $status), see $log"
        continue
    fi

    if [[ ! -s "$raw" ]]; then
        echo "      skipped (no pixels — adapter likely missing a required capability)"
        continue
    fi

    expected=$((cap_w * cap_h * 4 * FRAMES))
    actual=$(wc -c <"$raw")
    if [[ "$actual" -ne "$expected" ]]; then
        echo "      raw size $actual != $expected, see $log"
        continue
    fi

    ffmpeg -y -loglevel error \
        -f rawvideo -pix_fmt rgba -s:v "${cap_w}x${cap_h}" -r "$FPS" -i "$raw" \
        -c:v libvpx-vp9 -crf 45 -b:v 500k -deadline good -cpu-used 2 -an \
        "$OUT_DIR/$name.webm"

    printf '      %s\n' "$(du -h "$OUT_DIR/$name.webm" | cut -f1)"
done

echo "Wrote captures to $OUT_DIR"
