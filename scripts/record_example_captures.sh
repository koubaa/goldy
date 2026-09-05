#!/usr/bin/env bash
# Record every windowed example into a short, looping WebM for the book.
#
# Each example runs for real against a GPU backend on a virtual X11 display,
# and ffmpeg grabs the window. Nothing is faked or hand-drawn: the clips under
# docs/src/assets/examples/ are the examples themselves.
#
# Requirements: Xvfb, ffmpeg (libvpx-vp9), and a backend that can present to
# X11. Goldy's Vulkan surface path is Wayland-only on Linux, so this defaults
# to the WebGPU backend, which reaches X11 through wgpu. Software rendering
# (lavapipe) is fine — the recordings are wall-clock, not benchmarks.
#
# Usage:
#   scripts/record_example_captures.sh                 # every example
#   scripts/record_example_captures.sh triangle plasma # a subset
#
# Environment:
#   GOLDY_BACKEND   backend to record with (default: webgpu)
#   DISPLAY_NUM     X display to spawn (default: 99)
#   WARMUP          seconds before recording starts (default: 6)
#   DURATION        seconds of video per example (default: 5)
#   OUT_DIR         output directory (default: docs/src/assets/examples)
#
# Rebuild the whole set after changing an example's visuals; the book embeds
# these clips directly.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

GOLDY_BACKEND="${GOLDY_BACKEND:-webgpu}"
DISPLAY_NUM="${DISPLAY_NUM:-99}"
WARMUP="${WARMUP:-6}"
DURATION="${DURATION:-5}"
OUT_DIR="${OUT_DIR:-docs/src/assets/examples}"
FRAMERATE=20
OUT_WIDTH=640

# Examples with no window to grab: both exit 0 when the backend lacks the
# capability they exist to show, which is the case on WebGPU.
SKIP=(mesh_triangle ray_query)

# Default window is 800x600; multi_window lays three of them out side by side.
screen_size_for() {
    case "$1" in
        multi_window) echo "1600x900" ;;
        *) echo "800x600" ;;
    esac
}

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

echo "Building examples for $GOLDY_BACKEND..."
cargo build --release "${CARGO_ARGS[@]}" --examples

mkdir -p "$OUT_DIR"

xvfb_pid=""
current_screen=""

stop_xvfb() {
    if [[ -n "$xvfb_pid" ]] && kill -0 "$xvfb_pid" 2>/dev/null; then
        kill "$xvfb_pid" 2>/dev/null || true
        wait "$xvfb_pid" 2>/dev/null || true
    fi
    xvfb_pid=""
    current_screen=""
}

start_xvfb() {
    local size="$1"
    [[ "$size" == "$current_screen" ]] && return
    stop_xvfb
    Xvfb ":$DISPLAY_NUM" -screen 0 "${size}x24" -nolisten tcp >/tmp/goldy-capture-xvfb.log 2>&1 &
    xvfb_pid=$!
    current_screen="$size"
    sleep 2
}

trap stop_xvfb EXIT

for name in "${EXAMPLES[@]}"; do
    if [[ " ${SKIP[*]} " == *" $name "* ]]; then
        echo "skip  $name (no window on $GOLDY_BACKEND)"
        continue
    fi

    binary="target/release/examples/$name"
    if [[ ! -x "$binary" ]]; then
        echo "skip  $name (not built)"
        continue
    fi

    size="$(screen_size_for "$name")"
    start_xvfb "$size"

    echo "rec   $name (${size}, ${DURATION}s)"
    DISPLAY=":$DISPLAY_NUM" GOLDY_BACKEND="$GOLDY_BACKEND" \
        GOLDY_EXAMPLE_TIMEOUT="$((WARMUP + DURATION + 4))" \
        "$binary" >"/tmp/goldy-capture-$name.log" 2>&1 &
    example_pid=$!

    sleep "$WARMUP"

    if ! kill -0 "$example_pid" 2>/dev/null; then
        echo "      exited during warmup, see /tmp/goldy-capture-$name.log"
        continue
    fi

    # Grab at native window size, then downscale — these ship in the repo, so
    # constrain the bitrate rather than chasing a pixel-exact capture.
    DISPLAY=":$DISPLAY_NUM" ffmpeg -y -loglevel error \
        -f x11grab -draw_mouse 0 -framerate "$FRAMERATE" -video_size "$size" -i ":$DISPLAY_NUM.0" \
        -t "$DURATION" -vf "fps=15,scale=${OUT_WIDTH}:-2:flags=lanczos" \
        -c:v libvpx-vp9 -crf 45 -b:v 500k -deadline good -cpu-used 2 -an \
        "$OUT_DIR/$name.webm"

    # The run limit normally ends the example on its own; don't hang on one
    # that ignores it.
    for _ in $(seq 1 10); do
        kill -0 "$example_pid" 2>/dev/null || break
        sleep 1
    done
    kill "$example_pid" 2>/dev/null || true
    wait "$example_pid" 2>/dev/null || true

    printf '      %s\n' "$(du -h "$OUT_DIR/$name.webm" | cut -f1)"
done

stop_xvfb
echo "Wrote captures to $OUT_DIR"
