#!/usr/bin/env bash
# Run all goldy examples one at a time. Each example blocks until the user
# dismisses it (e.g. Esc or close window), then the next one starts.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Extract example names from Cargo.toml (name line immediately after [[example]])
EXAMPLES=()
while IFS= read -r name; do
    [[ -n "$name" ]] && EXAMPLES+=("$name")
done < <(grep -A1 '\[\[example\]\]' Cargo.toml | grep 'name = ' | sed 's/.*name = "\([^"]*\)".*/\1/')

# Fallback if extraction fails
if [[ ${#EXAMPLES[@]} -eq 0 ]]; then
    EXAMPLES=(triangle digital_clock gradient plasma starfield mandelbrot bouncing_lines spinning_cube metaballs checkerboard instancing particles waveform tunnel multi_window compute_particles game_of_life depth_quads)
fi

total=${#EXAMPLES[@]}
for i in "${!EXAMPLES[@]}"; do
    name="${EXAMPLES[$i]}"
    [[ -z "$name" ]] && continue
    n=$((i + 1))
    echo ""
    echo "========================================"
    echo "[$n/$total] Running example: $name"
    echo "Close the window or press Esc to continue to next"
    echo "========================================"
    cargo run --example "$name"
done

echo ""
echo "All $total examples finished."
