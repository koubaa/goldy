#!/usr/bin/env bash
# Run all goldy examples one at a time. Each example blocks until the user
# dismisses it (e.g. Esc or close window), then the next one starts.
#
# Prints wall-clock duration for each example, and average FPS when the
# example outputs a GOLDY_PERF line (format: GOLDY_PERF: frames=N elapsed=Ts avg_fps=F).
#
# Usage:
#   GOLDY_BACKEND=vk ./run_all_examples.sh          # normal
#   GOLDY_BACKEND=cuda ./run_all_examples.sh         # enables cuda feature + CUDA backend
#   SLEEP_BETWEEN=3 ./run_all_examples.sh            # 3s sleep between examples
#   EXAMPLE_TIMEOUT=10 ./run_all_examples.sh         # auto-exit each example after 10s
#   VULKAN_VALIDATE=1 ./run_all_examples.sh          # with validation layers

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SLEEP_BETWEEN="${SLEEP_BETWEEN:-0}"
EXAMPLE_TIMEOUT="${EXAMPLE_TIMEOUT:-0}"

CARGO_FEATURES="examples"
case "${GOLDY_BACKEND,,}" in
    cuda)
        CARGO_FEATURES="examples,cuda"
        echo "GOLDY_BACKEND=cuda: building with --features $CARGO_FEATURES"
        ;;
esac

if [[ "${VULKAN_VALIDATE:-0}" == "1" ]]; then
    export VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation
    echo "Vulkan validation layers ENABLED"
fi

if [[ "$EXAMPLE_TIMEOUT" -gt 0 ]]; then
    export GOLDY_EXAMPLE_TIMEOUT="$EXAMPLE_TIMEOUT"
fi

# Extract example names from Cargo.toml
EXAMPLES=()
while IFS= read -r name; do
    [[ -n "$name" ]] && EXAMPLES+=("$name")
done < <(grep -A1 '\[\[example\]\]' Cargo.toml | grep 'name = ' | sed 's/.*name = "\([^"]*\)".*/\1/')

if [[ ${#EXAMPLES[@]} -eq 0 ]]; then
    echo "ERROR: No examples found in Cargo.toml"
    exit 1
fi

# Collect results for the summary table
declare -a RESULTS

total=${#EXAMPLES[@]}
for i in "${!EXAMPLES[@]}"; do
    name="${EXAMPLES[$i]}"
    [[ -z "$name" ]] && continue
    n=$((i + 1))
    echo ""
    echo "========================================"
    echo "[$n/$total] Running example: $name"
    if [[ "$EXAMPLE_TIMEOUT" -gt 0 ]]; then
        echo "Auto-advancing after ${EXAMPLE_TIMEOUT}s (EXAMPLE_TIMEOUT)"
    else
        echo "Close the window or press Esc to continue to next"
    fi
    echo "========================================"

    start_ns=$(date +%s%N 2>/dev/null || python3 -c 'import time; print(int(time.time()*1e9))')
    output=$(cargo run --example "$name" --features "$CARGO_FEATURES" 2>&1) || true
    end_ns=$(date +%s%N 2>/dev/null || python3 -c 'import time; print(int(time.time()*1e9))')

    echo "$output"

    wall_ms=$(( (end_ns - start_ns) / 1000000 ))
    wall_s=$(awk "BEGIN {printf \"%.2f\", $wall_ms / 1000}")

    # Look for GOLDY_PERF line from the example
    perf_line=$(echo "$output" | grep '^GOLDY_PERF:' | tail -1)
    if [[ -n "$perf_line" ]]; then
        fps=$(echo "$perf_line" | sed 's/.*avg_fps=\([0-9.]*\).*/\1/')
        frames=$(echo "$perf_line" | sed 's/.*frames=\([0-9]*\).*/\1/')
        RESULTS+=("$(printf "%-25s %8s ms  %6s frames  %7s fps" "$name" "$wall_ms" "$frames" "$fps")")
    else
        RESULTS+=("$(printf "%-25s %8s ms" "$name" "$wall_ms")")
    fi

    if [[ "$SLEEP_BETWEEN" -gt 0 ]] && [[ $n -lt $total ]]; then
        echo "Sleeping ${SLEEP_BETWEEN}s before next example..."
        sleep "$SLEEP_BETWEEN"
    fi
done

echo ""
echo "========================================"
echo "Summary ($total examples)"
echo "========================================"
printf "%-25s %11s  %13s  %11s\n" "Example" "Wall time" "Frames" "Avg FPS"
printf "%-25s %11s  %13s  %11s\n" "-------" "---------" "------" "-------"
for line in "${RESULTS[@]}"; do
    echo "$line"
done
echo ""
