#!/usr/bin/env bash
# Run cargo with GOLDY_VALIDATION=1 so Vulkan enables Khronos validation (when built with
# the vulkan feature) and Metal sets MTL_SHADER_VALIDATION before device creation.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export GOLDY_VALIDATION=1
exec cargo "$@"
