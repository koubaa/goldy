#!/bin/bash
# Build WASM artifacts for RAG documentation
#
# Prerequisites:
#   - wasm-pack: cargo install wasm-pack
#   - slang-wasm: download from slang-wasm releases

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAG_ROOT="$(dirname "$SCRIPT_DIR")"
WASM_OUT="$SCRIPT_DIR/src/wasm"

echo "Building rag-web WASM..."
cd "$RAG_ROOT/rag-web"
wasm-pack build --target web --out-dir "$WASM_OUT"

echo ""
echo "WASM build complete!"
echo "Output: $WASM_OUT"
echo ""
echo "NOTE: You still need to copy slang-wasm files manually:"
echo "  cp slang-wasm.js slang-wasm.wasm $WASM_OUT/"

