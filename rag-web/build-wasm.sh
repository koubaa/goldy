#!/bin/bash
# Build rag-web WASM module for documentation

set -e

echo "Building rag-web for WebGPU..."

# Check for wasm-pack
if ! command -v wasm-pack &> /dev/null; then
    echo "Installing wasm-pack..."
    cargo install wasm-pack
fi

# Build the library (not example, library exports the demo)
wasm-pack build --target web --out-dir ../docs/src/wasm

echo "WASM build complete! Output in docs/src/wasm/"
echo ""
echo "To test locally:"
echo "  cd ../docs && mdbook serve --open"

