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

echo "WASM build complete!"
echo ""
echo "Files in docs/src/wasm/:"
ls -la ../docs/src/wasm/

echo ""
echo "Required files for demos:"
echo "  - rag_web.js, rag_web_bg.wasm (demos)"
echo "  - slang-wasm.js, slang-wasm.wasm (shader compiler, ~15MB)"
echo ""
echo "To test locally:"
echo "  cd ../docs && mdbook build && cd book && python -m http.server 8080"
echo "  Then open http://localhost:8080"

