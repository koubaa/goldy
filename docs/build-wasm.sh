#!/bin/bash
# Build WASM demos for RAG documentation
# This is internal tooling - the WASM module powers interactive docs demos

set -e

cd "$(dirname "$0")/wasm-demos"

echo "Building documentation WASM demos..."

# Check for wasm-pack
if ! command -v wasm-pack &> /dev/null; then
    echo "Installing wasm-pack..."
    cargo install wasm-pack
fi

# Build the library
wasm-pack build --target web --out-dir ../src/wasm

echo "WASM build complete!"
echo ""
echo "Files in docs/src/wasm/:"
ls -la ../src/wasm/

echo ""
echo "Required files for demos:"
echo "  - rag_docs_wasm.js, rag_docs_wasm_bg.wasm (demos)"
echo "  - slang-wasm.js, slang-wasm.wasm (shader compiler, ~15MB)"
echo ""
echo "To test locally:"
echo "  cd .. && mdbook build && cd book && python -m http.server 8080"
echo "  Then open http://localhost:8080"
