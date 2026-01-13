#!/bin/bash
# Build native libraries for all platforms (run on each platform)

set -e

cd "$(dirname "$0")/.."

# Detect platform
case "$(uname -s)" in
    Linux*)
        TARGET="x86_64-unknown-linux-gnu"
        ARTIFACT="libgoldy_ffi.so"
        RUNTIME="linux-x64"
        ;;
    Darwin*)
        # Check architecture
        if [[ "$(uname -m)" == "arm64" ]]; then
            TARGET="aarch64-apple-darwin"
            RUNTIME="osx-arm64"
        else
            TARGET="x86_64-apple-darwin"
            RUNTIME="osx-x64"
        fi
        ARTIFACT="libgoldy_ffi.dylib"
        ;;
    MINGW*|MSYS*|CYGWIN*)
        TARGET="x86_64-pc-windows-msvc"
        ARTIFACT="goldy_ffi.dll"
        RUNTIME="win-x64"
        ;;
    *)
        echo "Unknown platform: $(uname -s)"
        exit 1
        ;;
esac

echo "Building for $TARGET..."
cargo build --release -p goldy-ffi --target $TARGET

# Copy to runtime folder
mkdir -p dotnet/Goldy/runtimes/$RUNTIME/native
cp target/$TARGET/release/$ARTIFACT dotnet/Goldy/runtimes/$RUNTIME/native/

echo "Built $ARTIFACT for $RUNTIME"

