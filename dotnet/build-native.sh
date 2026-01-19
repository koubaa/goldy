#!/bin/bash
# Build native libraries and copy Slang dependencies (run on each platform)

set -e

cd "$(dirname "$0")/.."

# Detect platform
case "$(uname -s)" in
    Linux*)
        TARGET="x86_64-unknown-linux-gnu"
        ARTIFACT="libgoldy_ffi.so"
        RUNTIME="linux-x64"
        SLANG_PLATFORM="linux-x86_64"
        ;;
    Darwin*)
        if [[ "$(uname -m)" == "arm64" ]]; then
            TARGET="aarch64-apple-darwin"
            RUNTIME="osx-arm64"
            SLANG_PLATFORM="macos-aarch64"
        else
            TARGET="x86_64-apple-darwin"
            RUNTIME="osx-x64"
            SLANG_PLATFORM="macos-x86_64"
        fi
        ARTIFACT="libgoldy_ffi.dylib"
        ;;
    MINGW*|MSYS*|CYGWIN*)
        TARGET="x86_64-pc-windows-msvc"
        ARTIFACT="goldy_ffi.dll"
        RUNTIME="win-x64"
        SLANG_PLATFORM="windows-x86_64"
        ;;
    *)
        echo "Unknown platform: $(uname -s)"
        exit 1
        ;;
esac

echo "Building for $TARGET..."
cargo build --release -p goldy-ffi --target $TARGET

# Copy FFI library to runtime folder
RUNTIME_DIR="dotnet/Goldy/runtimes/$RUNTIME/native"
mkdir -p "$RUNTIME_DIR"
cp "target/$TARGET/release/$ARTIFACT" "$RUNTIME_DIR/"

echo "Built $ARTIFACT for $RUNTIME"

# Copy Slang libraries
echo "Copying Slang libraries..."
SLANG_DIR="slang/bin/$SLANG_PLATFORM"

if [ -d "$SLANG_DIR" ]; then
    # Copy all library files from slang bin directory
    copied=0
    case "$(uname -s)" in
        Linux*)
            for f in "$SLANG_DIR"/*.so; do
                [ -f "$f" ] && cp "$f" "$RUNTIME_DIR/" && copied=$((copied + 1))
            done
            ;;
        Darwin*)
            for f in "$SLANG_DIR"/*.dylib; do
                [ -f "$f" ] && cp "$f" "$RUNTIME_DIR/" && copied=$((copied + 1))
            done
            ;;
        MINGW*|MSYS*|CYGWIN*)
            for f in "$SLANG_DIR"/*.dll; do
                [ -f "$f" ] && cp "$f" "$RUNTIME_DIR/" && copied=$((copied + 1))
            done
            ;;
    esac
    echo "Copied $copied Slang libraries to $RUNTIME_DIR"
else
    echo "Warning: Slang binaries not found at $SLANG_DIR"
    echo "Run slang/download.sh to download Slang binaries"
fi
