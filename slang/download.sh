#!/bin/bash
# Downloads Slang binaries for all supported platforms
# Usage: ./download.sh [version]
#
# This script downloads pre-built Slang binaries from GitHub releases
# and extracts them to the appropriate bin/ subdirectories.

set -e

VERSION="${1:-2025.24.3}"
BASE_URL="https://github.com/shader-slang/slang/releases/download/v${VERSION}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Downloading Slang v${VERSION} binaries..."

# Platform configurations: "github-name local-dir"
PLATFORMS=(
    "windows-x86_64 windows-x86_64"
    "linux-x86_64 linux-x86_64"
    "linux-aarch64 linux-aarch64"
    "macos-x86_64 macos-x86_64"
    "macos-aarch64 macos-aarch64"
)

for platform_pair in "${PLATFORMS[@]}"; do
    read -r github_name local_dir <<< "$platform_pair"
    
    echo "  Downloading ${github_name}..."
    
    mkdir -p "bin/${local_dir}"
    
    ZIP_URL="${BASE_URL}/slang-${VERSION}-${github_name}.zip"
    TMP_ZIP="/tmp/slang-${github_name}.zip"
    TMP_DIR="/tmp/slang-extract-${github_name}"
    
    if curl -fsSL "$ZIP_URL" -o "$TMP_ZIP" 2>/dev/null; then
        # Extract to temp directory first
        rm -rf "$TMP_DIR"
        mkdir -p "$TMP_DIR"
        unzip -q -o "$TMP_ZIP" -d "$TMP_DIR"
        
        # Copy libraries - handle different archive structures
        # Archives may have files in: lib/, bin/, or nested in a subdirectory
        # Use -L to dereference symlinks (Linux uses symlinks for .so files)
        
        # Try direct lib/ and bin/ directories first (newer archive format)
        for dir in "$TMP_DIR/lib" "$TMP_DIR/bin"; do
            if [ -d "$dir" ]; then
                # Windows: slang.dll, slang-glslang.dll
                cp -L "$dir"/slang*.dll "bin/${local_dir}/" 2>/dev/null || true
                # Linux: libslang*.so* (including versioned like libslang-glslang-2025.24.3.so)
                cp -L "$dir"/libslang*.so* "bin/${local_dir}/" 2>/dev/null || true
                # macOS: libslang*.dylib
                cp -L "$dir"/libslang*.dylib "bin/${local_dir}/" 2>/dev/null || true
            fi
        done
        
        # Try nested subdirectory (older archive format: slang-VERSION-PLATFORM/lib/)
        for dir in "$TMP_DIR"/*/lib "$TMP_DIR"/*/bin; do
            if [ -d "$dir" ]; then
                cp -L "$dir"/slang*.dll "bin/${local_dir}/" 2>/dev/null || true
                cp -L "$dir"/libslang*.so* "bin/${local_dir}/" 2>/dev/null || true
                cp -L "$dir"/libslang*.dylib "bin/${local_dir}/" 2>/dev/null || true
            fi
        done
        
        # Cleanup
        rm -rf "$TMP_DIR" "$TMP_ZIP"
        
        # Verify what we got
        count=$(ls -1 "bin/${local_dir}/"*slang* 2>/dev/null | wc -l || echo 0)
        if [ "$count" -gt 0 ]; then
            echo "    ✓ ${github_name} (${count} files)"
        else
            echo "    ✗ ${github_name} (no libraries found in archive)"
        fi
    else
        echo "    ✗ ${github_name} (not available or download failed)"
    fi
done

echo ""
echo "Done! Libraries downloaded to bin/"
ls -la bin/*/

echo ""
echo "Note: For development, you can also use slang from your Vulkan SDK:"
echo "  - Windows: C:/VulkanSDK/*/Bin/slang.dll"
echo "  - Linux: /usr/share/vulkan/..."
