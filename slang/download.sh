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
        
        # Find and copy all required libraries
        # We need: slang, slang-glslang (for SPIR-V generation)
        find "$TMP_DIR" -type f \( \
            -name "libslang.so*" -o \
            -name "libslang-glslang.so*" -o \
            -name "slang.dll" -o \
            -name "slang-glslang.dll" -o \
            -name "libslang.dylib" -o \
            -name "libslang-glslang.dylib" \
        \) -exec cp {} "bin/${local_dir}/" \; 2>/dev/null || true
        
        # Also try lib/ and bin/ subdirectories explicitly
        for subdir in lib bin release/lib release/bin; do
            if [ -d "$TMP_DIR"/*/"$subdir" ] 2>/dev/null; then
                cp "$TMP_DIR"/*/"$subdir"/*slang*.so* "bin/${local_dir}/" 2>/dev/null || true
                cp "$TMP_DIR"/*/"$subdir"/*slang*.dll "bin/${local_dir}/" 2>/dev/null || true
                cp "$TMP_DIR"/*/"$subdir"/*slang*.dylib "bin/${local_dir}/" 2>/dev/null || true
            fi
        done
        
        # Cleanup
        rm -rf "$TMP_DIR" "$TMP_ZIP"
        
        # Verify what we got
        count=$(ls -1 "bin/${local_dir}/"*slang* 2>/dev/null | wc -l)
        echo "    ✓ ${github_name} (${count} files)"
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
