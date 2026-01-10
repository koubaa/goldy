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
    "windows-aarch64 windows-aarch64"
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
    
    if curl -fsSL "$ZIP_URL" -o "$TMP_ZIP" 2>/dev/null; then
        # Extract only the library files we need
        unzip -o -j "$TMP_ZIP" "*/lib/*slang-compiler*" -d "bin/${local_dir}/" 2>/dev/null || \
        unzip -o -j "$TMP_ZIP" "*slang-compiler*" -d "bin/${local_dir}/" 2>/dev/null || \
        unzip -o -j "$TMP_ZIP" "*/bin/*slang-compiler*" -d "bin/${local_dir}/" 2>/dev/null || true
        
        rm -f "$TMP_ZIP"
        echo "    ✓ ${github_name}"
    else
        echo "    ✗ ${github_name} (not available or download failed)"
    fi
done

echo ""
echo "Done! Libraries downloaded to bin/"
echo ""
echo "Note: For development, you can also use slang from your Vulkan SDK:"
echo "  - Windows: C:/VulkanSDK/*/Bin/slang.dll"
echo "  - Linux: /usr/share/vulkan/..."

