#!/usr/bin/env python3
"""Copy Slang libraries into the Python package tree for **release wheel** builds.

Not used for local development: `pip install -e ".[dev]"` embeds Slang via
`goldy/build.rs` at compile time.

Run before `maturin build --release` (CI / PyPI publish):

    python build-slang.py
    maturin build --release

Copies from `../slang/bin/{platform}/` to `python/goldy/` for `pyproject.toml`
`include` globs.
"""

import json
import platform
import shutil
import sys
from pathlib import Path


def get_platform_key():
    """Get the platform key for manifest.json."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "windows":
        return "windows-x86_64"
    elif system == "linux":
        if machine in ("x86_64", "amd64"):
            return "linux-x86_64"
        elif machine in ("aarch64", "arm64"):
            return "linux-aarch64"
    elif system == "darwin":
        if machine in ("x86_64", "amd64"):
            return "macos-x86_64"
        elif machine in ("arm64", "aarch64"):
            return "macos-aarch64"

    return None


def main():
    script_dir = Path(__file__).parent
    goldy_root = script_dir.parent

    # Load manifest
    manifest_path = goldy_root / "slang" / "manifest.json"
    if not manifest_path.exists():
        print(f"Error: {manifest_path} not found")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Get platform info
    platform_key = get_platform_key()
    if not platform_key:
        print(f"Error: Unsupported platform: {platform.system()} {platform.machine()}")
        sys.exit(1)

    if platform_key not in manifest["platforms"]:
        print(f"Error: Platform {platform_key} not in manifest")
        sys.exit(1)

    platform_info = manifest["platforms"][platform_key]
    slang_dir = goldy_root / "slang" / "bin" / platform_key

    if not slang_dir.exists():
        print(f"Error: Slang binaries not found at {slang_dir}")
        print("Run slang/download.sh to download Slang binaries")
        sys.exit(1)

    # Destination directory
    dest_dir = script_dir / "python" / "goldy"
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Copy files
    copied = 0
    for filename in platform_info["files"]:
        src = slang_dir / filename
        if src.exists():
            dest = dest_dir / filename
            shutil.copy2(src, dest)
            print(f"  Copied {filename}")
            copied += 1
        else:
            print(f"  Warning: {filename} not found")

    print(f"\nCopied {copied}/{len(platform_info['files'])} Slang libraries to {dest_dir}")

    if copied == 0:
        print("Error: No Slang libraries were copied!")
        sys.exit(1)

    print("\nSlang libraries ready. Now run:")
    print("  maturin build --release")


if __name__ == "__main__":
    main()
