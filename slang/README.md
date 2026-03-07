# Vendored Slang Binaries

This directory contains vendored [Slang](https://github.com/shader-slang/slang) shader compiler binaries.

## Structure

```
slang/
├── manifest.json      # Version and file list (source of truth)
├── bin/
│   ├── windows-x86_64/
│   │   └── slang-compiler.dll, slang.dll, slang-rt.dll, ...
│   ├── linux-x86_64/
│   │   └── libslang-compiler.so, libslang.so, ...
│   ├── macos-aarch64/
│   │   └── libslang-compiler.dylib, libslang.dylib, ...
│   └── ...
├── include/
│   └── slang.h
├── download.sh
└── README.md
```

## Downloading Binaries

Run the download script to fetch pre-built binaries for all platforms:

```bash
./download.sh           # Latest pinned version
./download.sh 2026.4 # Specific version
```

## Development Fallback

For development, Goldy will automatically fall back to using Slang from:

1. `RAG_SLANG_PATH` environment variable
2. Vendored binaries in this directory
3. Vulkan SDK (Windows: `C:\VulkanSDK\*\Bin\slang.dll`)

## Version

Current pinned version: **2026.4**

## License

Slang is licensed under Apache 2.0 with LLVM exception.
See https://github.com/shader-slang/slang/blob/master/LICENSE

