# Vendored Slang Binaries

This directory contains vendored [Slang](https://github.com/shader-slang/slang) shader compiler binaries.

## Structure

```
slang/
├── bin/
│   ├── windows-x86_64/
│   │   └── slang-compiler.dll
│   ├── linux-x86_64/
│   │   └── libslang-compiler.so
│   ├── macos-aarch64/
│   │   └── libslang-compiler.dylib
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
./download.sh 2025.24.3 # Specific version
```

## Development Fallback

For development, RAG will automatically fall back to using Slang from:

1. `RAG_SLANG_PATH` environment variable
2. Vendored binaries in this directory
3. Vulkan SDK (Windows: `C:\VulkanSDK\*\Bin\slang.dll`)

## Version

Current pinned version: **2025.24.3**

## License

Slang is licensed under Apache 2.0 with LLVM exception.
See https://github.com/shader-slang/slang/blob/master/LICENSE

