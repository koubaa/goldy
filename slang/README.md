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

## Who runs `download.sh`?

**Application developers do not.** `cargo build` / `pip install -e` runs `goldy/build.rs`,
which downloads the pinned Slang version into `bin/{platform}/` when missing, then
embeds those bytes into the Goldy library.

Run `download.sh` when **bumping the pinned Slang version** or preparing **release
artifacts** (CI, wheels, FFI packages) that copy DLLs next to shipped binaries:

```bash
./download.sh              # version from manifest.json
./download.sh 2026.13      # explicit version
```

## How is slang.h used

It is actually unused. It is just a local reference for the ffi implementation.

## Runtime loading

At runtime Goldy loads Slang dynamically (search order in `goldy/src/slang/loader.rs`):

1. `GOLDY_SLANG_PATH` if set
2. Slang DLLs next to the running executable (wheel / FFI layout)
3. Cache directory — extracted from bytes embedded at compile time

### iOS

Official Slang releases have no iOS zip. `build.rs` maps `aarch64-apple-ios` to
`ios-aarch64` and does **not** embed macOS dylibs.

Cross-compile on macOS (Xcode + CMake + Ninja):

```bash
./build-ios.sh                  # writes bin/ios-aarch64/
./build-ios.sh /tmp/slang-ios   # custom dest
```

Koba Screen copies those dylibs into `KobaScreen.app/Frameworks` and codesigns
them. At runtime Goldy loads `@executable_path/Frameworks/libslang-compiler.dylib`.

## Version

Current pinned version: **2026.13**

## License

Slang is licensed under Apache 2.0 with LLVM exception.
See https://github.com/shader-slang/slang/blob/master/LICENSE

