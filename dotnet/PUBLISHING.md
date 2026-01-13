# Publishing Goldy to NuGet

## Automated Publishing (CI)

The GitHub Actions workflow automatically builds and publishes to NuGet when:
1. Changes are pushed to `main` branch
2. The `NUGET_API_KEY` secret is configured in the repository

### Setting up the NuGet API Key

1. Go to [nuget.org](https://www.nuget.org/) and sign in
2. Go to your account → API Keys
3. Create a new API key with "Push new packages and package versions" scope
4. Add the key as a repository secret named `NUGET_API_KEY`

## Manual Publishing

### Prerequisites

- .NET 8.0 SDK
- Rust toolchain with cross-compilation targets
- Native libraries built for all platforms

### Build Native Libraries

On Windows:
```powershell
.\dotnet\build-native.ps1
```

On Linux/macOS:
```bash
./dotnet/build-native.sh
```

### Cross-compile for other platforms

```bash
# Add targets
rustup target add x86_64-pc-windows-msvc
rustup target add x86_64-unknown-linux-gnu
rustup target add x86_64-apple-darwin
rustup target add aarch64-apple-darwin

# Build each (may require cross-compilation setup)
cargo build --release -p goldy-ffi --target x86_64-pc-windows-msvc
cargo build --release -p goldy-ffi --target x86_64-unknown-linux-gnu
cargo build --release -p goldy-ffi --target x86_64-apple-darwin
cargo build --release -p goldy-ffi --target aarch64-apple-darwin
```

### Pack and Publish

```bash
cd dotnet/Goldy

# Update version in Goldy.csproj first

# Pack
dotnet pack --configuration Release

# Publish (requires API key)
dotnet nuget push bin/Release/Goldy.*.nupkg \
  --api-key YOUR_API_KEY \
  --source https://api.nuget.org/v3/index.json
```

## Package Structure

The NuGet package contains:
```
Goldy.nupkg
├── lib/
│   └── net8.0/
│       └── Goldy.dll
├── runtimes/
│   ├── win-x64/native/goldy_ffi.dll
│   ├── linux-x64/native/libgoldy_ffi.so
│   ├── osx-x64/native/libgoldy_ffi.dylib
│   └── osx-arm64/native/libgoldy_ffi.dylib
└── README.md
```

## Version Management

Update the version in `dotnet/Goldy/Goldy.csproj`:

```xml
<Version>0.1.0</Version>
```

Follow [Semantic Versioning](https://semver.org/):
- MAJOR: Breaking API changes
- MINOR: New features, backward compatible
- PATCH: Bug fixes, backward compatible

