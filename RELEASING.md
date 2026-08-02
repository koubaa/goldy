# Releasing Goldy

This guide covers the release process for all Goldy packages across different languages and package managers.

## Overview

Goldy is published to multiple package registries:

| Language | Package Manager | Registry | Automated |
|----------|-----------------|----------|-----------|
| Rust | Cargo | [crates.io](https://crates.io/crates/goldy) | ❌ Manual |
| Python | pip | [PyPI](https://pypi.org/project/goldy/) | ✅ On GitHub release |
| C++ | vcpkg | [microsoft/vcpkg](https://github.com/microsoft/vcpkg) | ❌ Manual PR |
| C++ | Conan | [conan-center-index](https://github.com/conan-io/conan-center-index) | ❌ Manual PR |
| .NET | NuGet | [nuget.org](https://www.nuget.org/packages/Goldy/) | ✅ On push to main |

## Release Checklist

### 1. Prepare the Release

- [ ] Update version numbers in all relevant files
- [ ] Update CHANGELOG.md
- [ ] Ensure all tests pass
- [ ] Merge to main branch

### 2. Create GitHub Release

```bash
git tag v0.X.0
git push origin v0.X.0
```

Then create a release on GitHub. The `release-cpp.yml` workflow will automatically:
- Build native binaries for all platforms (Windows, Linux, macOS)
- Attach binaries to the release

### 3. Publish to Each Registry

Follow the sections below for each package manager.

---

## Rust (crates.io)

### Prerequisites
- Logged into crates.io: `cargo login`

### Publish

Publish **`goldy_shader_ir` first**, then **`goldy_derive`**, then **`goldy`**.
Keep their `version` fields in sync in `shader_ir/Cargo.toml`, `derive/Cargo.toml`,
and the root `Cargo.toml`.

```bash
# 1. Shared KernelAbi / shader IR
cargo publish -p goldy_shader_ir

# 2. Proc-macro crate (depends on goldy_shader_ir)
cargo publish -p goldy_derive

# 3. Core library (depends on goldy_derive + goldy_shader_ir on crates.io)
cargo publish -p goldy

# 4. FFI library — not published to crates.io (`publish = false`);
#    distribute via GitHub release / vcpkg / Conan / language bindings
```

The root `Cargo.toml` lists `goldy_derive` / `goldy_shader_ir` with both `path` and `version`
so local builds use the workspace crates and `cargo publish` resolves dependencies from the registry.

### Version Files
- `Cargo.toml` (goldy) - Update `version = "X.Y.Z"` and bump `goldy_derive` / `goldy_shader_ir` dependency versions to match
- `derive/Cargo.toml` / `shader_ir/Cargo.toml` - Same `version = "X.Y.Z"` as goldy for releases

---

## Python (PyPI)

### Automated Publishing

The `publish.yml` workflow builds wheels for Linux (x86_64, aarch64), Windows,
macOS (x86_64, aarch64), and sdist, then uploads to PyPI when a GitHub release
is published. Authentication uses PyPI Trusted Publishers (no API token in CI).

Wheels embed Slang (~100–160 MB on some platforms). PyPI file-size limit for
`goldy` is **175 MB** ([limit request #11613](https://github.com/pypi/support/issues/11613)).

### Republishing missing wheels

If a release upload partially failed (for example before the size limit was
raised), re-run from Actions:

1. **Publish to PyPI** → **Run workflow**
2. Check **Upload built wheels to PyPI**
3. Run workflow (builds all platforms, uploads with `skip-existing`)

### Manual publish (single platform)

```bash
cd python
python build-slang.py
maturin build --release
maturin publish
```

### Version Files
- `python/Cargo.toml` - Update `version = "X.Y.Z"`
- `python/pyproject.toml` - Update `version = "X.Y.Z"`

---

## C++ (vcpkg)

vcpkg requires a **manual PR** to [microsoft/vcpkg](https://github.com/microsoft/vcpkg) for each new version.

### Prerequisites
- Fork of microsoft/vcpkg
- vcpkg CLI installed

### Process

1. **Wait for GitHub Release** to complete (binaries must be uploaded)

2. **Compute SHA512 hashes** for all release assets:
   ```bash
   curl -L -o source.tar.gz "https://github.com/koubaa/goldy/archive/refs/tags/vX.Y.Z.tar.gz"
   sha512sum source.tar.gz
   
   # Repeat for each binary
   curl -L -o linux.tar.gz "https://github.com/koubaa/goldy/releases/download/vX.Y.Z/goldy_ffi-linux-x64.tar.gz"
   sha512sum linux.tar.gz
   ```

3. **Update port files** in your vcpkg fork:
   ```bash
   cd /path/to/vcpkg
   git checkout -b update-goldy-vX.Y.Z
   
   # Edit ports/goldy/vcpkg.json - update version
   # Edit ports/goldy/portfile.cmake - update SHA512 hashes
   
   # Update version database
   ./vcpkg x-add-version goldy --overwrite-version
   ```

4. **Submit PR** to microsoft/vcpkg:
   ```bash
   git add ports/goldy versions/
   git commit -m "[goldy] Update to version X.Y.Z"
   git push origin update-goldy-vX.Y.Z
   
   gh pr create --repo microsoft/vcpkg --title "[goldy] Update to version X.Y.Z"
   ```

### Version Files
- `cpp/vcpkg/vcpkg.json` - Update `"version": "X.Y.Z"`
- `cpp/vcpkg/portfile.cmake` - Update SHA512 hashes

---

## C++ (Conan Center)

Conan Center requires a **manual PR** to [conan-center-index](https://github.com/conan-io/conan-center-index) for each new version.

### Prerequisites
- Fork of conan-io/conan-center-index
- Conan 2.x installed: `pip install conan`

### Process

1. **Wait for GitHub Release** to complete

2. **Compute SHA256 hashes** for release assets:
   ```bash
   curl -L -o source.tar.gz "https://github.com/koubaa/goldy/archive/refs/tags/vX.Y.Z.tar.gz"
   sha256sum source.tar.gz
   
   # Repeat for each binary
   ```

3. **Update recipe** in your conan-center-index fork:
   ```bash
   cd /path/to/conan-center-index
   git checkout -b update-goldy-vX.Y.Z
   ```

4. **Edit `recipes/goldy/config.yml`** - add new version:
   ```yaml
   versions:
     "0.2.0":
       folder: all
     "0.1.0":
       folder: all
   ```

5. **Edit `recipes/goldy/all/conandata.yml`** - add new version entry with URLs and hashes

6. **Test locally**:
   ```bash
   cd recipes/goldy/all
   conan create . --version=X.Y.Z
   ```

7. **Submit PR**:
   ```bash
   git add recipes/goldy
   git commit -m "[goldy] Update to version X.Y.Z"
   git push origin update-goldy-vX.Y.Z
   
   gh pr create --repo conan-io/conan-center-index --title "[goldy] Update to version X.Y.Z"
   ```

### Version Files
- `cpp/conan/conanfile.py` - Update `version = "X.Y.Z"` (for local testing)
- conan-center-index: `config.yml`, `conandata.yml`

---

## .NET (NuGet)

NuGet publishing is **automated** via GitHub Actions on push to main.

### Automated Publishing
The `dotnet.yml` workflow automatically publishes when:
- Changes are pushed to `main` branch
- `NUGET_API_KEY` secret is configured

### Manual Publishing (if needed)

```bash
cd dotnet/Goldy

# Update version in Goldy.csproj
dotnet pack --configuration Release

dotnet nuget push bin/Release/Goldy.*.nupkg \
  --api-key $NUGET_API_KEY \
  --source https://api.nuget.org/v3/index.json
```

### Version Files
- `dotnet/Goldy/Goldy.csproj` - Update `<Version>X.Y.Z</Version>`

See [dotnet/PUBLISHING.md](dotnet/PUBLISHING.md) for detailed instructions.

---

## Slang Bundling

All packages bundle the Slang shader compiler libraries automatically. See [PACKAGING.md](PACKAGING.md) for details on:

- How the manifest-driven bundling works
- Platform-specific file lists
- How to update Slang version

## Version Synchronization

When releasing a new version, update these files:

| File | Field |
|------|-------|
| `Cargo.toml` | `version` (+ `goldy_derive` / `goldy_shader_ir` dependency versions) |
| `derive/Cargo.toml` | `version` (must match goldy) |
| `shader_ir/Cargo.toml` | `version` (must match goldy) |
| `ffi/Cargo.toml` | `version` (`publish = false`) |
| `ffi-client/Cargo.toml` | `version` (`publish = false`) |
| `python/Cargo.toml` | `version` (`publish = false`) |
| `python/pyproject.toml` | `version` |
| `python/python/goldy/__init__.py` | `__version__` |
| `cpp/vcpkg/vcpkg.json` | `version` |
| `cpp/conan/conanfile.py` | `version` (+ release asset URLs/hashes after GitHub release) |
| `dotnet/Goldy/Goldy.csproj` | `<Version>` |
| `CHANGELOG.md` | Add release notes |

crates.io publish order: **`goldy_shader_ir`**, then **`goldy_derive`**, then **`goldy`**.
Packaging resolves those crates from the registry, so each dependency must already be live.

---

## Timeline

Typical release timeline:

1. **Day 1**: Create GitHub release with tag
   - CI builds and uploads C++ binaries (~10 min)
   
2. **Day 1**: Publish Rust and Python
   - `cargo publish` (~5 min)
   - `maturin publish` (~10 min per platform)
   
3. **Day 1**: .NET auto-publishes on merge to main

4. **Day 1-2**: Submit vcpkg and Conan PRs
   - Compute hashes, update files, submit PRs
   
5. **Day 3-14**: vcpkg/Conan PRs reviewed and merged
   - May require addressing reviewer feedback
