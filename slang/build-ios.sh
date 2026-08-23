#!/usr/bin/env bash
# Cross-compile Slang shared libraries for iphoneos arm64.
# Official shader-slang releases do not ship iOS zips.
#
# Usage: ./build-ios.sh [dest-dir]
# Default dest: <this-dir>/bin/ios-aarch64
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="${SCRIPT_DIR}/manifest.json"
VERSION="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['version'])" "${MANIFEST}")"
DEST="${1:-${SCRIPT_DIR}/bin/ios-aarch64}"
TAG="v${VERSION}"
WORK="${SLANG_IOS_WORK_DIR:-${TMPDIR:-/tmp}/goldy-slang-ios-${VERSION}}"
SRC="${WORK}/slang"
HOST_BUILD="${WORK}/build-host"
HOST_GEN="${WORK}/generators"
IOS_BUILD="${WORK}/build-ios"
IOS_SDK="$(xcrun --sdk iphoneos --show-sdk-path)"
DEPLOY="${IPHONEOS_DEPLOYMENT_TARGET:-16.0}"

if [[ -f "${DEST}/libslang-compiler.dylib" ]]; then
  echo "iOS Slang already present at ${DEST}"
  ls -la "${DEST}"
  exit 0
fi

if ! command -v cmake >/dev/null 2>&1; then
  echo "ERROR: cmake is required" >&2
  exit 1
fi
if ! command -v ninja >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then
    brew install ninja
  else
    echo "ERROR: ninja is required" >&2
    exit 1
  fi
fi

mkdir -p "${WORK}" "${DEST}"

if [[ ! -d "${SRC}/.git" ]]; then
  echo "Cloning shader-slang/slang ${TAG}..."
  git clone --depth 1 --recurse-submodules --shallow-submodules \
    --branch "${TAG}" \
    https://github.com/shader-slang/slang.git "${SRC}"
fi

COMMON_FLAGS=(
  -DCMAKE_BUILD_TYPE=Release
  -DSLANG_ENABLE_GFX=OFF
  -DSLANG_ENABLE_SLANG_RHI=OFF
  -DSLANG_ENABLE_SLANGRT=OFF
  -DSLANG_ENABLE_EXAMPLES=OFF
  -DSLANG_ENABLE_TESTS=OFF
  -DSLANG_ENABLE_SLANGD=OFF
  -DSLANG_ENABLE_SLANGI=OFF
  -DSLANG_ENABLE_REPLAYER=OFF
  -DSLANG_ENABLE_PREBUILT_BINARIES=OFF
  -DSLANG_SLANG_LLVM_FLAVOR=DISABLE
)

# CI iOS jobs export IPHONEOS_DEPLOYMENT_TARGET, which makes host clang
# target iPhone (Lua's system() is then unavailable). Generators must be macOS.
HOST_ENV=(env -u IPHONEOS_DEPLOYMENT_TARGET -u SDKROOT)

echo "Building host generators..."
"${HOST_ENV[@]}" cmake -S "${SRC}" -B "${HOST_BUILD}" -G Ninja \
  "${COMMON_FLAGS[@]}" \
  -DCMAKE_OSX_SYSROOT=macosx \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=13.0 \
  -DSLANG_LIB_TYPE=SHARED
"${HOST_ENV[@]}" cmake --build "${HOST_BUILD}" --target all-generators
"${HOST_ENV[@]}" cmake --install "${HOST_BUILD}" --prefix "${HOST_GEN}" --component generators

GEN_BIN="${HOST_GEN}/bin"
if [[ ! -d "${GEN_BIN}" ]]; then
  GEN_BIN="${HOST_GEN}"
fi

echo "Building Slang for iOS arm64 (sdk ${IOS_SDK})..."
cmake -S "${SRC}" -B "${IOS_BUILD}" -G Ninja \
  "${COMMON_FLAGS[@]}" \
  -DCMAKE_SYSTEM_NAME=iOS \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="${DEPLOY}" \
  -DCMAKE_OSX_SYSROOT="${IOS_SDK}" \
  -DCMAKE_INSTALL_NAME_DIR=@rpath \
  -DCMAKE_BUILD_WITH_INSTALL_NAME_DIR=ON \
  -DSLANG_LIB_TYPE=SHARED \
  -DSLANG_ENABLE_SLANGC=OFF \
  -DSLANG_GENERATORS_PATH="${GEN_BIN}"
cmake --build "${IOS_BUILD}" --target slang

shopt -s nullglob
copied=0
for dir in "${IOS_BUILD}/lib" "${IOS_BUILD}" "${IOS_BUILD}/Release" "${IOS_BUILD}/lib/Release"; do
  [[ -d "${dir}" ]] || continue
  for lib in "${dir}"/libslang*.dylib; do
    cp -L "${lib}" "${DEST}/"
    copied=1
  done
done
shopt -u nullglob

if [[ ! -f "${DEST}/libslang-compiler.dylib" ]]; then
  echo "ERROR: libslang-compiler.dylib not produced. Build tree:" >&2
  find "${IOS_BUILD}" -name 'libslang*' | head -50 >&2
  exit 1
fi

fix_id() {
  local lib="$1"
  local base
  base="$(basename "${lib}")"
  install_name_tool -id "@rpath/${base}" "${lib}" 2>/dev/null || true
  while IFS= read -r dep; do
    local depbase
    depbase="$(basename "${dep}")"
    if [[ -f "${DEST}/${depbase}" && "${dep}" != @rpath/* ]]; then
      install_name_tool -change "${dep}" "@rpath/${depbase}" "${lib}" 2>/dev/null || true
    fi
  done < <(otool -L "${lib}" | awk 'NR>1 {print $1}')
}

for lib in "${DEST}"/*.dylib; do
  fix_id "${lib}"
done

echo "iOS Slang ${VERSION} -> ${DEST}"
ls -la "${DEST}"
file "${DEST}/libslang-compiler.dylib"
otool -l "${DEST}/libslang-compiler.dylib" | awk '/LC_BUILD_VERSION/,/platform/{print}' | head -20
