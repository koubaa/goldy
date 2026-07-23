# Goldy vcpkg portfile
# Downloads pre-built native library (C ABI) and installs C++ headers
#
# Architecture:
#   - goldy.hpp: Header-only C++ wrapper (built from source)
#   - goldy_ffi: Pre-built native library with stable C ABI
#
# This is the standard approach for C++ packages wrapping non-C++ libraries
# (similar to CUDA, Vulkan SDK, Intel MKL, etc.)

vcpkg_check_linkage(ONLY_DYNAMIC_LIBRARY)

# Download source for headers and license
vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO koubaa/goldy
    REF "v${VERSION}"
    SHA512 b2ffdc144d7091a02b9ced0ae423bdc298811c1ebd93d8cde980ad5a661528b3e96106676a4508fe37cc7bfcb2306b00982343f2adef79e9641072e308af30c2
    HEAD_REF main
)

# Download pre-built native library for target platform
if(VCPKG_TARGET_IS_WINDOWS AND VCPKG_TARGET_ARCHITECTURE STREQUAL "x64")
    vcpkg_download_distfile(GOLDY_FFI_ARCHIVE
        URLS "https://github.com/koubaa/goldy/releases/download/v${VERSION}/goldy_ffi-windows-x64.zip"
        FILENAME "goldy_ffi-${VERSION}-windows-x64.zip"
        SHA512 29d8c54092a71b016079ee9c285c55c9edfe70b301d374cac8bd88166bfa56193d3cfef31ceef184d6a907f887ef25ca097fa0cdcea89696192fa23369c48f50
    )
    set(ARCHIVE_TYPE ZIP)
elseif(VCPKG_TARGET_IS_LINUX AND VCPKG_TARGET_ARCHITECTURE STREQUAL "x64")
    vcpkg_download_distfile(GOLDY_FFI_ARCHIVE
        URLS "https://github.com/koubaa/goldy/releases/download/v${VERSION}/goldy_ffi-linux-x64.tar.gz"
        FILENAME "goldy_ffi-${VERSION}-linux-x64.tar.gz"
        SHA512 72fd5a8c87892e00a249cbdd13ea4a070bd2d79b49b340fff59fe42bb0202cbff7fc70ae30e4e7485b1b12101171577d77727bd7dc3c1326d32fffc318c87914
    )
    set(ARCHIVE_TYPE TARGZ)
elseif(VCPKG_TARGET_IS_OSX AND VCPKG_TARGET_ARCHITECTURE STREQUAL "x64")
    vcpkg_download_distfile(GOLDY_FFI_ARCHIVE
        URLS "https://github.com/koubaa/goldy/releases/download/v${VERSION}/goldy_ffi-macos-x64.tar.gz"
        FILENAME "goldy_ffi-${VERSION}-macos-x64.tar.gz"
        SHA512 1a7ea0c3a2fa18320d7e7c79da928fe6d25da6879a72f64e926541335c62f09180020fb7516ee0871ff94fc7e9b53f9ea81bf8809bb3a835178e38135c6832ec
    )
    set(ARCHIVE_TYPE TARGZ)
elseif(VCPKG_TARGET_IS_OSX AND VCPKG_TARGET_ARCHITECTURE STREQUAL "arm64")
    vcpkg_download_distfile(GOLDY_FFI_ARCHIVE
        URLS "https://github.com/koubaa/goldy/releases/download/v${VERSION}/goldy_ffi-macos-arm64.tar.gz"
        FILENAME "goldy_ffi-${VERSION}-macos-arm64.tar.gz"
        SHA512 e0efae0e8e26e2447e7549bc0506bf3efaf206936767c8c893bf1847fde79276de9768a6bee906d4ab06b30312abd5f721a774e6ead434aeec52dd1f0be4ee84
    )
    set(ARCHIVE_TYPE TARGZ)
else()
    message(FATAL_ERROR "Unsupported platform: ${VCPKG_TARGET_TRIPLET}")
endif()

# Extract the pre-built library archive
vcpkg_extract_source_archive(
    BINARY_PATH
    ARCHIVE "${GOLDY_FFI_ARCHIVE}"
    NO_REMOVE_ONE_LEVEL
)

# Install headers from source
file(INSTALL "${SOURCE_PATH}/cpp/include/goldy.h" DESTINATION "${CURRENT_PACKAGES_DIR}/include")
file(INSTALL "${SOURCE_PATH}/cpp/include/goldy.hpp" DESTINATION "${CURRENT_PACKAGES_DIR}/include")

# Determine Slang platform directory
if(VCPKG_TARGET_IS_WINDOWS AND VCPKG_TARGET_ARCHITECTURE STREQUAL "x64")
    set(SLANG_PLATFORM "windows-x86_64")
elseif(VCPKG_TARGET_IS_LINUX AND VCPKG_TARGET_ARCHITECTURE STREQUAL "x64")
    set(SLANG_PLATFORM "linux-x86_64")
elseif(VCPKG_TARGET_IS_LINUX AND VCPKG_TARGET_ARCHITECTURE STREQUAL "arm64")
    set(SLANG_PLATFORM "linux-aarch64")
elseif(VCPKG_TARGET_IS_OSX AND VCPKG_TARGET_ARCHITECTURE STREQUAL "x64")
    set(SLANG_PLATFORM "macos-x86_64")
elseif(VCPKG_TARGET_IS_OSX AND VCPKG_TARGET_ARCHITECTURE STREQUAL "arm64")
    set(SLANG_PLATFORM "macos-aarch64")
endif()

set(SLANG_BIN_DIR "${SOURCE_PATH}/slang/bin/${SLANG_PLATFORM}")

# Install native library from pre-built archive
if(VCPKG_TARGET_IS_WINDOWS)
    file(INSTALL "${BINARY_PATH}/lib/goldy_ffi.dll"
         DESTINATION "${CURRENT_PACKAGES_DIR}/bin")
    file(INSTALL "${BINARY_PATH}/lib/goldy_ffi.dll.lib"
         DESTINATION "${CURRENT_PACKAGES_DIR}/lib"
         RENAME "goldy_ffi.lib")
    
    # Install Slang libraries
    if(EXISTS "${SLANG_BIN_DIR}")
        file(GLOB SLANG_DLLS "${SLANG_BIN_DIR}/*.dll")
        file(INSTALL ${SLANG_DLLS} DESTINATION "${CURRENT_PACKAGES_DIR}/bin")
    endif()
    
    # vcpkg expects debug libraries - use release for both since we don't ship debug builds
    file(MAKE_DIRECTORY "${CURRENT_PACKAGES_DIR}/debug/bin")
    file(MAKE_DIRECTORY "${CURRENT_PACKAGES_DIR}/debug/lib")
    file(INSTALL "${BINARY_PATH}/lib/goldy_ffi.dll"
         DESTINATION "${CURRENT_PACKAGES_DIR}/debug/bin")
    file(INSTALL "${BINARY_PATH}/lib/goldy_ffi.dll.lib"
         DESTINATION "${CURRENT_PACKAGES_DIR}/debug/lib"
         RENAME "goldy_ffi.lib")
    if(EXISTS "${SLANG_BIN_DIR}")
        file(INSTALL ${SLANG_DLLS} DESTINATION "${CURRENT_PACKAGES_DIR}/debug/bin")
    endif()
elseif(VCPKG_TARGET_IS_LINUX)
    file(INSTALL "${BINARY_PATH}/lib/libgoldy_ffi.so"
         DESTINATION "${CURRENT_PACKAGES_DIR}/lib")
    
    # Install Slang libraries
    if(EXISTS "${SLANG_BIN_DIR}")
        file(GLOB SLANG_LIBS "${SLANG_BIN_DIR}/*.so")
        file(INSTALL ${SLANG_LIBS} DESTINATION "${CURRENT_PACKAGES_DIR}/lib")
    endif()
    
    file(MAKE_DIRECTORY "${CURRENT_PACKAGES_DIR}/debug/lib")
    file(INSTALL "${BINARY_PATH}/lib/libgoldy_ffi.so"
         DESTINATION "${CURRENT_PACKAGES_DIR}/debug/lib")
    if(EXISTS "${SLANG_BIN_DIR}")
        file(INSTALL ${SLANG_LIBS} DESTINATION "${CURRENT_PACKAGES_DIR}/debug/lib")
    endif()
elseif(VCPKG_TARGET_IS_OSX)
    file(INSTALL "${BINARY_PATH}/lib/libgoldy_ffi.dylib"
         DESTINATION "${CURRENT_PACKAGES_DIR}/lib")
    
    # Install Slang libraries
    if(EXISTS "${SLANG_BIN_DIR}")
        file(GLOB SLANG_LIBS "${SLANG_BIN_DIR}/*.dylib")
        file(INSTALL ${SLANG_LIBS} DESTINATION "${CURRENT_PACKAGES_DIR}/lib")
    endif()
    
    file(MAKE_DIRECTORY "${CURRENT_PACKAGES_DIR}/debug/lib")
    file(INSTALL "${BINARY_PATH}/lib/libgoldy_ffi.dylib"
         DESTINATION "${CURRENT_PACKAGES_DIR}/debug/lib")
    if(EXISTS "${SLANG_BIN_DIR}")
        file(INSTALL ${SLANG_LIBS} DESTINATION "${CURRENT_PACKAGES_DIR}/debug/lib")
    endif()
endif()

# Create CMake config
file(WRITE "${CURRENT_PACKAGES_DIR}/share/${PORT}/goldy-config.cmake" [[
include(CMakeFindDependencyMacro)

if(NOT TARGET goldy::goldy)
    add_library(goldy::goldy INTERFACE IMPORTED)
    
    get_filename_component(_IMPORT_PREFIX "${CMAKE_CURRENT_LIST_DIR}/../../" ABSOLUTE)
    
    set_target_properties(goldy::goldy PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${_IMPORT_PREFIX}/include"
    )
    
    if(WIN32)
        set_target_properties(goldy::goldy PROPERTIES
            INTERFACE_LINK_LIBRARIES "${_IMPORT_PREFIX}/lib/goldy_ffi.lib"
        )
    else()
        find_library(_GOLDY_FFI_LIB goldy_ffi PATHS "${_IMPORT_PREFIX}/lib" NO_DEFAULT_PATH)
        set_target_properties(goldy::goldy PROPERTIES
            INTERFACE_LINK_LIBRARIES "${_GOLDY_FFI_LIB}"
        )
    endif()
endif()
]])

# Handle copyright
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")

# Configure usage
file(WRITE "${CURRENT_PACKAGES_DIR}/share/${PORT}/usage" [[
The package goldy provides CMake targets:

    find_package(goldy CONFIG REQUIRED)
    target_link_libraries(main PRIVATE goldy::goldy)

Include the headers:
    #include <goldy.hpp>  // C++ RAII wrapper
    #include <goldy.h>    // C API

Note: goldy_ffi.dll and Slang libraries (slang-compiler.dll, etc.) must be
in your PATH or next to your executable at runtime. Both are included in
the vcpkg bin directory.
]])
