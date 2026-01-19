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
    SHA512 8ff9ac74d796cc5ac4660232cf55edda9f848aca4fd565e59d4e6a90c3d2c1ced444b96aa4f2ab4cfe300049eda5aaa275238ba4e7c341771ef3c8b723df79a0
    HEAD_REF main
)

# Download pre-built native library for target platform
if(VCPKG_TARGET_IS_WINDOWS AND VCPKG_TARGET_ARCHITECTURE STREQUAL "x64")
    vcpkg_download_distfile(GOLDY_FFI_ARCHIVE
        URLS "https://github.com/koubaa/goldy/releases/download/v${VERSION}/goldy_ffi-windows-x64.zip"
        FILENAME "goldy_ffi-${VERSION}-windows-x64.zip"
        SHA512 15142e06536046d4f2768c95256471efb8c0cb1b52a905f34aaab9636d5f98139b7a038afeace680879298695dfe952061a404eb7da5ef6999f65b8317455ef9
    )
    set(ARCHIVE_TYPE ZIP)
elseif(VCPKG_TARGET_IS_LINUX AND VCPKG_TARGET_ARCHITECTURE STREQUAL "x64")
    vcpkg_download_distfile(GOLDY_FFI_ARCHIVE
        URLS "https://github.com/koubaa/goldy/releases/download/v${VERSION}/goldy_ffi-linux-x64.tar.gz"
        FILENAME "goldy_ffi-${VERSION}-linux-x64.tar.gz"
        SHA512 ebc70ffdc0895ed8755a5e475d0e06e91114998e0dab1a6a2db4f909a1b606a3b6c150e9325a23ff498a3d82c67ff7e433fe5524399bbbb4e4308f2969ac527f
    )
    set(ARCHIVE_TYPE TARGZ)
elseif(VCPKG_TARGET_IS_OSX AND VCPKG_TARGET_ARCHITECTURE STREQUAL "x64")
    vcpkg_download_distfile(GOLDY_FFI_ARCHIVE
        URLS "https://github.com/koubaa/goldy/releases/download/v${VERSION}/goldy_ffi-macos-x64.tar.gz"
        FILENAME "goldy_ffi-${VERSION}-macos-x64.tar.gz"
        SHA512 416ad1957f96fb7a9e6a5a0711ae58e75c0658f8f84014797b041b482698b4780fd422ffcd515e46796395a69400ece2487ff2225b7df729959c87b05da826e1
    )
    set(ARCHIVE_TYPE TARGZ)
elseif(VCPKG_TARGET_IS_OSX AND VCPKG_TARGET_ARCHITECTURE STREQUAL "arm64")
    vcpkg_download_distfile(GOLDY_FFI_ARCHIVE
        URLS "https://github.com/koubaa/goldy/releases/download/v${VERSION}/goldy_ffi-macos-arm64.tar.gz"
        FILENAME "goldy_ffi-${VERSION}-macos-arm64.tar.gz"
        SHA512 7451fb6cbec47f869c295db480a2d9f32a064e0602e40cc8b9742543e78fccf35fae542e24d883b0ea632d803cc2789a120dfe880ca84e806b5f0091edf5007a
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
