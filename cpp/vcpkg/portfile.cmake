# Goldy vcpkg portfile
# This portfile builds goldy-ffi from source using Cargo and installs the C++ headers

vcpkg_check_linkage(ONLY_DYNAMIC_LIBRARY)

# Find cargo
find_program(CARGO cargo REQUIRED)
if(NOT CARGO)
    message(FATAL_ERROR "Cargo is required to build goldy. Install Rust from https://rustup.rs")
endif()

# Download source
vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO koubaa/goldy
    REF "v${VERSION}"
    SHA512 0  # Update with actual SHA after first release
    HEAD_REF main
)

# Build the Rust FFI library
message(STATUS "Building goldy-ffi with Cargo...")

vcpkg_execute_required_process(
    COMMAND ${CARGO} build --package goldy-ffi --release
    WORKING_DIRECTORY "${SOURCE_PATH}"
    LOGNAME build-goldy-ffi
)

# Install headers
file(INSTALL "${SOURCE_PATH}/cpp/include/goldy.h" DESTINATION "${CURRENT_PACKAGES_DIR}/include")
file(INSTALL "${SOURCE_PATH}/cpp/include/goldy.hpp" DESTINATION "${CURRENT_PACKAGES_DIR}/include")

# Install libraries
if(VCPKG_TARGET_IS_WINDOWS)
    file(INSTALL "${SOURCE_PATH}/target/release/goldy_ffi.dll"
         DESTINATION "${CURRENT_PACKAGES_DIR}/bin")
    file(INSTALL "${SOURCE_PATH}/target/release/goldy_ffi.dll.lib"
         DESTINATION "${CURRENT_PACKAGES_DIR}/lib"
         RENAME "goldy_ffi.lib")
    
    # Debug builds
    vcpkg_execute_required_process(
        COMMAND ${CARGO} build --package goldy-ffi
        WORKING_DIRECTORY "${SOURCE_PATH}"
        LOGNAME build-goldy-ffi-debug
    )
    
    file(INSTALL "${SOURCE_PATH}/target/debug/goldy_ffi.dll"
         DESTINATION "${CURRENT_PACKAGES_DIR}/debug/bin")
    file(INSTALL "${SOURCE_PATH}/target/debug/goldy_ffi.dll.lib"
         DESTINATION "${CURRENT_PACKAGES_DIR}/debug/lib"
         RENAME "goldy_ffi.lib")
elseif(VCPKG_TARGET_IS_LINUX)
    file(INSTALL "${SOURCE_PATH}/target/release/libgoldy_ffi.so"
         DESTINATION "${CURRENT_PACKAGES_DIR}/lib")
    
    vcpkg_execute_required_process(
        COMMAND ${CARGO} build --package goldy-ffi
        WORKING_DIRECTORY "${SOURCE_PATH}"
        LOGNAME build-goldy-ffi-debug
    )
    
    file(INSTALL "${SOURCE_PATH}/target/debug/libgoldy_ffi.so"
         DESTINATION "${CURRENT_PACKAGES_DIR}/debug/lib")
elseif(VCPKG_TARGET_IS_OSX)
    file(INSTALL "${SOURCE_PATH}/target/release/libgoldy_ffi.dylib"
         DESTINATION "${CURRENT_PACKAGES_DIR}/lib")
    
    vcpkg_execute_required_process(
        COMMAND ${CARGO} build --package goldy-ffi
        WORKING_DIRECTORY "${SOURCE_PATH}"
        LOGNAME build-goldy-ffi-debug
    )
    
    file(INSTALL "${SOURCE_PATH}/target/debug/libgoldy_ffi.dylib"
         DESTINATION "${CURRENT_PACKAGES_DIR}/debug/lib")
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

Note: goldy_ffi.dll must be in your PATH or next to your executable at runtime.
The Slang compiler (slang.dll) is also required at runtime (from Vulkan SDK).
]])

