cmake_minimum_required(VERSION 3.20)

project(goldy
    VERSION 0.1.0
    DESCRIPTION "Goldy - Modern GPU Library"
    HOMEPAGE_URL "https://github.com/koubaa/goldy"
    LANGUAGES CXX
)

if(APPLE)
    enable_language(OBJCXX)
endif()

# Ninja + MSVC from non-Developer shells (Git Bash, IDE terminals) need explicit
# Windows SDK / MSVC include and link paths.
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/InitMSVCEnvironment.cmake")

# Options
option(GOLDY_BUILD_EXAMPLES "Build examples" ON)
option(GOLDY_BUILD_FROM_SOURCE "Build goldy-ffi from Rust source" OFF)

# C++20 required for std::span, std::string_view
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =============================================================================
# Find or build the native library
# =============================================================================

if(GOLDY_BUILD_FROM_SOURCE)
    # Build the Rust FFI library from source
    find_program(CARGO cargo REQUIRED)
    
    set(RUST_TARGET_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../target")
    
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(RUST_PROFILE "debug")
    else()
        set(RUST_PROFILE "release")
    endif()
    
    set(GOLDY_FFI_LIB_DIR "${RUST_TARGET_DIR}/${RUST_PROFILE}")
    
    if(WIN32)
        set(GOLDY_FFI_LIB "${GOLDY_FFI_LIB_DIR}/goldy_ffi.dll")
        set(GOLDY_FFI_IMPLIB "${GOLDY_FFI_LIB_DIR}/goldy_ffi.dll.lib")
    elseif(APPLE)
        set(GOLDY_FFI_LIB "${GOLDY_FFI_LIB_DIR}/libgoldy_ffi.dylib")
    else()
        set(GOLDY_FFI_LIB "${GOLDY_FFI_LIB_DIR}/libgoldy_ffi.so")
    endif()
    
    # Custom command to build Rust library
    add_custom_command(
        OUTPUT ${GOLDY_FFI_LIB}
        COMMAND ${CARGO} build --package goldy-ffi $<$<NOT:$<CONFIG:Debug>>:--release>
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
        COMMENT "Building goldy-ffi with Cargo"
        VERBATIM
    )
    
    add_custom_target(goldy_ffi_build DEPENDS ${GOLDY_FFI_LIB})
    
else()
    # Use pre-built library
    # Look for the library in standard locations
    find_library(GOLDY_FFI_LIB
        NAMES goldy_ffi
        PATHS
            "${CMAKE_CURRENT_SOURCE_DIR}/../target/release"
            "${CMAKE_CURRENT_SOURCE_DIR}/lib"
            "${CMAKE_INSTALL_PREFIX}/lib"
    )
    
    if(WIN32)
        find_file(GOLDY_FFI_DLL
            NAMES goldy_ffi.dll
            PATHS
                "${CMAKE_CURRENT_SOURCE_DIR}/../target/release"
                "${CMAKE_CURRENT_SOURCE_DIR}/bin"
                "${CMAKE_INSTALL_PREFIX}/bin"
        )
    endif()
    
    if(NOT GOLDY_FFI_LIB)
        message(FATAL_ERROR 
            "Could not find goldy_ffi library. "
            "Either set GOLDY_BUILD_FROM_SOURCE=ON or provide the pre-built library.")
    endif()
endif()

# =============================================================================
# Main library target (header-only wrapper + native library)
# =============================================================================

add_library(goldy INTERFACE)
add_library(goldy::goldy ALIAS goldy)

target_include_directories(goldy
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_features(goldy INTERFACE cxx_std_20)

# Link to the native library
if(GOLDY_BUILD_FROM_SOURCE)
    add_dependencies(goldy goldy_ffi_build)
    if(WIN32)
        target_link_libraries(goldy INTERFACE "${GOLDY_FFI_IMPLIB}")
    else()
        target_link_libraries(goldy INTERFACE "${GOLDY_FFI_LIB}")
    endif()
else()
    target_link_libraries(goldy INTERFACE ${GOLDY_FFI_LIB})
endif()

# =============================================================================
# Find Slang libraries
# =============================================================================

# Determine platform for Slang binaries
if(WIN32)
    set(SLANG_PLATFORM "windows-x86_64")
    set(SLANG_LIB_PATTERN "*.dll")
elseif(APPLE)
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
        set(SLANG_PLATFORM "macos-aarch64")
    else()
        set(SLANG_PLATFORM "macos-x86_64")
    endif()
    set(SLANG_LIB_PATTERN "*.dylib")
else()
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
        set(SLANG_PLATFORM "linux-aarch64")
    else()
        set(SLANG_PLATFORM "linux-x86_64")
    endif()
    set(SLANG_LIB_PATTERN "*.so")
endif()

set(SLANG_BIN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../slang/bin/${SLANG_PLATFORM}")

# Find Slang libraries
if(EXISTS "${SLANG_BIN_DIR}")
    file(GLOB SLANG_LIBS "${SLANG_BIN_DIR}/${SLANG_LIB_PATTERN}")
    list(LENGTH SLANG_LIBS SLANG_LIBS_COUNT)
    message(STATUS "Found ${SLANG_LIBS_COUNT} Slang libraries in ${SLANG_BIN_DIR}")
else()
    message(WARNING "Slang binaries not found at ${SLANG_BIN_DIR}")
    message(WARNING "Run slang/download.sh to download Slang binaries")
    set(SLANG_LIBS "")
endif()

# =============================================================================
# Examples
# =============================================================================

if(GOLDY_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# =============================================================================
# Installation
# =============================================================================

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Install headers
install(
    FILES
        include/goldy.h
        include/goldy.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install native library (goldy_ffi)
if(WIN32)
    if(GOLDY_BUILD_FROM_SOURCE)
        install(FILES ${GOLDY_FFI_LIB} DESTINATION ${CMAKE_INSTALL_BINDIR})
        install(FILES ${GOLDY_FFI_IMPLIB} DESTINATION ${CMAKE_INSTALL_LIBDIR})
    elseif(GOLDY_FFI_DLL)
        install(FILES ${GOLDY_FFI_DLL} DESTINATION ${CMAKE_INSTALL_BINDIR})
        install(FILES ${GOLDY_FFI_LIB} DESTINATION ${CMAKE_INSTALL_LIBDIR})
    endif()
else()
    install(FILES ${GOLDY_FFI_LIB} DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

# Install Slang libraries alongside goldy_ffi
if(SLANG_LIBS)
    if(WIN32)
        install(FILES ${SLANG_LIBS} DESTINATION ${CMAKE_INSTALL_BINDIR})
    else()
        install(FILES ${SLANG_LIBS} DESTINATION ${CMAKE_INSTALL_LIBDIR})
    endif()
    message(STATUS "Will install Slang libraries to ${CMAKE_INSTALL_BINDIR}")
endif()

# Install CMake config
install(TARGETS goldy
    EXPORT goldyTargets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT goldyTargets
    FILE goldyTargets.cmake
    NAMESPACE goldy::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/goldy
)

# Create config file
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/goldyConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/goldyConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/goldy
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/goldyConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/goldyConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/goldyConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/goldy
)

# =============================================================================
# Package info
# =============================================================================

message(STATUS "")
message(STATUS "Goldy GPU Library v${PROJECT_VERSION}")
message(STATUS "  Build from source: ${GOLDY_BUILD_FROM_SOURCE}")
message(STATUS "  Build examples: ${GOLDY_BUILD_EXAMPLES}")
if(GOLDY_FFI_LIB)
    message(STATUS "  FFI library: ${GOLDY_FFI_LIB}")
endif()
message(STATUS "")

