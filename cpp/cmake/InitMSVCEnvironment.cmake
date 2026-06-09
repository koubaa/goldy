# Populate MSVC + Windows SDK include/link paths when CMake leaves them empty.
#
# This happens when configuring with the Ninja generator from Git Bash, Cursor,
# or other shells that do not run vcvars64.bat. cl.exe is discovered on PATH
# but standard headers (stdarg.h, etc.) are not on the include path.

if(NOT MSVC)
    return()
endif()

if(CMAKE_CXX_FLAGS MATCHES "/I\".*MSVC/.*/include\"")
    return()
endif()

if(NOT CMAKE_CXX_COMPILER)
    message(WARNING "InitMSVCEnvironment: CMAKE_CXX_COMPILER is not set")
    return()
endif()

# MSVC toolset include/lib directories (derived from cl.exe location).
get_filename_component(_GOLDY_MSVC_BIN_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
get_filename_component(_GOLDY_MSVC_HOST_DIR "${_GOLDY_MSVC_BIN_DIR}" DIRECTORY)
get_filename_component(_GOLDY_MSVC_BIN_ROOT "${_GOLDY_MSVC_HOST_DIR}" DIRECTORY)
get_filename_component(_GOLDY_MSVC_TOOLSET_DIR "${_GOLDY_MSVC_BIN_ROOT}" DIRECTORY)

set(_GOLDY_MSVC_INCLUDE "${_GOLDY_MSVC_TOOLSET_DIR}/include")
set(_GOLDY_MSVC_LIB "${_GOLDY_MSVC_TOOLSET_DIR}/lib/x64")

# Windows SDK version: prefer the version CMake already found for mt.exe/rc.exe.
set(_GOLDY_WINSDK_VERSION "")
if(CMAKE_MT)
    get_filename_component(_GOLDY_MT_BIN_DIR "${CMAKE_MT}" DIRECTORY)
    get_filename_component(_GOLDY_WINSDK_VERSION "${_GOLDY_MT_BIN_DIR}" NAME)
endif()

set(_GOLDY_WINSDK_ROOT "C:/Program Files (x86)/Windows Kits/10")
if(NOT _GOLDY_WINSDK_VERSION OR NOT IS_DIRECTORY "${_GOLDY_WINSDK_ROOT}/Include/${_GOLDY_WINSDK_VERSION}")
    file(GLOB _GOLDY_WINSDK_VERSIONS LIST_DIRECTORIES true "${_GOLDY_WINSDK_ROOT}/Include/*")
    list(SORT _GOLDY_WINSDK_VERSIONS COMPARE NATURAL ORDER DESCENDING)
    foreach(_GOLDY_CANDIDATE IN LISTS _GOLDY_WINSDK_VERSIONS)
        get_filename_component(_GOLDY_WINSDK_VERSION "${_GOLDY_CANDIDATE}" NAME)
        if(_GOLDY_WINSDK_VERSION MATCHES "^10\\.")
            break()
        endif()
    endforeach()
endif()

set(_GOLDY_WINSDK_INCLUDE_ROOT "${_GOLDY_WINSDK_ROOT}/Include/${_GOLDY_WINSDK_VERSION}")
set(_GOLDY_WINSDK_LIB_ROOT "${_GOLDY_WINSDK_ROOT}/Lib/${_GOLDY_WINSDK_VERSION}")
set(_GOLDY_WINSDK_UCRT_INCLUDE "${_GOLDY_WINSDK_INCLUDE_ROOT}/ucrt")
set(_GOLDY_WINSDK_SHARED_INCLUDE "${_GOLDY_WINSDK_INCLUDE_ROOT}/shared")
set(_GOLDY_WINSDK_UM_INCLUDE "${_GOLDY_WINSDK_INCLUDE_ROOT}/um")
set(_GOLDY_WINSDK_UCRT_LIB "${_GOLDY_WINSDK_LIB_ROOT}/ucrt/x64")
set(_GOLDY_WINSDK_UM_LIB "${_GOLDY_WINSDK_LIB_ROOT}/um/x64")

foreach(_GOLDY_REQUIRED_DIR IN ITEMS
    "${_GOLDY_MSVC_INCLUDE}"
    "${_GOLDY_WINSDK_UCRT_INCLUDE}"
    "${_GOLDY_WINSDK_SHARED_INCLUDE}"
    "${_GOLDY_WINSDK_UM_INCLUDE}"
)
    if(NOT IS_DIRECTORY "${_GOLDY_REQUIRED_DIR}")
        message(FATAL_ERROR
            "InitMSVCEnvironment: required MSVC/Windows SDK directory not found: ${_GOLDY_REQUIRED_DIR}\n"
            "Open the 'x64 Native Tools Command Prompt for VS 2022' and re-run CMake, "
            "or install the MSVC v143 toolset and Windows 10/11 SDK.")
    endif()
endforeach()

# Cache variables help try_compile and other CMake checks.
set(CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES
    "${_GOLDY_MSVC_INCLUDE}"
    "${_GOLDY_WINSDK_UCRT_INCLUDE}"
    "${_GOLDY_WINSDK_SHARED_INCLUDE}"
    "${_GOLDY_WINSDK_UM_INCLUDE}"
    CACHE STRING "MSVC implicit include directories" FORCE)
set(CMAKE_C_IMPLICIT_INCLUDE_DIRECTORIES
    "${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES}"
    CACHE STRING "MSVC implicit include directories" FORCE)

set(CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES
    "${_GOLDY_MSVC_LIB}"
    "${_GOLDY_WINSDK_UCRT_LIB}"
    "${_GOLDY_WINSDK_UM_LIB}"
    CACHE STRING "MSVC implicit link directories" FORCE)
set(CMAKE_C_IMPLICIT_LINK_DIRECTORIES
    "${CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES}"
    CACHE STRING "MSVC implicit link directories" FORCE)

set(CMAKE_CXX_IMPLICIT_LINK_LIBRARIES
    kernel32.lib user32.lib gdi32.lib winspool.lib shell32.lib ole32.lib oleaut32.lib uuid.lib comdlg32.lib advapi32.lib
    CACHE STRING "MSVC implicit link libraries" FORCE)
set(CMAKE_C_IMPLICIT_LINK_LIBRARIES
    "${CMAKE_CXX_IMPLICIT_LINK_LIBRARIES}"
    CACHE STRING "MSVC implicit link libraries" FORCE)

# Ninja does not always honor late implicit-dir cache updates; append cl.exe flags directly.
string(APPEND CMAKE_CXX_FLAGS " /I\"${_GOLDY_MSVC_INCLUDE}\"")
string(APPEND CMAKE_CXX_FLAGS " /I\"${_GOLDY_WINSDK_UCRT_INCLUDE}\"")
string(APPEND CMAKE_CXX_FLAGS " /I\"${_GOLDY_WINSDK_SHARED_INCLUDE}\"")
string(APPEND CMAKE_CXX_FLAGS " /I\"${_GOLDY_WINSDK_UM_INCLUDE}\"")
string(APPEND CMAKE_C_FLAGS " /I\"${_GOLDY_MSVC_INCLUDE}\"")
string(APPEND CMAKE_C_FLAGS " /I\"${_GOLDY_WINSDK_UCRT_INCLUDE}\"")
string(APPEND CMAKE_C_FLAGS " /I\"${_GOLDY_WINSDK_SHARED_INCLUDE}\"")
string(APPEND CMAKE_C_FLAGS " /I\"${_GOLDY_WINSDK_UM_INCLUDE}\"")

string(APPEND CMAKE_EXE_LINKER_FLAGS " /LIBPATH:\"${_GOLDY_MSVC_LIB}\"")
string(APPEND CMAKE_EXE_LINKER_FLAGS " /LIBPATH:\"${_GOLDY_WINSDK_UCRT_LIB}\"")
string(APPEND CMAKE_EXE_LINKER_FLAGS " /LIBPATH:\"${_GOLDY_WINSDK_UM_LIB}\"")
string(APPEND CMAKE_SHARED_LINKER_FLAGS " /LIBPATH:\"${_GOLDY_MSVC_LIB}\"")
string(APPEND CMAKE_SHARED_LINKER_FLAGS " /LIBPATH:\"${_GOLDY_WINSDK_UCRT_LIB}\"")
string(APPEND CMAKE_SHARED_LINKER_FLAGS " /LIBPATH:\"${_GOLDY_WINSDK_UM_LIB}\"")

message(STATUS "InitMSVCEnvironment: configured MSVC includes for Ninja (SDK ${_GOLDY_WINSDK_VERSION})")
