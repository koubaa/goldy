# Goldy C++ Examples

# Triangle example - basic rendering
add_executable(triangle triangle.cpp)
target_link_libraries(triangle PRIVATE goldy::goldy)

# Platform-specific shared library handling
if(GOLDY_FFI_LIB)
    if(WIN32)
        # Copy DLL to output directory on Windows
        add_custom_command(TARGET triangle POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${GOLDY_FFI_LIB}"
                "$<TARGET_FILE_DIR:triangle>"
            COMMENT "Copying goldy_ffi.dll to output directory"
        )
    elseif(APPLE)
        # Set rpath for macOS to find the dylib at runtime
        set_target_properties(triangle PROPERTIES
            BUILD_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/../../target/release;${CMAKE_CURRENT_SOURCE_DIR}/../../target/debug"
            INSTALL_RPATH "@executable_path/../lib;@executable_path"
        )
        # Copy dylib to output directory
        add_custom_command(TARGET triangle POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${GOLDY_FFI_LIB}"
                "$<TARGET_FILE_DIR:triangle>"
            COMMENT "Copying libgoldy_ffi.dylib to output directory"
        )
    else()
        # Set rpath for Linux to find the .so at runtime
        set_target_properties(triangle PROPERTIES
            BUILD_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/../../target/release;${CMAKE_CURRENT_SOURCE_DIR}/../../target/debug"
            INSTALL_RPATH "$ORIGIN/../lib:$ORIGIN"
        )
        # Copy .so to output directory
        add_custom_command(TARGET triangle POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${GOLDY_FFI_LIB}"
                "$<TARGET_FILE_DIR:triangle>"
            COMMENT "Copying libgoldy_ffi.so to output directory"
        )
    endif()
endif()

