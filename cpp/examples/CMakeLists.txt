# Goldy C++ Examples

# Triangle example - basic rendering
add_executable(triangle triangle.cpp)
target_link_libraries(triangle PRIVATE goldy::goldy)

# Copy DLL to output directory on Windows
if(WIN32 AND GOLDY_FFI_LIB)
    add_custom_command(TARGET triangle POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${GOLDY_FFI_LIB}"
            "$<TARGET_FILE_DIR:triangle>"
        COMMENT "Copying goldy_ffi.dll to output directory"
    )
endif()

