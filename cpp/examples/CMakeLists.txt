# Goldy C++ Examples

# Triangle example - interactive Surface + TaskGraph render
add_executable(triangle triangle.cpp)
target_link_libraries(triangle PRIVATE goldy::goldy)

if(APPLE)
    set_source_files_properties(triangle.cpp PROPERTIES LANGUAGE OBJCXX)
    target_link_libraries(triangle PRIVATE "-framework AppKit" "-framework QuartzCore")
elseif(WIN32)
    target_link_libraries(triangle PRIVATE user32 gdi32)
endif()

set(GOLDY_EXAMPLES triangle)

# Platform-specific shared library handling for all examples
if(GOLDY_FFI_LIB)
    foreach(example IN LISTS GOLDY_EXAMPLES)
        if(WIN32)
            # Copy DLL to output directory on Windows (GOLDY_FFI_DLL is the runtime DLL;
            # GOLDY_FFI_LIB is the import lib used only at link time)
            if(GOLDY_FFI_DLL)
                add_custom_command(TARGET ${example} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${GOLDY_FFI_DLL}"
                        "$<TARGET_FILE_DIR:${example}>"
                    COMMENT "Copying goldy_ffi.dll to output directory"
                )
            endif()
        elseif(APPLE)
            # Set rpath for macOS to find the dylib at runtime
            set_target_properties(${example} PROPERTIES
                BUILD_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/../../target/release;${CMAKE_CURRENT_SOURCE_DIR}/../../target/debug"
                INSTALL_RPATH "@executable_path/../lib;@executable_path"
            )
            # Copy dylib to output directory
            add_custom_command(TARGET ${example} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${GOLDY_FFI_LIB}"
                    "$<TARGET_FILE_DIR:${example}>"
                COMMENT "Copying libgoldy_ffi.dylib to output directory"
            )
        else()
            # Set rpath for Linux to find the .so at runtime
            set_target_properties(${example} PROPERTIES
                BUILD_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/../../target/release;${CMAKE_CURRENT_SOURCE_DIR}/../../target/debug"
                INSTALL_RPATH "$ORIGIN/../lib:$ORIGIN"
            )
            # Copy .so to output directory
            add_custom_command(TARGET ${example} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${GOLDY_FFI_LIB}"
                    "$<TARGET_FILE_DIR:${example}>"
                COMMENT "Copying libgoldy_ffi.so to output directory"
            )
        endif()
        
        # Copy Slang libraries to output directory (required for runtime)
        if(SLANG_LIBS)
            foreach(slang_lib IN LISTS SLANG_LIBS)
                add_custom_command(TARGET ${example} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${slang_lib}"
                        "$<TARGET_FILE_DIR:${example}>"
                    COMMENT "Copying Slang library to output directory"
                )
            endforeach()
        endif()
    endforeach()
endif()

