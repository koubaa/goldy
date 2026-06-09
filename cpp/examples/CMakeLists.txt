# Goldy C++ Examples

include(FetchContent)
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

add_executable(triangle triangle.cpp)
target_link_libraries(triangle PRIVATE goldy::goldy glfw)

add_executable(triangle_headless triangle_headless.cpp)
target_link_libraries(triangle_headless PRIVATE goldy::goldy)

add_executable(game_of_life game_of_life.cpp)
target_link_libraries(game_of_life PRIVATE goldy::goldy glfw)

if(APPLE)
    set_source_files_properties(triangle.cpp PROPERTIES LANGUAGE OBJCXX)
    set_source_files_properties(game_of_life.cpp PROPERTIES LANGUAGE OBJCXX)
    target_link_libraries(triangle PRIVATE "-framework Cocoa" "-framework QuartzCore")
    target_link_libraries(game_of_life PRIVATE "-framework Cocoa" "-framework QuartzCore")
endif()

set(GOLDY_EXAMPLES triangle triangle_headless game_of_life)

# Platform-specific shared library handling for all examples
if(GOLDY_FFI_LIB)
    foreach(example IN LISTS GOLDY_EXAMPLES)
        if(WIN32)
            if(GOLDY_FFI_DLL)
                add_custom_command(TARGET ${example} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${GOLDY_FFI_DLL}"
                        "$<TARGET_FILE_DIR:${example}>"
                    COMMENT "Copying goldy_ffi.dll to output directory"
                )
            endif()
        elseif(APPLE)
            set_target_properties(${example} PROPERTIES
                BUILD_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/../../target/release;${CMAKE_CURRENT_SOURCE_DIR}/../../target/debug"
                INSTALL_RPATH "@executable_path/../lib;@executable_path"
            )
            add_custom_command(TARGET ${example} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${GOLDY_FFI_LIB}"
                    "$<TARGET_FILE_DIR:${example}>"
                COMMENT "Copying libgoldy_ffi.dylib to output directory"
            )
        else()
            set_target_properties(${example} PROPERTIES
                BUILD_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/../../target/release;${CMAKE_CURRENT_SOURCE_DIR}/../../target/debug"
                INSTALL_RPATH "$ORIGIN/../lib:$ORIGIN"
            )
            add_custom_command(TARGET ${example} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${GOLDY_FFI_LIB}"
                    "$<TARGET_FILE_DIR:${example}>"
                COMMENT "Copying libgoldy_ffi.so to output directory"
            )
        endif()

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
