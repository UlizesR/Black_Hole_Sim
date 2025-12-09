cmake_minimum_required(VERSION 3.21)
project(BlackHoleSim VERSION 1.0.0 LANGUAGES CXX)

# C++20 standard for modern features
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Detect platform
if(APPLE)
    set(PLATFORM_MACOS TRUE)
endif()

# Dependencies
find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)

# Common dependencies
set(DEPS glfw glm::glm)

# OpenGL dependencies (for OpenGL backend)
find_package(GLEW)
find_package(OpenGL)

if(GLEW_FOUND AND OpenGL_FOUND)
    set(OPENGL_AVAILABLE TRUE)
    list(APPEND DEPS GLEW::GLEW OpenGL::GL)
endif()

# Metal (macOS only)
if(PLATFORM_MACOS)
    find_library(METAL_FRAMEWORK Metal)
    find_library(METALKIT_FRAMEWORK MetalKit)
    find_library(QUARTZCORE_FRAMEWORK QuartzCore)
    find_library(COCOA_FRAMEWORK Cocoa)
    if(METAL_FRAMEWORK AND METALKIT_FRAMEWORK AND QUARTZCORE_FRAMEWORK AND COCOA_FRAMEWORK)
        set(METAL_AVAILABLE TRUE)
        list(APPEND DEPS ${METAL_FRAMEWORK} ${METALKIT_FRAMEWORK} ${QUARTZCORE_FRAMEWORK} ${COCOA_FRAMEWORK})
    endif()
endif()

# ============================================================================
# Source Files - Organized Structure
# ============================================================================

# Main application
set(MAIN_SOURCES
    src/main.cpp
)

# Engine (CRTP base + Metal backend)
set(ENGINE_SOURCES
    src/engine/RenderEngine.hpp
    src/engine/MetalEngine.hpp
)

# Scene data
set(SCENE_SOURCES
    src/scene/Scene.hpp
)

# Metal shaders
set(METAL_SHADERS
    src/shaders/shaders.metal
)

# ============================================================================
# Executable Target
# ============================================================================

add_executable(BlackHole3D 
    ${MAIN_SOURCES}
    ${ENGINE_SOURCES}
    ${SCENE_SOURCES}
)

# ============================================================================
# Platform Detection & Backend Selection
# ============================================================================

if(PLATFORM_MACOS)
    target_compile_definitions(BlackHole3D PRIVATE PLATFORM_MACOS=1)
    message(STATUS "Platform: macOS")
endif()

# Metal backend (macOS only)
if(PLATFORM_MACOS AND METAL_AVAILABLE)
    target_sources(BlackHole3D PRIVATE src/engine/MetalEngine.mm)
    target_compile_definitions(BlackHole3D PRIVATE METAL_BACKEND_AVAILABLE)
    
    # Configure Objective-C++ compilation
    set_source_files_properties(src/engine/MetalEngine.mm PROPERTIES 
        COMPILE_FLAGS "-x objective-c++ -fobjc-arc -std=c++20")
    
    # Compile Metal shaders to .metallib
    set(METALLIB_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/default.metallib")
    set(SHADER_AIR "${CMAKE_CURRENT_BINARY_DIR}/shaders.air")
    set(SHADER_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/shaders.metal")
    
    add_custom_command(
        OUTPUT ${METALLIB_OUTPUT}
        COMMAND xcrun -sdk macosx metal -c "${SHADER_SOURCE}" -o "${SHADER_AIR}"
        COMMAND xcrun -sdk macosx metallib "${SHADER_AIR}" -o "${METALLIB_OUTPUT}"
        DEPENDS "${SHADER_SOURCE}"
        COMMENT "Compiling Metal shaders to default.metallib"
        VERBATIM
    )
    
    add_custom_target(MetalShaders ALL DEPENDS ${METALLIB_OUTPUT})
    add_dependencies(BlackHole3D MetalShaders)
    
    # Copy metallib to executable directory
    add_custom_command(TARGET BlackHole3D POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${METALLIB_OUTPUT}
            $<TARGET_FILE_DIR:BlackHole3D>/default.metallib
        COMMENT "Copying default.metallib to output directory"
    )
    
    message(STATUS "Backend: Metal (macOS)")
elseif(OPENGL_AVAILABLE)
    message(STATUS "Backend: OpenGL")
else()
    message(FATAL_ERROR "No rendering backend available!")
endif()

# ============================================================================
# Include Directories
# ============================================================================

target_include_directories(BlackHole3D PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/engine
    ${CMAKE_CURRENT_SOURCE_DIR}/src/scene
)

# ============================================================================
# Link Libraries
# ============================================================================

target_link_libraries(BlackHole3D PRIVATE ${DEPS})

# ============================================================================
# Compiler Options
# ============================================================================

# Optimization flags
if(CMAKE_BUILD_TYPE STREQUAL "Release" OR NOT CMAKE_BUILD_TYPE)
    target_compile_options(BlackHole3D PRIVATE
        -O3
        -march=native
        -ffast-math
        -DNDEBUG
        -flto
    )
    target_link_options(BlackHole3D PRIVATE -flto)
else()
    target_compile_options(BlackHole3D PRIVATE
        -g
        -O0
        -Wall
        -Wextra
    )
endif()

# ============================================================================
# Code Signing (macOS - for Instruments profiling)
# ============================================================================

if(PLATFORM_MACOS)
    set(ENTITLEMENTS_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/resources/entitlements.plist")
    if(EXISTS ${ENTITLEMENTS_FILE})
        add_custom_command(TARGET BlackHole3D POST_BUILD
            COMMAND codesign --entitlements ${ENTITLEMENTS_FILE} --force --sign - $<TARGET_FILE:BlackHole3D>
            COMMENT "Code signing for Instruments debugging"
            VERBATIM
        )
    endif()
endif()

