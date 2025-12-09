#!/bin/bash

# Build script for test directory
set -e

# Parse command line arguments
DEV_MODE=0
VERBOSE_PERF=0
for arg in "$@"; do
    if [[ "$arg" == "--dev" ]]; then
        DEV_MODE=1
        echo "Building in DEV mode (runtime shader compilation enabled)"
    elif [[ "$arg" == "--perf" ]]; then
        VERBOSE_PERF=1
        echo "Building with VERBOSE_PERF (detailed performance logging)"
    fi
done

TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$TEST_DIR/build"
mkdir -p "$BUILD_DIR"

# Precompile Metal shaders to metallib for faster startup (unless in dev mode)
METALLIB_COMPILED=0
if [[ $DEV_MODE -eq 0 ]]; then
    echo "Attempting to precompile Metal shaders..."
    # Check if metal compiler is actually available
    if xcrun -sdk macosx metal --version &> /dev/null; then
        cd "$TEST_DIR"
        if xcrun -sdk macosx metal -c shaders.metal -o "$BUILD_DIR/shaders.air" 2>&1 && \
           xcrun -sdk macosx metallib "$BUILD_DIR/shaders.air" -o "$BUILD_DIR/default.metallib" 2>&1; then
            rm -f "$BUILD_DIR"/*.air
            echo "✓ Metal shaders compiled to default.metallib"
            METALLIB_COMPILED=1
        else
            echo "⚠ Metal shader compilation failed"
        fi
    else
        echo "⚠ Metal compiler not found - enabling runtime compilation"
        echo "  To install: xcode-select --install"
    fi
    
    # If metallib compilation failed, enable DEV mode automatically
    if [[ $METALLIB_COMPILED -eq 0 ]]; then
        echo "  → Enabling runtime shader compilation (slower startup)"
        DEV_MODE=1
    fi
else
    echo "Skipping metallib precompilation (DEV mode - will compile at runtime)"
fi

# Compiler optimization flags
# -O3: Maximum optimization
# -march=native: Optimize for current CPU architecture
# -flto: Link Time Optimization (enables cross-module optimizations)
# -ffast-math: Fast floating-point math (may reduce precision, but faster)
# -DNDEBUG: Disable assertions in release builds
# -DDEV_RUNTIME_COMPILE: Enable runtime shader compilation when metallib is not available
# -DVERBOSE_PERF: Enable detailed performance logging
if [[ $DEV_MODE -eq 1 ]]; then
    OPT_FLAGS="-O0 -g -DDEV_RUNTIME_COMPILE"
    echo "Using debug flags with runtime shader compilation"
else
    OPT_FLAGS="-O3 -march=native -flto -ffast-math -DNDEBUG"
fi

# Add verbose performance logging if requested
if [[ $VERBOSE_PERF -eq 1 ]]; then
    OPT_FLAGS="$OPT_FLAGS -DVERBOSE_PERF"
fi

# C++20 standard with all modern features
STD_FLAGS="-std=c++20"

# Warning flags for better code quality
WARN_FLAGS="-Wall -Wextra -Wpedantic -Wno-deprecated-declarations"

# Common include paths
INCLUDE_FLAGS="-I\"$TEST_DIR\" -I\"$TEST_DIR/..\" -I\"$TEST_DIR/../src\" $(pkg-config --cflags glfw3 glm 2>/dev/null || echo \"-I/opt/homebrew/include -I/usr/local/include\")"

echo "Compiling C++ sources with C++20 and optimizations..."
# Compile main.cpp (pure C++)
clang++ $STD_FLAGS $OPT_FLAGS $WARN_FLAGS -c "$TEST_DIR/main.cpp" -o "$BUILD_DIR/main.o" \
    $INCLUDE_FLAGS

echo "Compiling Objective-C++ sources with C++20 and optimizations..."
# Compile MetalEngine.mm (Objective-C++)
clang++ $STD_FLAGS $OPT_FLAGS $WARN_FLAGS -x objective-c++ -fobjc-arc -c "$TEST_DIR/MetalEngine.mm" -o "$BUILD_DIR/MetalEngine.o" \
    $INCLUDE_FLAGS \
    -framework Cocoa -framework Metal -framework MetalKit -framework QuartzCore \
    -framework IOKit -framework CoreGraphics -framework CoreVideo

echo "Linking executable with LTO..."
# Link everything together with LTO
clang++ $OPT_FLAGS -o "$BUILD_DIR/test_app" \
    "$BUILD_DIR/main.o" \
    "$BUILD_DIR/MetalEngine.o" \
    $(pkg-config --libs glfw3 glm 2>/dev/null || echo "-L/opt/homebrew/lib -lglfw -lglm") \
    -framework Cocoa -framework Metal -framework MetalKit -framework QuartzCore \
    -framework IOKit -framework CoreGraphics -framework CoreVideo

# Code sign with entitlements for Instruments profiling
echo "Code signing for Instruments debugging..."
codesign --entitlements "$TEST_DIR/entitlements.plist" --force --sign - "$BUILD_DIR/test_app" 2>&1 || echo "Note: Code signing optional, but helps with Instruments"

# Copy metallib to build directory if it exists (Metal will look for it)
if [ -f "$BUILD_DIR/default.metallib" ]; then
    cp "$BUILD_DIR/default.metallib" "$BUILD_DIR/"
fi

echo ""
echo "✓ Build complete! Executable: $BUILD_DIR/test_app"
if [[ $METALLIB_COMPILED -eq 1 ]]; then
    echo "  ⚡ Using precompiled shaders (fast startup)"
elif [[ $DEV_MODE -eq 1 ]]; then
    echo "  ⚠️  Using runtime shader compilation (slower startup)"
    echo "     First launch will take a few seconds to compile shaders"
fi
echo ""
echo "Run with: $BUILD_DIR/test_app"
echo ""
echo "Usage:"
echo "  Release build:         ./build.sh"
echo "  Dev build:             ./build.sh --dev"
echo "  Performance profiling: ./build.sh --perf"
echo "  Dev + profiling:       ./build.sh --dev --perf"

