// GraphicsCommon.hpp - Shared constants and logic for all render backends (Metal, OpenGL, etc.)
#pragma once

#include <chrono>
#include <cmath>
#include <cstddef>
#include <utility>

// ============================================================================
// Buffer alignment (Metal requires 256-byte alignment for uniform buffers)
// ============================================================================

constexpr size_t alignTo256(size_t size) noexcept {
    return (size + 255) & ~size_t(255);
}

// ============================================================================
// Frame / resource constants (shared by backends that use double/triple buffering)
// ============================================================================

namespace Graphics {
    inline constexpr int MAX_FRAMES_IN_FLIGHT = 3;
}

// ============================================================================
// Quality presets (shared across backends)
// ============================================================================

namespace QualityPreset {
    inline constexpr int FAST = 0;       // ~100 FPS, 3-4k steps
    inline constexpr int MEDIUM = 1;     // ~50 FPS, 6-8k steps
    inline constexpr int CINEMATIC = 2;  // ~20 FPS, 15-20k steps
}

// ============================================================================
// Adaptive quality - backend-agnostic frame time and resolution scaling
// ============================================================================

struct AdaptiveQuality {
    float targetFrameTimeMs = 16.67f;  // 60 FPS target
    float currentFrameTimeMs = 20.0f;
    float renderScale = 0.25f;
    int measurementCount = 0;
    std::chrono::high_resolution_clock::time_point lastFrameTime;

    AdaptiveQuality()
        : lastFrameTime(std::chrono::high_resolution_clock::now()) {}

    void updateFrameTime() {
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - lastFrameTime);
        currentFrameTimeMs = static_cast<float>(elapsed.count()) / 1000.0f;
        lastFrameTime = now;
        measurementCount++;
    }

    void adjustQuality() {
        if (measurementCount < 30) return;

        if (currentFrameTimeMs > targetFrameTimeMs * 1.15f) {
            renderScale = std::max(0.15f, renderScale * 0.95f);
        } else if (currentFrameTimeMs < targetFrameTimeMs * 0.85f) {
            renderScale = std::min(0.5f, renderScale * 1.02f);
        }
    }

    std::pair<int, int> getResolution(int width, int height) const {
        return {
            static_cast<int>(width * renderScale),
            static_cast<int>(height * renderScale)
        };
    }

    // Optional: for verbose/debug logging (e.g. ADAPTIVE_QUALITY)
    bool shouldReduceQuality() const { return currentFrameTimeMs > targetFrameTimeMs * 1.15f; }
    bool shouldIncreaseQuality() const { return currentFrameTimeMs < targetFrameTimeMs * 0.85f; }
};
