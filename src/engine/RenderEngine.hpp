// RenderEngine.hpp - CRTP Base class for render backends
#pragma once

#include "GraphicsCommon.hpp"
#include <vector>
#include <cstdint>
#include <type_traits>

// Forward declarations
struct GLFWwindow;
namespace simd { struct float4x4; }
class Camera;
struct ObjectData;

// ============================================================================
// CRTP Base Class - Zero-cost abstraction for render backends
// ============================================================================

template<typename Derived>
class RenderEngine {
public:
    // C++11: Static polymorphism via CRTP (no virtual functions, no vtable)
    
    // Window management
    [[nodiscard]] GLFWwindow* window() const {
        return derived().windowImpl();
    }
    
    [[nodiscard]] int width() const {
        return derived().widthImpl();
    }
    
    [[nodiscard]] int height() const {
        return derived().heightImpl();
    }
    
    [[nodiscard]] int computeWidth() const {
        return derived().computeWidthImpl();
    }
    
    [[nodiscard]] int computeHeight() const {
        return derived().computeHeightImpl();
    }
    
    // Main rendering entry point
    void frame(const Camera& cam,
               std::vector<ObjectData>& objects,
               const simd::float4x4& viewProj,
               bool gravityEnabled = false) {
        derived().frameImpl(cam, objects, viewProj, gravityEnabled);
    }
    
    // Backend information
    [[nodiscard]] const char* backendName() const {
        return derived().backendNameImpl();
    }
    
    // Quality control
    void setQualityPreset(int preset) {
        derived().setQualityPresetImpl(preset);
    }
    
protected:
    // C++11: constexpr for compile-time type checking
    // Ensures this is only used with CRTP pattern
    RenderEngine() {
        static_assert(std::is_base_of<RenderEngine, Derived>::value,
                      "Derived must inherit from RenderEngine<Derived>");
    }
    
    ~RenderEngine() = default;
    
    // Non-copyable (backends manage GPU resources)
    RenderEngine(const RenderEngine&) = delete;
    RenderEngine& operator=(const RenderEngine&) = delete;
    
    // C++11: Movable by default (backends can implement custom move)
    RenderEngine(RenderEngine&&) = default;
    RenderEngine& operator=(RenderEngine&&) = default;

private:
    // CRTP accessor
    [[nodiscard]] Derived& derived() {
        return static_cast<Derived&>(*this);
    }
    
    [[nodiscard]] const Derived& derived() const {
        return static_cast<const Derived&>(*this);
    }
};
