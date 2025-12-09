// MetalEngine.hpp - Metal backend implementation
#pragma once

#include "RenderEngine.hpp"  // Same directory
#include <memory>

// Forward declarations
struct GLFWwindow;
namespace simd { struct float4x4; }
class Camera;
struct ObjectData;

// ============================================================================
// MetalEngine - Metal API backend using CRTP for zero-cost abstraction
// ============================================================================

class MetalEngine : public RenderEngine<MetalEngine> {
public:
    // C++11: Explicit constructor
    explicit MetalEngine(int width = 800, int height = 600,
                        int computeWidth = 200, int computeHeight = 150);
    ~MetalEngine();

    // Non-copyable, non-movable (manages GPU resources)
    MetalEngine(const MetalEngine&) = delete;
    MetalEngine& operator=(const MetalEngine&) = delete;
    MetalEngine(MetalEngine&&) = delete;
    MetalEngine& operator=(MetalEngine&&) = delete;

    // ========================================================================
    // CRTP Implementation Interface (called by base class)
    // ========================================================================
    
    [[nodiscard]] GLFWwindow* windowImpl() const;
    [[nodiscard]] int widthImpl() const;
    [[nodiscard]] int heightImpl() const;
    [[nodiscard]] int computeWidthImpl() const;
    [[nodiscard]] int computeHeightImpl() const;
    [[nodiscard]] const char* backendNameImpl() const { return "Metal"; }
    
    void frameImpl(const Camera& cam,
                   std::vector<ObjectData>& objects,
                   const simd::float4x4& viewProj,
                   bool gravityEnabled);
    
    void setQualityPresetImpl(int preset);

private:
    // Pimpl idiom: Hide Objective-C++ implementation details
    struct Impl;
    std::unique_ptr<Impl> impl_;
    
    // Friend base class for CRTP access
    friend class RenderEngine<MetalEngine>;
};

