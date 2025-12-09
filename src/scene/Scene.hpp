// Scene.hpp - All scene data structures and classes (header-only)
#pragma once

#include <simd/simd.h>
#include <glm/glm.hpp>
#include <vector>
#include <cmath>
#include <cstdint>

// ============================================================================
// PHYSICS CONSTANTS
// ============================================================================

// C++17: inline constexpr for ODR-safe constants (can be in header)
inline constexpr double c_light = 299792458.0;
inline constexpr double G_grav  = 6.67430e-11;

// C++11: constexpr helper functions for compile-time physics
constexpr double schwarzschildRadius(double mass) noexcept {
    return 2.0 * G_grav * mass / (c_light * c_light);
}

// ============================================================================
// CAMERA CONSTANTS
// ============================================================================

namespace constants {
    inline constexpr float PI_F = 3.14159265358979323846f;
    inline constexpr float MIN_ELEVATION = 0.01f;
    inline constexpr float MAX_ELEVATION = PI_F - 0.01f;
    inline constexpr float DEFAULT_ELEVATION = PI_F / 2.0f;
    
    // C++11: constexpr trig for compile-time calculations
    constexpr float toRadians(float degrees) noexcept {
        return degrees * PI_F / 180.0f;
    }
}

// ============================================================================
// UNIFORM BUFFER OBJECTS (must match Metal shaders)
// ============================================================================

struct CameraUBO {
    simd::float3 camPos;     float _pad0;
    simd::float3 camRight;   float _pad1;
    simd::float3 camUp;      float _pad2;
    simd::float3 camForward; float _pad3;
    float  tanHalfFov;
    float  aspect;
    uint32_t   moving;
    float  time;
};

struct DiskUBO {
    float disk_r1;
    float disk_r2;
    float disk_num;
    float thickness;
};

struct ObjectsUBO {
    int   numObjects;
    float _pad0, _pad1, _pad2;
    simd::float4 objPosRadius[16];
    simd::float4 objColor[16];
    float  mass[16];
};

// ============================================================================
// CAMERA CLASS
// ============================================================================

class Camera {
public:
    glm::vec3 target = glm::vec3(0.0f, 0.0f, 0.0f);
    float radius = 17.0e10f;
    float minRadius = 1e10f, maxRadius = 25.0e10f;
    float azimuth = 0.0f;
    float elevation = constants::PI_F / 2.4f;
    float orbitSpeed = 0.01f;
    float panSpeed = 0.005f;
    double zoomSpeed = 25e9f;
    bool dragging = false;
    bool panning = false;
    bool moving = false;

    [[nodiscard]] glm::vec3 position() const {
        const float clampedElevation = glm::clamp(elevation, constants::MIN_ELEVATION, constants::MAX_ELEVATION);
        const float sinElev = std::sin(clampedElevation);
        const float cosElev = std::cos(clampedElevation);
        const float cosAz = std::cos(azimuth);
        const float sinAz = std::sin(azimuth);
        
        return glm::vec3(
            radius * sinElev * cosAz,
            radius * cosElev,
            radius * sinElev * sinAz
        );
    }
    
    void update() {
        target = glm::vec3(0.0f, 0.0f, 0.0f);
        moving = dragging || panning;
    }
    
    void onMouseButton(int button, int action) {
        if (button == 0 || button == 2) {
            if (action == 1) {
                dragging = true;
                panning = false;
            } else if (action == 0) {
                dragging = false;
                panning = false;
            }
        }
    }
    
    void onMouseMove(double x, double y) {
        const float dx = static_cast<float>(x - lastX_);
        const float dy = static_cast<float>(y - lastY_);
        
        if (dragging && !panning) {
            azimuth += dx * orbitSpeed;
            elevation -= dy * orbitSpeed;
            elevation = glm::clamp(elevation, constants::MIN_ELEVATION, constants::MAX_ELEVATION);
        }
        
        lastX_ = x;
        lastY_ = y;
        update();
    }
    
    void onScroll(double, double yoffset) {
        radius -= yoffset * zoomSpeed;
        radius = glm::clamp(radius, minRadius, maxRadius);
        update();
    }

private:
    double lastX_ = 0.0, lastY_ = 0.0;
};

// ============================================================================
// PHYSICS OBJECTS
// ============================================================================

struct BlackHole {
    glm::vec3 position;
    double mass;
    double r_s;

    BlackHole(glm::vec3 pos, double m) : position(pos), mass(m) {
        r_s = 2.0 * G_grav * mass / (c_light * c_light);
    }
};

struct ObjectData {
    simd::float4 posRadius;
    simd::float4 color;
    float mass;
    glm::vec3 velocity = glm::vec3(0.0f, 0.0f, 0.0f);
};

// ============================================================================
// SCENE CLASS
// ============================================================================

class Scene {
public:
    Scene() 
        : blackHole_(glm::vec3(0.0f, 0.0f, 0.0f), 8.54e36)
        , gravityEnabled_(false)
    {
        // Initialize celestial bodies
        objects_ = {
            // Blue star
            { simd::float4{2.3e11f, 0.0f, 0.0f, 4e10f}, 
              simd::float4{0.4f, 0.7f, 1.0f, 1.0f},
              1.98892e30f, 
              glm::vec3(0.0f, 0.0f, 5.34e7f) },
            // Red star
            { simd::float4{-1.6e11f, 0.0f, 0.0f, 4e10f}, 
              simd::float4{0.8f, 0.3f, 0.2f, 1.0f},
              1.98892e30f,
              glm::vec3(0.0f, 0.0f, -5.34e7f) },
            // Black hole at center
            { simd::float4{0.0f, 0.0f, 0.0f, static_cast<float>(blackHole_.r_s)}, 
              simd::float4{0.0f, 0.0f, 0.0f, 1.0f}, 
              static_cast<float>(blackHole_.mass),
              glm::vec3(0.0f, 0.0f, 0.0f) }
        };
    }
    
    const std::vector<ObjectData>& objects() const { return objects_; }
    std::vector<ObjectData>& objects() { return objects_; }
    const BlackHole& blackHole() const { return blackHole_; }
    bool gravityEnabled() const { return gravityEnabled_; }
    
    void setGravityEnabled(bool enabled) { gravityEnabled_ = enabled; }
    void toggleGravity() { gravityEnabled_ = !gravityEnabled_; }
    
private:
    std::vector<ObjectData> objects_;
    BlackHole blackHole_;
    bool gravityEnabled_;
};

// ============================================================================
// GLOBAL INSTANCES
// ============================================================================

inline Camera camera;
inline Scene scene;

