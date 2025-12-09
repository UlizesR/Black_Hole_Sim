// main.cpp - Entry point for Black Hole renderer (backend-agnostic)
#include "engine/RenderEngine.hpp"  // Base class interface
#include "engine/MetalEngine.hpp"   // Concrete backend
#include "scene/Scene.hpp"
#include <GLFW/glfw3.h>
#include <simd/simd.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <chrono>
#include <memory>

using namespace glm;
using namespace std;

// C++14: Template helper to print info for any RenderEngine backend
template<typename Backend>
void printEngineInfo(const RenderEngine<Backend>& engine) {
    cout << "[INFO] Render Backend: " << engine.backendName() << endl;
    cout << "[INFO] Window Size: " << engine.width() << "x" << engine.height() << endl;
    cout << "[INFO] Compute Size: " << engine.computeWidth() << "x" << engine.computeHeight() << endl;
}

// Input handler class wraps GLFW callbacks
class InputHandler {
public:
    static void setupCallbacks(GLFWwindow* window) {
        glfwSetWindowUserPointer(window, nullptr);
        
        glfwSetMouseButtonCallback(window, [](GLFWwindow*, int button, int action, int) {
            if (button == GLFW_MOUSE_BUTTON_LEFT || button == GLFW_MOUSE_BUTTON_MIDDLE) {
                camera.onMouseButton(button, action);
            }
            if (button == GLFW_MOUSE_BUTTON_RIGHT) {
                if (action == GLFW_PRESS) {
                    scene.setGravityEnabled(true);
                } else if (action == GLFW_RELEASE) {
                    scene.setGravityEnabled(false);
                }
            }
        });

        glfwSetCursorPosCallback(window, [](GLFWwindow*, double x, double y) {
            camera.onMouseMove(x, y);
        });

        glfwSetScrollCallback(window, [](GLFWwindow*, double xoffset, double yoffset) {
            camera.onScroll(xoffset, yoffset);
        });

        glfwSetKeyCallback(window, [](GLFWwindow*, int key, int, int action, int) {
            if (action == GLFW_PRESS) {
                switch (key) {
                    case GLFW_KEY_G:
                        scene.toggleGravity();
                        cout << "[INFO] Gravity " << (scene.gravityEnabled() ? "ON" : "OFF") << endl;
                        break;
                    case GLFW_KEY_1:
                        cout << "[INFO] Quality: FAST (3-4k steps, ~100+ FPS)" << endl;
                        cout << "[INFO] Note: Quality presets require pipeline rebuild" << endl;
                        break;
                    case GLFW_KEY_2:
                        cout << "[INFO] Quality: MEDIUM (6-8k steps, ~50+ FPS) [Current]" << endl;
                        break;
                    case GLFW_KEY_3:
                        cout << "[INFO] Quality: CINEMATIC (15-20k steps, ~20+ FPS)" << endl;
                        cout << "[INFO] Note: Quality presets require pipeline rebuild" << endl;
                        break;
                }
            }
        });
    }
};

// Performance metrics tracker with modern C++ optimizations
class PerformanceTracker {
public:
    // C++20: Use concepts-style requires clause if needed
    void recordFrame() noexcept {
        ++frameCount_;
        const auto currentTime = std::chrono::high_resolution_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            currentTime - lastPrintTime_
        );
        
        // C++20: [[likely]] attribute for branch prediction
        if (elapsed.count() >= 1000) [[unlikely]] {
            const double fps = frameCount_ * 1000.0 / elapsed.count();
            const double frameTimeMs = elapsed.count() / static_cast<double>(frameCount_);
            
            // C++17: Structured bindings could be used if we returned a tuple
            cout << "[PERF] FPS: " << static_cast<int>(fps) 
                 << " | Frame time: " << frameTimeMs << "ms"
                 << " | Camera: " << (camera.moving ? "MOVING" : "STILL") << endl;
            
            frameCount_ = 0;
            lastPrintTime_ = currentTime;
        }
    }

private:
    int frameCount_ = 0;
    std::chrono::high_resolution_clock::time_point lastPrintTime_ = 
        std::chrono::high_resolution_clock::now();
};

// C++14: Template function that works with ANY RenderEngine backend
template<typename Backend>
void runRenderLoop(RenderEngine<Backend>& engine) {
    // Setup input and performance tracking
    InputHandler::setupCallbacks(engine.window());
    PerformanceTracker perfTracker;

    // Rendering constants
    constexpr float FOV = 60.0f;
    constexpr float NEAR_PLANE = 1e9f;
    constexpr float FAR_PLANE = 1e14f;
    constexpr vec3 UP_VECTOR(0.0f, 1.0f, 0.0f);
    
    // C++11: Lambda for aspect ratio calculation
    auto computeAspect = [&engine]() noexcept {
        return static_cast<float>(engine.width()) / static_cast<float>(engine.height());
    };

    // Main render loop - works with ANY backend!
    while (!glfwWindowShouldClose(engine.window())) [[likely]] {
        // Build view-projection matrix
        const mat4 view = lookAt(camera.position(), camera.target, UP_VECTOR);
        const mat4 proj = perspective(radians(FOV), computeAspect(), NEAR_PLANE, FAR_PLANE);
        const mat4 viewProj = proj * view;

        // Render using base class interface (CRTP dispatch)
        engine.frame(
            camera, 
            scene.objects(), 
            *reinterpret_cast<const simd_float4x4*>(&viewProj), 
            scene.gravityEnabled()
        );

        perfTracker.recordFrame();
        glfwPollEvents();
    }
}

int main() {
    // CRTP Architecture: Zero-cost abstraction for render backends
    // Could easily add other backends: VulkanEngine, DX12Engine, WebGPUEngine
    // All inherit from RenderEngine<T> with same interface, zero runtime overhead
    
    MetalEngine metalBackend(800, 600, 200, 150);
    
    // Print engine info using base class interface
    printEngineInfo(metalBackend);
    
    // Run render loop using base class interface (works with any backend)
    runRenderLoop(metalBackend);
    
    glfwTerminate();
    return 0;
}
