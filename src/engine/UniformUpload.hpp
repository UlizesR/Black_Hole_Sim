// UniformUpload.hpp - Shared logic to fill uniform/UBO structs (Metal and OpenGL)
#pragma once

#include "../scene/Scene.hpp"
#include <algorithm>
#include <cmath>
#include <simd/simd.h>
#include <glm/glm.hpp>

namespace engine {

// Maximum objects in GPU buffers (must match shader definitions)
inline constexpr size_t MAX_OBJECTS = 16;

inline void fillCameraUBO(CameraUBO* data,
                          const Camera& cam,
                          int computeWidth,
                          int computeHeight,
                          float time) {
    glm::vec3 pos = cam.position();
    simd_float3 camPosSimd{pos.x, pos.y, pos.z};
    simd_float3 targetSimd{cam.target.x, cam.target.y, cam.target.z};
    simd_float3 fwd = simd_normalize(targetSimd - camPosSimd);
    simd_float3 up{0, 1, 0};
    simd_float3 right = simd_normalize(simd_cross(fwd, up));
    up = simd_cross(right, fwd);

    data->camPos     = camPosSimd;
    data->camRight   = right;
    data->camUp      = up;
    data->camForward = fwd;
    data->tanHalfFov = std::tan(glm::radians(60.0f * 0.5f));
    data->aspect     = static_cast<float>(computeWidth) / static_cast<float>(computeHeight);
    data->moving     = (cam.dragging || cam.panning) ? 1u : 0u;
    data->time       = time;
}

inline void fillDiskUBO(DiskUBO* data, double r_s) {
    float r = static_cast<float>(r_s);
    data->disk_r1 = r * 2.2f;
    data->disk_r2 = r * 5.2f;
    data->disk_num = 2.0f;
    data->thickness = 1e9f;
}

inline void fillObjectsUBO(ObjectsUBO* data, const std::vector<ObjectData>& objs) {
    const size_t count = std::min(objs.size(), MAX_OBJECTS);
    data->numObjects = static_cast<int>(count);
    for (size_t i = 0; i < count; ++i) {
        data->objPosRadius[i] = objs[i].posRadius;
        data->objColor[i]    = objs[i].color;
        data->mass[i]        = objs[i].mass;
    }
}

} // namespace engine
