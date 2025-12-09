// shaders.metal - All Metal shaders in one file  
#include <metal_stdlib>
using namespace metal;
using namespace metal::fast;

#pragma clang fp contract(fast)

// ============================================================================
// SHARED TYPES & CONSTANTS
// ============================================================================

constant float G_CONST = 6.67430e-11;
constant float c_light = 299792458.0;
constant float SagA_rs = 1.269e10f;
constant float ESCAPE_R = 1e30f;

// QUALITY PRESETS
constant int QUALITY_FAST [[function_constant(0)]];
constant int QUALITY_MEDIUM [[function_constant(1)]];
constant int QUALITY_CINEMATIC [[function_constant(2)]];

constant float D_LAMBDA_FAST = 2e8f;
constant float D_LAMBDA_MEDIUM = 1e8f;
constant float D_LAMBDA_CINEMATIC = 5e7f;

constant int MAX_STEPS_FAST_MOVING = 3000;
constant int MAX_STEPS_FAST_STATIC = 4000;
constant int MAX_STEPS_MEDIUM_MOVING = 6000;
constant int MAX_STEPS_MEDIUM_STATIC = 8000;
constant int MAX_STEPS_CINEMATIC_MOVING = 15000;
constant int MAX_STEPS_CINEMATIC_STATIC = 20000;

constant float D_LAMBDA = QUALITY_CINEMATIC ? D_LAMBDA_CINEMATIC :
                          QUALITY_MEDIUM ? D_LAMBDA_MEDIUM :
                          D_LAMBDA_FAST;

constant int MAX_STEPS_MOVING = QUALITY_CINEMATIC ? MAX_STEPS_CINEMATIC_MOVING :
                                QUALITY_MEDIUM ? MAX_STEPS_MEDIUM_MOVING :
                                MAX_STEPS_FAST_MOVING;

constant int MAX_STEPS_STATIC = QUALITY_CINEMATIC ? MAX_STEPS_CINEMATIC_STATIC :
                                QUALITY_MEDIUM ? MAX_STEPS_MEDIUM_STATIC :
                                MAX_STEPS_FAST_STATIC;

constant float EARLY_ESCAPE_R = SagA_rs * 200.0f;
constant int DEBUG_GRADIENT = 0;

struct CameraUBO {
    float3 camPos;     float _pad0;
    float3 camRight;   float _pad1;
    float3 camUp;      float _pad2;
    float3 camForward; float _pad3;
    float  tanHalfFov;
    float  aspect;
    uint   moving;
    float  time;
};

struct DiskUBO {
    float disk_r1;
    float disk_r2;
    float disk_num;
    float thickness;
};

struct ObjectsUBO {
    int numObjects;
    float _pad0, _pad1, _pad2;
    float4 objPosRadius[16];
    float4 objColor[16];
    float  mass[16];
};

struct GravityObject {
    float4 posRadius;
    float4 color;
    float  mass;
    float3 velocity;
};

struct GravityBuffer {
    int numObjects;
    float _pad0, _pad1, _pad2;
    GravityObject objects[16];
};

// ============================================================================
// FULLSCREEN QUAD SHADERS
// ============================================================================

struct FSOut {
    float4 position [[position]];
    float2 uv;
};

vertex FSOut fullscreenVS(uint vid [[vertex_id]]) {
    float2 pos[3] = { {-1.0, -1.0}, {3.0, -1.0}, {-1.0, 3.0} };
    float2 uv[3]  = { {0.0, 0.0}, {2.0, 0.0}, {0.0, 2.0} };
    FSOut out;
    out.position = float4(pos[vid], 0, 1);
    out.uv = uv[vid];
    return out;
}

fragment float4 fullscreenFS(FSOut in [[stage_in]],
                             texture2d<float> screenTex [[texture(0)]]) {
    // Use linear filtering for smooth upscaling from quarter-res
    constexpr sampler s(address::clamp_to_edge, filter::linear);
    
    // Simple FXAA-style edge smoothing for the raymarched image
    float2 texel = 1.0 / float2(screenTex.get_width(), screenTex.get_height());
    
    // Sample center and neighbors
    float3 c = screenTex.sample(s, in.uv).rgb;
    float3 l = screenTex.sample(s, in.uv + float2(-texel.x, 0)).rgb;
    float3 r = screenTex.sample(s, in.uv + float2( texel.x, 0)).rgb;
    float3 u = screenTex.sample(s, in.uv + float2(0, -texel.y)).rgb;
    float3 d = screenTex.sample(s, in.uv + float2(0,  texel.y)).rgb;
    
    // Edge detection (horizontal and vertical)
    float3 edgeH = fast::abs(l - r);
    float3 edgeV = fast::abs(u - d);
    float edge = fast::max(fast::max(edgeH.r, edgeH.g), edgeH.b);
    edge = fast::max(edge, fast::max(fast::max(edgeV.r, edgeV.g), edgeV.b));
    
    // Apply edge-aware blur
    float aa = smoothstep(0.05, 0.15, edge);
    float3 blur = (l + r + u + d + c) * 0.2;  // 5-tap average
    float3 final = mix(c, blur, aa * 0.5);  // Subtle AA, preserve detail
    
    return float4(final, 1.0);
}

// ============================================================================
// GRAVITY SIMULATION  
// ============================================================================

kernel void gravityKernel(device GravityBuffer& gravityData [[buffer(0)]],
                          uint tid [[thread_position_in_grid]])
{
    if (tid >= uint(gravityData.numObjects)) return;
    device GravityObject& obj = gravityData.objects[tid];
    float3 totalAcceleration = float3(0.0f);
    for (int i = 0; i < gravityData.numObjects; ++i) {
        if (i == int(tid)) continue;
        device GravityObject& obj2 = gravityData.objects[i];
        float3 diff = obj2.posRadius.xyz - obj.posRadius.xyz;
        float distSq = dot(diff, diff);
        if (distSq > 1e-10f) {
            float dist = fast::sqrt(distSq);
            float invDist = 1.0f / dist;
            float3 direction = diff * invDist;
            float acceleration = (G_CONST * obj2.mass) / distSq;
            totalAcceleration += direction * acceleration;
        }
    }
    obj.velocity += totalAcceleration;
    obj.posRadius.xyz += obj.velocity;
}

// ============================================================================
// STARS RENDERING
// ============================================================================

struct StarVSOut {
    float4 position [[position]];
    float  size     [[point_size]];
};

vertex StarVSOut starVS(uint vid [[vertex_id]],
                        const device float3* positions [[buffer(0)]],
                        constant float4x4& viewProj [[buffer(1)]]) {
    StarVSOut out;
    float3 worldPos = positions[vid];
    out.position = viewProj * float4(worldPos, 1.0);
    out.size = 2.0;   // point size in pixels
    return out;
}

fragment float4 starFS() {
    // White star, additive blending in pipeline does the rest
    return float4(1.0, 1.0, 1.0, 1.0);
}

// ============================================================================
// GRID RENDERING
// ============================================================================

struct GridVSOut {
    float4 position [[position]];
};

float computeGridWarp(float3 gridPos, constant ObjectsUBO& objs) {
    float y = 0.0f;
    for (int i = 0; i < objs.numObjects; ++i) {
        float3 objPos = objs.objPosRadius[i].xyz;
        float mass = objs.mass[i];
        float r_s = 2.0f * G_CONST * mass / (c_light * c_light);
        float dx = gridPos.x - objPos.x;
        float dz = gridPos.z - objPos.z;
        float distSq = dx*dx + dz*dz;
        float dist = fast::sqrt(distSq);
        if (dist > r_s) {
            float deltaY = 2.0f * fast::sqrt(r_s * (dist - r_s));
            y += deltaY - 3e10f;
        } else {
            y += 2.0f * r_s - 3e10f;
        }
    }
    return y;
}

vertex GridVSOut gridVS(const device float3* positions [[buffer(0)]],
                        constant float4x4& viewProj [[buffer(1)]],
                        constant ObjectsUBO& objs [[buffer(2)]],
                        uint vid [[vertex_id]]) {
    GridVSOut out;
    float3 gridPos = positions[vid];
    gridPos.y = computeGridWarp(gridPos, objs);
    float4 world = float4(gridPos, 1.0);
    out.position = viewProj * world;
    return out;
}

fragment float4 gridFS(GridVSOut in [[stage_in]]) {
    // Light cyan grid with 60% opacity for alpha blending overlay
    return float4(0.4, 0.8, 1.0, 0.6);
}

// ============================================================================
// GEODESIC RAY TRACING
// ============================================================================

struct Ray {
    float x, y, z, r, theta, phi;
    float dr, dtheta, dphi;
    float E, L;
    float sinTheta, cosTheta, sinPhi, cosPhi;
    float rSq, invR, f;
};

void updateRayCache(thread Ray& ray) {
    ray.rSq = ray.r * ray.r;
    ray.invR = 1.0f / ray.r;
    ray.f = 1.0f - SagA_rs * ray.invR;
}

Ray initRay(float3 pos, float3 dir) {
    Ray ray;
    ray.x = pos.x; ray.y = pos.y; ray.z = pos.z;
    ray.r = fast::length(pos);
    ray.theta = fast::acos(pos.z * (1.0f / ray.r));
    ray.phi = fast::atan2(pos.y, pos.x);
    ray.sinTheta = fast::sin(ray.theta);
    ray.cosTheta = fast::cos(ray.theta);
    ray.sinPhi = fast::sin(ray.phi);
    ray.cosPhi = fast::cos(ray.phi);
    float dx = dir.x, dy = dir.y, dz = dir.z;
    ray.dr = ray.sinTheta*ray.cosPhi*dx + ray.sinTheta*ray.sinPhi*dy + ray.cosTheta*dz;
    ray.dtheta = (ray.cosTheta*ray.cosPhi*dx + ray.cosTheta*ray.sinPhi*dy - ray.sinTheta*dz) * (1.0f / ray.r);
    ray.dphi = (-ray.sinPhi*dx + ray.cosPhi*dy) / (ray.r * ray.sinTheta);
    updateRayCache(ray);
    ray.L = ray.rSq * ray.sinTheta * ray.dphi;
    float dt_dL = fast::sqrt((ray.dr*ray.dr) * (1.0f / ray.f) + ray.rSq*(ray.dtheta*ray.dtheta + ray.sinTheta*ray.sinTheta*ray.dphi*ray.dphi));
    ray.E = ray.f * dt_dL;
    return ray;
}

bool intercept(Ray ray, float rs) { return ray.r <= rs; }

float randomStar(float3 p) {
    return fast::fract(fast::sin(dot(p, float3(12.9898f, 78.233f, 151.7182f))) * 43758.5453f);
}

float4 getStarColor(float3 dir) {
    const float star_density = 0.9995f;
    float r = randomStar(dir);
    if (r > star_density) {
        float star_brightness = (r - star_density) / (1.0f - star_density);
        return float4(star_brightness, star_brightness, star_brightness, 1.0f);
    }
    return float4(0.0f);
}

struct HitInfo {
    float4 objectColor;
    float3 hitCenter;
    float  hitRadius;
};

bool interceptObject(Ray ray, constant ObjectsUBO& objs, thread HitInfo& hit) {
    float3 P = float3(ray.x, ray.y, ray.z);
    for (int i=0; i<objs.numObjects; ++i) {
        float3 center = objs.objPosRadius[i].xyz;
        float radius = objs.objPosRadius[i].w;
        float radiusSq = radius * radius;
        float3 diff = P - center;
        float distSq = dot(diff, diff);
        if (distSq <= radiusSq) {
            hit.objectColor = objs.objColor[i];
            hit.hitCenter = center;
            hit.hitRadius = radius;
            return true;
        }
    }
    return false;
}

void geodesicRHS(Ray ray, thread float3& d1, thread float3& d2) {
    float dr = ray.dr, dtheta = ray.dtheta, dphi = ray.dphi;
    float dt_dL = ray.E * (1.0f / ray.f);
    float sinTheta = ray.sinTheta;
    float cosTheta = ray.cosTheta;
    float sinThetaSq = sinTheta * sinTheta;
    float invRSq = ray.invR * ray.invR;
    float halfRsSq = SagA_rs * 0.5f * invRSq;
    d1 = float3(dr, dtheta, dphi);
    d2.x = - halfRsSq * ray.f * dt_dL * dt_dL + halfRsSq * (1.0f / ray.f) * dr * dr + ray.r * (dtheta*dtheta + sinThetaSq*dphi*dphi);
    d2.y = -2.0f*dr*dtheta*ray.invR + sinTheta*cosTheta*dphi*dphi;
    d2.z = -2.0f*dr*dphi*ray.invR - 2.0f*cosTheta/sinTheta * dtheta * dphi;
}

void rk2Step(thread Ray& ray, float dL) {
    float r0 = ray.r, theta0 = ray.theta, phi0 = ray.phi;
    float dr0 = ray.dr, dtheta0 = ray.dtheta, dphi0 = ray.dphi;
    float3 k1a, k1b;
    geodesicRHS(ray, k1a, k1b);
    float halfDL = dL * 0.5f;
    ray.r = r0 + halfDL * k1a.x;
    ray.theta = theta0 + halfDL * k1a.y;
    ray.phi = phi0 + halfDL * k1a.z;
    ray.dr = dr0 + halfDL * k1b.x;
    ray.dtheta = dtheta0 + halfDL * k1b.y;
    ray.dphi = dphi0 + halfDL * k1b.z;
    ray.sinTheta = fast::sin(ray.theta);
    ray.cosTheta = fast::cos(ray.theta);
    ray.sinPhi = fast::sin(ray.phi);
    ray.cosPhi = fast::cos(ray.phi);
    updateRayCache(ray);
    ray.x = ray.r * ray.sinTheta * ray.cosPhi;
    ray.y = ray.r * ray.sinTheta * ray.sinPhi;
    ray.z = ray.r * ray.cosTheta;
    float3 k2a, k2b;
    geodesicRHS(ray, k2a, k2b);
    ray.r = r0 + dL * k2a.x;
    ray.theta = theta0 + dL * k2a.y;
    ray.phi = phi0 + dL * k2a.z;
    ray.dr = dr0 + dL * k2b.x;
    ray.dtheta = dtheta0 + dL * k2b.y;
    ray.dphi = dphi0 + dL * k2b.z;
    ray.sinTheta = fast::sin(ray.theta);
    ray.cosTheta = fast::cos(ray.theta);
    ray.sinPhi = fast::sin(ray.phi);
    ray.cosPhi = fast::cos(ray.phi);
    updateRayCache(ray);
    ray.x = ray.r * ray.sinTheta * ray.cosPhi;
    ray.y = ray.r * ray.sinTheta * ray.sinPhi;
    ray.z = ray.r * ray.cosTheta;
}

bool crossesEquatorialPlane(float3 oldPos, float3 newPos, constant DiskUBO& disk) {
    bool crossed = (oldPos.y * newPos.y < 0.0);
    if (!crossed) return false;
    float rSq = newPos.x*newPos.x + newPos.z*newPos.z;
    float r2Sq = disk.disk_r2 * disk.disk_r2;
    if (rSq > r2Sq) return false;
    float r = fast::sqrt(rSq);
    return (r >= disk.disk_r1 && r <= disk.disk_r2);
}

kernel void geodesicKernel(texture2d<float, access::write> outImage [[texture(0)]],
                           constant CameraUBO& cam [[buffer(1)]],
                           constant DiskUBO& disk [[buffer(2)]],
                           constant ObjectsUBO& objs [[buffer(3)]],
                           ushort2 tid [[thread_position_in_grid]],
                           ushort2 grid [[threads_per_grid]])
{
    if (tid.x >= grid.x || tid.y >= grid.y) return;
    float u = (2.0 * (float(tid.x) + 0.5) / float(grid.x) - 1.0) * cam.aspect * cam.tanHalfFov;
    float v = (1.0 - 2.0 * (float(tid.y) + 0.5) / float(grid.y)) * cam.tanHalfFov;
    if (DEBUG_GRADIENT) {
        outImage.write(float4(float(tid.x) / float(grid.x), float(tid.y) / float(grid.y), 0.0f, 1.0f), tid);
        return;
    }
    float3 dir = fast::normalize(u * cam.camRight - v * cam.camUp + cam.camForward);
    Ray ray = initRay(cam.camPos, dir);
    half4 color = half4(0.0h);
    float3 prevPos = float3(ray.x, ray.y, ray.z);
    bool hitBH = false, hitDisk = false, hitObj = false;
    HitInfo hitInfo;
    bool isMoving = (cam.moving != 0);
    const int maxSteps = isMoving ? MAX_STEPS_MOVING : MAX_STEPS_STATIC;
    
    // OPTIMIZATION: Direction-based early escape detection
    // If ray is moving radially outward and far from BH, it won't curve back
    float3 initialDir = dir;
    
    for (int i=0; i<maxSteps; ++i) {
        if (intercept(ray, SagA_rs)) { hitBH = true; break; }
        
        float stepScale = fast::clamp(ray.r / (SagA_rs * 10.0f), 0.5f, 10.0f);
        float dynamicStep = D_LAMBDA * stepScale;
        rk2Step(ray, dynamicStep);
        
        float3 newPos = float3(ray.x, ray.y, ray.z);
        
        // OPTIMIZATION: Skip disk check if ray is far above/below disk plane
        // Disk is at y=0, thickness is small compared to radius
        bool nearDiskPlane = fast::abs(newPos.y) < disk.thickness;
        if (nearDiskPlane && crossesEquatorialPlane(prevPos, newPos, disk)) { 
            hitDisk = true; break; 
        }
        
        if (interceptObject(ray, objs, hitInfo)) { hitObj = true; break; }
        
        prevPos = newPos;
        
        // Standard escape
        if (ray.r > ESCAPE_R) break;
        
        // OPTIMIZATION: Direction-based early escape
        // If ray is far from BH and moving radially outward, it's escaping
        if (ray.r > EARLY_ESCAPE_R) {
            float3 currentPos = float3(ray.x, ray.y, ray.z);
            float3 radialDir = fast::normalize(currentPos);
            float dotProduct = dot(radialDir, initialDir);
            
            // If moving outward (dot > 0.95 means angle < ~18°), definitely escaping
            if (dotProduct > 0.95f) break;
        }
        
        // OPTIMIZATION: If well past disk and no hits yet, likely to escape
        if (!hitDisk && !hitObj && ray.r > SagA_rs * 200.0f) break;
    }
    if (hitDisk) {
        float3 hitPos = float3(ray.x, ray.y, ray.z);
        float r_norm = (fast::length(hitPos) - disk.disk_r1) / (disk.disk_r2 - disk.disk_r1);
        r_norm = fast::clamp(r_norm, 0.0f, 1.0f);
        half3 color_hot = half3(1.0h, 1.0h, 0.8h);
        half3 color_mid = half3(1.0h, 0.5h, 0.0h);
        half3 color_cool = half3(0.8h, 0.0h, 0.0h);
        half3 diskColor = mix(color_mid, color_hot, half(smoothstep(0.0f, 0.3f, 1.0f - r_norm)));
        diskColor = mix(color_cool, diskColor, half(smoothstep(0.3f, 1.0f, 1.0f - r_norm)));
        float angle = fast::atan2(hitPos.y, hitPos.x);
        half spiral = half(0.5f + 0.5f * fast::sin(angle * 10.0f - r_norm * 20.0f - cam.time * 0.1f));
        diskColor *= 0.8h + 0.4h * spiral;
        color = half4(diskColor, 1.0h);
    } else if (hitBH) {
        color = half4(0.0h, 0.0h, 0.0h, 1.0h);
    } else if (hitObj) {
        float3 P = float3(ray.x, ray.y, ray.z);
        float3 N = fast::normalize(P - hitInfo.hitCenter);
        float3 V = fast::normalize(cam.camPos - P);
        float3 L = fast::normalize(float3(-1.0f, 1.0f, -1.0f));
        half ambient = 0.5h;
        half diff = half(fast::max(dot(N, L), 0.0f));
        half3 shaded = half3(hitInfo.objectColor.rgb) * (ambient + diff);
        float3 H = fast::normalize(L + V);
        half spec = half(fast::pow(fast::max(dot(N, H), 0.0f), 32.0f));
        half3 specular = half3(1.0h, 1.0h, 1.0h) * spec * 0.5h;
        color = half4(shaded + specular, half(hitInfo.objectColor.a));
    } else {
        color = half4(getStarColor(dir));
    }
    outImage.write(float4(color), tid);
}
