// MetalEngine.mm (Objective-C++ implementation ONLY)
#define GLFW_EXPOSE_NATIVE_COCOA
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#import <Cocoa/Cocoa.h>
#import <GLFW/glfw3.h>
#import <GLFW/glfw3native.h>

#include "MetalEngine.hpp"
#include "../scene/Scene.hpp"
#include <vector>
#include <array>
#include <span>
#include <memory>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <random>
#include <glm/glm.hpp>

// ===== Constants =====
// C++17: inline constexpr for header-safe constants
inline constexpr int MAX_FRAMES_IN_FLIGHT = 3;

// C++11: constexpr functions for compile-time computation
constexpr size_t alignTo256(size_t size) noexcept {
    return (size + 255) & ~size_t(255);
}

// Buffer alignment: 256 bytes for macOS (computed at compile time)
inline constexpr size_t alignedCameraUBOSize  = alignTo256(sizeof(CameraUBO));
inline constexpr size_t alignedDiskUBOSize    = alignTo256(sizeof(DiskUBO));
inline constexpr size_t alignedObjectsUBOSize = alignTo256(sizeof(ObjectsUBO));

// Gravity buffer structure (matches Shaders.metal)
struct GravityObject {
    simd::float4 posRadius;
    simd::float4 color;
    float mass;
    simd::float3 velocity;
};

struct GravityBuffer {
    int numObjects;
    float _pad0, _pad1, _pad2;
    GravityObject objects[16];
};

inline constexpr size_t alignedGravityBufferSize = alignTo256(sizeof(GravityBuffer));

// ================= Impl =================
struct MetalEngine::Impl {
    GLFWwindow* window = nullptr;
    int WIDTH = 800, HEIGHT = 600;
    int COMPUTE_WIDTH = 200, COMPUTE_HEIGHT = 150;

    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;        // Main queue (geodesic + render)
    id<MTLCommandQueue> gravityQueue = nil; // Async compute queue for gravity
    CAMetalLayer* layer = nil;

    id<MTLComputePipelineState> computePSO = nil;
    id<MTLComputePipelineState> gravityPSO = nil;
    id<MTLRenderPipelineState> quadPSO = nil;
    id<MTLRenderPipelineState> gridPSO = nil;
    id<MTLRenderPipelineState> starsPSO = nil;

    // C++11: std::array for type-safe fixed-size buffers
    std::array<id<MTLBuffer>, MAX_FRAMES_IN_FLIGHT> cameraBuf;
    std::array<id<MTLBuffer>, MAX_FRAMES_IN_FLIGHT> diskBuf;
    std::array<id<MTLBuffer>, MAX_FRAMES_IN_FLIGHT> objectsBuf;
    std::array<id<MTLBuffer>, MAX_FRAMES_IN_FLIGHT> gravityBuf;

    id<MTLBuffer> gridVB = nil;
    id<MTLBuffer> gridIB = nil;
    uint32_t gridIndexCount = 0;
    
    id<MTLBuffer> starsVB = nil;
    uint32_t starCount = 0;

    id<MTLTexture> computeTex = nil;
    id<MTLTexture> msaaColorTex = nil;
    NSUInteger sampleCount = 4;  // 4x MSAA for smooth edges
    dispatch_semaphore_t inflight;
    
    // Adaptive quality system - dynamically adjusts render scale for target FPS
    struct AdaptiveQuality {
        float targetFrameTimeMs = 16.67f;  // 60 FPS target
        float currentFrameTimeMs = 20.0f;
        float renderScale = 0.25f;  // Start at quarter res
        int measurementCount = 0;
        std::chrono::high_resolution_clock::time_point lastFrameTime;
        
        AdaptiveQuality() : lastFrameTime(std::chrono::high_resolution_clock::now()) {}
        
        void updateFrameTime() {
            auto now = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - lastFrameTime);
            currentFrameTimeMs = elapsed.count() / 1000.0f;
            lastFrameTime = now;
            measurementCount++;
        }
        
        void adjustQuality() {
            // Only adjust after warmup period
            if (measurementCount < 30) return;
            
            if (currentFrameTimeMs > targetFrameTimeMs * 1.15f) {
                // Too slow, reduce quality
                renderScale = std::max(0.15f, renderScale * 0.95f);
            } else if (currentFrameTimeMs < targetFrameTimeMs * 0.85f) {
                // Headroom, increase quality
                renderScale = std::min(0.5f, renderScale * 1.02f);
            }
        }
        
        std::pair<int, int> getResolution(int width, int height) const {
            return {
                static_cast<int>(width * renderScale),
                static_cast<int>(height * renderScale)
            };
        }
    } adaptiveQuality;

    Impl(int w, int h, int cw, int ch)
    : WIDTH(w), HEIGHT(h), COMPUTE_WIDTH(cw), COMPUTE_HEIGHT(ch)
    {
        // --- GLFW window ---
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window = glfwCreateWindow(WIDTH, HEIGHT, "Black Hole (Metal)", nullptr, nullptr);

        // --- Metal setup ---
        device = MTLCreateSystemDefaultDevice();
        queue  = [device newCommandQueue];  // Main queue
        gravityQueue = [device newCommandQueue];  // Async compute for gravity
        
        // Set queue labels for GPU debugging
        queue.label = @"MainQueue";
        gravityQueue.label = @"GravityQueue";

        // Attach CAMetalLayer to GLFW Cocoa window
        NSWindow* nswin = glfwGetCocoaWindow(window);
        layer = [CAMetalLayer layer];
        layer.device = device;
        layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
        // OPTIMIZATION: framebufferOnly = YES reduces memory traffic on tile-based GPUs
        // We only render to drawable, never read/sample from it
        layer.framebufferOnly = YES;
        nswin.contentView.wantsLayer = YES;
        nswin.contentView.layer = layer;

        inflight = dispatch_semaphore_create(MAX_FRAMES_IN_FLIGHT);

        // C++20: Range-based initialization for cleaner code
        for (auto& buf : cameraBuf) {
            buf = [device newBufferWithLength:alignedCameraUBOSize
                                      options:MTLResourceStorageModeShared];
        }
        for (auto& buf : diskBuf) {
            buf = [device newBufferWithLength:alignedDiskUBOSize
                                      options:MTLResourceStorageModeShared];
        }
        for (auto& buf : objectsBuf) {
            buf = [device newBufferWithLength:alignedObjectsUBOSize
                                      options:MTLResourceStorageModeShared];
        }
        for (auto& buf : gravityBuf) {
            buf = [device newBufferWithLength:alignedGravityBufferSize
                                      options:MTLResourceStorageModeShared];
        }

        buildPipelines();
        rebuildComputeTexture(COMPUTE_WIDTH, COMPUTE_HEIGHT);
        rebuildMSAATexture();
    }

    ~Impl() {
        // Note: GLFW cleanup should be handled by main, but we can clean up Metal resources
    }

    // Pipeline creation methods
    id<MTLLibrary> loadMetalLibrary() {
        NSError* err = nil;
        id<MTLLibrary> lib = nil;
        
        // Try to load from default library first (works if shaders are in app bundle)
        lib = [device newDefaultLibrary];
        
        // If that fails, try loading from build directory
        if (!lib) {
            lib = loadPrecompiledLibrary(&err);
        }
        
        // Runtime compilation: compile from source files (enables instant shader iteration)
        if (!lib) {
            lib = compileLibraryFromSource(&err);
        }
        
        if (!lib) {
            NSLog(@"❌ Failed to load Metal library from any source.");
            NSLog(@"   Checked: bundle, build directory, and runtime compilation.");
            abort();
        }
        
        return lib;
    }
    
    id<MTLLibrary> loadPrecompiledLibrary(NSError** err) {
        NSString* exePath = [[NSBundle mainBundle] executablePath];
        if (!exePath) return nil;
        
        NSString* exeDir = [exePath stringByDeletingLastPathComponent];
        NSString* metallibPath = [exeDir stringByAppendingPathComponent:@"default.metallib"];
        id<MTLLibrary> lib = [device newLibraryWithFile:metallibPath error:err];
        
        if (lib) {
            NSLog(@"✓ Loaded precompiled Metal library (fast startup)");
        }
        return lib;
    }
    
    id<MTLLibrary> compileLibraryFromSource(NSError** err) {
        NSLog(@"⚠ Compiling shaders from source at runtime...");
        // MetalEngine.mm is in src/engine/, shaders are in src/shaders/
        NSString* engineDir = [[NSString stringWithUTF8String:__FILE__] stringByDeletingLastPathComponent];
        NSString* srcDir = [engineDir stringByDeletingLastPathComponent];
        NSString* shadersPath = [[srcDir stringByAppendingPathComponent:@"shaders"] 
                                  stringByAppendingPathComponent:@"shaders.metal"];
        
        NSString* source = [NSString stringWithContentsOfFile:shadersPath encoding:NSUTF8StringEncoding error:err];
        if (!source) {
            NSLog(@"❌ Failed to load shader file %@: %@", shadersPath, *err);
            return nil;
        }
        
        id<MTLLibrary> lib = [device newLibraryWithSource:source options:nil error:err];
        if (!lib) {
            NSLog(@"❌ Failed to compile Metal shaders from source: %@", *err);
        }
        return lib;
    }
    
    id<MTLComputePipelineState> createGeodesicPipeline(id<MTLLibrary> lib) {
        NSError* err = nil;
        
        // Quality preset selection (use MEDIUM by default for balanced perf/quality)
        MTLFunctionConstantValues* constants = [MTLFunctionConstantValues new];
        int qualityFast = 0;
        int qualityMedium = 1;  // Default
        int qualityCinematic = 0;
        [constants setConstantValue:&qualityFast type:MTLDataTypeInt atIndex:0];
        [constants setConstantValue:&qualityMedium type:MTLDataTypeInt atIndex:1];
        [constants setConstantValue:&qualityCinematic type:MTLDataTypeInt atIndex:2];
        
        id<MTLFunction> k = [lib newFunctionWithName:@"geodesicKernel" constantValues:constants error:&err];
        if (!k) { 
            NSLog(@"Failed to create geodesicKernel with constants: %@", err); 
            abort(); 
        }
        
        MTLComputePipelineDescriptor* computeDesc = [MTLComputePipelineDescriptor new];
        computeDesc.computeFunction = k;
        computeDesc.threadGroupSizeIsMultipleOfThreadExecutionWidth = YES;
        
        MTLComputePipelineReflection* reflection = nil;
        id<MTLComputePipelineState> pso = [device newComputePipelineStateWithDescriptor:computeDesc
                                                                                  options:MTLPipelineOptionArgumentInfo
                                                                               reflection:&reflection
                                                                                    error:&err];
        if (!pso) { 
            NSLog(@"Geodesic PSO error %@", err); 
            abort(); 
        }
        return pso;
    }
    
    id<MTLComputePipelineState> createGravityPipeline(id<MTLLibrary> lib) {
        NSError* err = nil;
        id<MTLFunction> gravityFunc = [lib newFunctionWithName:@"gravityKernel"];
        
        MTLComputePipelineDescriptor* gravityDesc = [MTLComputePipelineDescriptor new];
        gravityDesc.computeFunction = gravityFunc;
        gravityDesc.threadGroupSizeIsMultipleOfThreadExecutionWidth = YES;
        
        id<MTLComputePipelineState> pso = [device newComputePipelineStateWithDescriptor:gravityDesc
                                                                                  options:0
                                                                               reflection:nil
                                                                                    error:&err];
        if (!pso) { 
            NSLog(@"Gravity PSO error %@", err); 
            abort(); 
        }
        return pso;
    }
    
    id<MTLRenderPipelineState> createQuadPipeline(id<MTLLibrary> lib) {
        NSError* err = nil;
        MTLRenderPipelineDescriptor* qd = [MTLRenderPipelineDescriptor new];
        qd.vertexFunction   = [lib newFunctionWithName:@"fullscreenVS"];
        qd.fragmentFunction = [lib newFunctionWithName:@"fullscreenFS"];
        qd.colorAttachments[0].pixelFormat = layer.pixelFormat;
        qd.rasterSampleCount = sampleCount;  // Enable MSAA
        
        id<MTLRenderPipelineState> pso = [device newRenderPipelineStateWithDescriptor:qd error:&err];
        if (!pso) { 
            NSLog(@"Quad PSO error %@", err); 
            abort(); 
        }
        return pso;
    }
    
    id<MTLRenderPipelineState> createGridPipeline(id<MTLLibrary> lib) {
        NSError* err = nil;
        MTLRenderPipelineDescriptor* gd = [MTLRenderPipelineDescriptor new];
        gd.vertexFunction   = [lib newFunctionWithName:@"gridVS"];
        gd.fragmentFunction = [lib newFunctionWithName:@"gridFS"];
        gd.colorAttachments[0].pixelFormat = layer.pixelFormat;
        gd.depthAttachmentPixelFormat = MTLPixelFormatInvalid;
        gd.rasterSampleCount = sampleCount;  // Enable MSAA
        
        // Enable alpha blending so grid overlays the black hole image
        MTLRenderPipelineColorAttachmentDescriptor* ca = gd.colorAttachments[0];
        ca.blendingEnabled = YES;
        ca.rgbBlendOperation = MTLBlendOperationAdd;
        ca.alphaBlendOperation = MTLBlendOperationAdd;
        ca.sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
        ca.sourceAlphaBlendFactor = MTLBlendFactorSourceAlpha;
        ca.destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        ca.destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        
        id<MTLRenderPipelineState> pso = [device newRenderPipelineStateWithDescriptor:gd error:&err];
        if (!pso) { 
            NSLog(@"Grid PSO error %@", err); 
            abort(); 
        }
        return pso;
    }
    
    id<MTLRenderPipelineState> createStarsPipeline(id<MTLLibrary> lib) {
        NSError* err = nil;
        MTLRenderPipelineDescriptor* sd = [MTLRenderPipelineDescriptor new];
        sd.vertexFunction   = [lib newFunctionWithName:@"starVS"];
        sd.fragmentFunction = [lib newFunctionWithName:@"starFS"];
        sd.colorAttachments[0].pixelFormat = layer.pixelFormat;
        sd.depthAttachmentPixelFormat = MTLPixelFormatInvalid;
        sd.rasterSampleCount = sampleCount;  // Enable MSAA
        
        // Additive blending for stars
        MTLRenderPipelineColorAttachmentDescriptor* ca = sd.colorAttachments[0];
        ca.blendingEnabled = YES;
        ca.rgbBlendOperation = MTLBlendOperationAdd;
        ca.alphaBlendOperation = MTLBlendOperationAdd;
        ca.sourceRGBBlendFactor = MTLBlendFactorOne;
        ca.sourceAlphaBlendFactor = MTLBlendFactorOne;
        ca.destinationRGBBlendFactor = MTLBlendFactorOne;
        ca.destinationAlphaBlendFactor = MTLBlendFactorOne;
        
        id<MTLRenderPipelineState> pso = [device newRenderPipelineStateWithDescriptor:sd error:&err];
        if (!pso) {
            NSLog(@"Stars PSO error %@", err);
            abort();
        }
        return pso;
    }
    
    void buildPipelines() {
        // Load Metal shader library
        id<MTLLibrary> lib = loadMetalLibrary();
        
        // Create all pipeline state objects
        computePSO = createGeodesicPipeline(lib);
        gravityPSO = createGravityPipeline(lib);
        quadPSO = createQuadPipeline(lib);
        gridPSO = createGridPipeline(lib);
        starsPSO = createStarsPipeline(lib);
        
        // Generate static geometry
        generateStaticGrid();
        generateStars();
    }

    void rebuildComputeTexture(int w, int h) {
        MTLTextureDescriptor* td =
          [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                            width:w height:h mipmapped:NO];
        td.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;
        // Optimization: Use Private storage mode for GPU-only texture
        td.storageMode = MTLStorageModePrivate;
        td.textureType = MTLTextureType2D;
        computeTex = [device newTextureWithDescriptor:td];
        computeTex.label = @"ComputeTexture";
    }
    
    void rebuildMSAATexture() {
        // Create MSAA color target for smooth rasterization
        MTLTextureDescriptor* td =
          [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:layer.pixelFormat
                                                            width:WIDTH
                                                           height:HEIGHT
                                                        mipmapped:NO];
        td.sampleCount = sampleCount;
        td.storageMode = MTLStorageModePrivate;
        td.textureType = MTLTextureType2DMultisample;
        td.usage = MTLTextureUsageRenderTarget;
        
        msaaColorTex = [device newTextureWithDescriptor:td];
        msaaColorTex.label = @"MSAAColor";
    }

    void uploadCamera(const Camera& cam, int frameIndex, int computeWidth, int computeHeight) {
        CameraUBO* data = (CameraUBO*)cameraBuf[frameIndex].contents;
        simd_float3 fwd = simd_normalize(simd_float3{cam.target.x, cam.target.y, cam.target.z} - simd_float3{cam.position().x, cam.position().y, cam.position().z});
        simd_float3 up  = {0,1,0};
        simd_float3 right = simd_normalize(simd_cross(fwd, up));
        up = simd_cross(right, fwd);

        glm::vec3 pos = cam.position();
        data->camPos     = simd_float3{pos.x, pos.y, pos.z};
        data->camRight   = right;
        data->camUp      = up;
        data->camForward = fwd;
        data->tanHalfFov = tanf(glm::radians(60.0f * 0.5f));
        // CRITICAL: Aspect ratio must match compute texture dimensions, not window dimensions
        // The shader uses compute texture size to calculate ray directions
        data->aspect     = float(computeWidth)/float(computeHeight);
        data->moving     = (cam.dragging || cam.panning) ? 1u : 0u;
        data->time       = static_cast<float>(glfwGetTime());  // For animated effects
    }

    void uploadDisk(int frameIndex) {
        DiskUBO* d = (DiskUBO*)diskBuf[frameIndex].contents;
        // Disk parameters matching C implementation
        // Note: Black hole Schwarzschild radius from scene
        float r_s = scene.blackHole().r_s;
        d->disk_r1 = r_s * 2.2f;   // From C: BLACK_HOLE_SCHWARZSCHILD_RADIUS * 2.2f
        d->disk_r2 = r_s * 5.2f;   // From C: BLACK_HOLE_SCHWARZSCHILD_RADIUS * 5.2f
        d->disk_num = 2.0f;
        d->thickness = 1e9f;
    }

    void uploadObjects(const std::vector<ObjectData>& objs, int frameIndex) {
        auto* o = static_cast<ObjectsUBO*>(objectsBuf[frameIndex].contents);
        
        // C++17: constexpr for compile-time limit
        constexpr size_t maxObjects = 16;
        const size_t count = std::min(objs.size(), maxObjects);
        
        o->numObjects = static_cast<int>(count);
        
        // C++20: Could use std::span<const ObjectData> for safer iteration
        for (size_t i = 0; i < count; ++i) {
            o->objPosRadius[i] = objs[i].posRadius;
            o->objColor[i]     = objs[i].color;
            o->mass[i]         = objs[i].mass;
        }
    }

    void uploadGravityData(const std::vector<ObjectData>& objs, int frameIndex) {
        auto* g = static_cast<GravityBuffer*>(gravityBuf[frameIndex].contents);
        constexpr size_t maxObjects = 16;
        const size_t count = std::min(objs.size(), maxObjects);
        
        g->numObjects = static_cast<int>(count);
        
        // C++11: Range-based loop with auto& to avoid copies
        for (size_t i = 0; i < count; ++i) {
            const auto& obj = objs[i];
            g->objects[i].posRadius = obj.posRadius;
            g->objects[i].color = obj.color;
            g->objects[i].mass = obj.mass;
            g->objects[i].velocity = simd::float3{obj.velocity.x, obj.velocity.y, obj.velocity.z};
        }
    }

    void downloadGravityData(std::vector<ObjectData>& objs, int frameIndex) {
        const auto* g = static_cast<const GravityBuffer*>(gravityBuf[frameIndex].contents);
        const size_t count = std::min(static_cast<size_t>(g->numObjects), objs.size());
        
        for (size_t i = 0; i < count; ++i) {
            auto& obj = objs[i];  // C++11: auto& for cleaner code
            obj.posRadius = g->objects[i].posRadius;
            const auto& vel = g->objects[i].velocity;  // Avoid multiple member accesses
            obj.velocity = glm::vec3(vel.x, vel.y, vel.z);
        }
    }

    void dispatchGravity(id<MTLComputeCommandEncoder> ce, int frameIndex) {
        [ce setComputePipelineState:gravityPSO];
        [ce setBuffer:gravityBuf[frameIndex] offset:0 atIndex:0];
        
        // Get buffer to determine number of objects
        GravityBuffer* g = (GravityBuffer*)gravityBuf[frameIndex].contents;
        int numObjects = g->numObjects;
        
        if (numObjects <= 0) return;
        
        // Dispatch one thread per object
        NSUInteger tew = gravityPSO.threadExecutionWidth;
        MTLSize threadsPerThreadgroup = MTLSizeMake(tew, 1, 1);
        MTLSize threadsPerGrid = MTLSizeMake(numObjects, 1, 1);
        [ce dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
    }

    void generateStaticGrid() {
        // Generate unwarped grid once - warp computed in vertex shader
        const int gridSize = 25;
        const float spacing = 1e10f;

        std::vector<simd_float3> verts;
        std::vector<uint32_t> idx;

        verts.reserve((gridSize+1)*(gridSize+1));
        idx.reserve(gridSize*gridSize*4);

        // Generate flat grid (y=0) - warp computed on GPU
        for (int z=0; z<=gridSize; ++z) {
            for (int x=0; x<=gridSize; ++x) {
                float worldX = (x - gridSize/2)*spacing;
                float worldZ = (z - gridSize/2)*spacing;
                verts.push_back(simd_float3{worldX, 0.0f, worldZ});
            }
        }

        for (int z=0; z<gridSize; ++z){
            for (int x=0; x<gridSize; ++x){
                int i = z*(gridSize+1)+x;
                idx.push_back(i); idx.push_back(i+1);
                idx.push_back(i); idx.push_back(i+gridSize+1);
            }
        }

        gridIndexCount = (uint32_t)idx.size();
        
        // Staging buffers in shared memory (CPU visible)
        id<MTLBuffer> stagingVB = [device newBufferWithBytes:verts.data()
                                                       length:verts.size() * sizeof(simd_float3)
                                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> stagingIB = [device newBufferWithBytes:idx.data()
                                                       length:idx.size() * sizeof(uint32_t)
                                                      options:MTLResourceStorageModeShared];
        
        // Final GPU-only buffers (better bandwidth and compression)
        gridVB = [device newBufferWithLength:stagingVB.length
                                     options:MTLResourceStorageModePrivate];
        gridIB = [device newBufferWithLength:stagingIB.length
                                     options:MTLResourceStorageModePrivate];
        
        // Upload via blit encoder
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        cb.label = @"GridUpload";
        id<MTLBlitCommandEncoder> be = [cb blitCommandEncoder];
        [be copyFromBuffer:stagingVB sourceOffset:0
                  toBuffer:gridVB destinationOffset:0
                      size:stagingVB.length];
        [be copyFromBuffer:stagingIB sourceOffset:0
                  toBuffer:gridIB destinationOffset:0
                      size:stagingIB.length];
        [be endEncoding];
        [cb commit];
        [cb waitUntilCompleted];  // One-time cost at startup is fine
    }
    
    void generateStars() {
        constexpr uint32_t N = 2000;        // C++11: constexpr for compile-time constant
        constexpr float radiusMin = 5e11f;  
        constexpr float radiusMax = 5e13f;  
        constexpr float radiusRange = radiusMax - radiusMin;  // Precompute at compile time
        
        std::vector<simd_float3> positions;
        positions.reserve(N);  // C++11: Reserve to avoid reallocations
        
        std::mt19937 rng(1337);
        std::uniform_real_distribution<float> u01(0.0f, 1.0f);
        
        // Generate random points in spherical coordinates
        for (uint32_t i = 0; i < N; ++i) {
            const float theta = 2.0f * M_PI * u01(rng);
            const float phi   = acosf(2.0f * u01(rng) - 1.0f);
            const float r     = radiusMin + radiusRange * u01(rng);
            
            // C++11: Cache sin/cos computations
            const float sinPhi = sinf(phi);
            const float cosPhi = cosf(phi);
            const float sinTheta = sinf(theta);
            const float cosTheta = cosf(theta);
            
            // C++11: emplace_back for in-place construction (no temporary)
            positions.emplace_back(simd_float3{
                r * sinPhi * cosTheta,
                r * cosPhi,
                r * sinPhi * sinTheta
            });
        }
        
        starCount = static_cast<uint32_t>(positions.size());
        
        // OPTIMIZATION: Use Private storage for GPU-only star data (same as grid)
        id<MTLBuffer> stagingStars = [device newBufferWithBytes:positions.data()
                                                          length:positions.size() * sizeof(simd_float3)
                                                         options:MTLResourceStorageModeShared];
        
        starsVB = [device newBufferWithLength:stagingStars.length
                                      options:MTLResourceStorageModePrivate];
        
        // Upload via blit encoder
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        cb.label = @"StarsUpload";
        id<MTLBlitCommandEncoder> be = [cb blitCommandEncoder];
        [be copyFromBuffer:stagingStars sourceOffset:0
                  toBuffer:starsVB destinationOffset:0
                      size:stagingStars.length];
        [be endEncoding];
        [cb commit];
        [cb waitUntilCompleted];  // One-time cost at startup is fine
    }

    void dispatchCompute(id<MTLComputeCommandEncoder> ce, int frameIndex, int cw, int ch) {
        // Rebuild compute texture if resolution changed
        if (static_cast<int>(computeTex.width) != cw || static_cast<int>(computeTex.height) != ch) {
            rebuildComputeTexture(cw, ch);
        }

        [ce setComputePipelineState:computePSO];
        [ce setTexture:computeTex atIndex:0];
        [ce setBuffer:cameraBuf[frameIndex]  offset:0 atIndex:1];
        [ce setBuffer:diskBuf[frameIndex]    offset:0 atIndex:2];
        [ce setBuffer:objectsBuf[frameIndex] offset:0 atIndex:3];

        // Optimization: Choose threadgroup size based on SIMD width
        NSUInteger tew = computePSO.threadExecutionWidth;             // SIMD width (typically 32)
        NSUInteger maxTG = computePSO.maxTotalThreadsPerThreadgroup;  // Max threads per group
        
        // Optimal layout: tgW = threadExecutionWidth, tgH = maxThreads / tgW
        // On Apple GPUs: tew=32, maxTG/tew=8 → 32x8 = 256 threads
        NSUInteger tgW = tew;
        NSUInteger tgH = maxTG / tgW;
        tgH = std::min<NSUInteger>(tgH, 8);  // Keep it cache-friendly
        
        MTLSize threadsPerThreadgroup = MTLSizeMake(tgW, tgH, 1);
        
        // NEW: dispatch by threads, let Metal handle edge threadgroups
        MTLSize threadsPerGrid = MTLSizeMake(cw, ch, 1);
        [ce dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
    }

    void drawGrid(id<MTLRenderCommandEncoder> re, const simd_float4x4& viewProj, int frameIndex) {
        // Safety check: Don't draw if grid buffers don't exist or are empty
        if (!gridVB || !gridIB || gridIndexCount == 0) {
            return;
        }
        
        [re setRenderPipelineState:gridPSO];
        [re setVertexBuffer:gridVB offset:0 atIndex:0];
        // Optimization: Use setBytes for small constant data (viewProj matrix)
        [re setVertexBytes:&viewProj length:sizeof(viewProj) atIndex:1];
        // Pass objects buffer for GPU warp computation
        [re setVertexBuffer:objectsBuf[frameIndex] offset:0 atIndex:2];
        [re drawIndexedPrimitives:MTLPrimitiveTypeLine
                       indexCount:gridIndexCount
                        indexType:MTLIndexTypeUInt32
                      indexBuffer:gridIB
                indexBufferOffset:0];
    }
    
    void drawStars(id<MTLRenderCommandEncoder> re, const simd_float4x4& viewProj) {
        if (!starsVB || starCount == 0) return;
        
        [re setRenderPipelineState:starsPSO];
        [re setVertexBuffer:starsVB offset:0 atIndex:0];
        [re setVertexBytes:&viewProj length:sizeof(viewProj) atIndex:1];
        [re drawPrimitives:MTLPrimitiveTypePoint
                vertexStart:0
                vertexCount:starCount];
    }

    void drawFullscreen(id<MTLRenderCommandEncoder> re) {
        [re setRenderPipelineState:quadPSO];
        [re setFragmentTexture:computeTex atIndex:0];
        // fullscreen triangle uses no VB
        [re drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    }

    void frame(const Camera& cam, std::vector<ObjectData>& objs,
               const simd_float4x4& viewProj, bool gravityEnabled) {
        dispatch_semaphore_wait(inflight, DISPATCH_TIME_FOREVER);
        static int frameIndex = 0;
        frameIndex = (frameIndex + 1) % MAX_FRAMES_IN_FLIGHT;

        // ADAPTIVE QUALITY: Dynamically adjust resolution to maintain target FPS
        adaptiveQuality.updateFrameTime();
        adaptiveQuality.adjustQuality();
        auto [cw, ch] = adaptiveQuality.getResolution(WIDTH, HEIGHT);
        
        #ifdef VERBOSE_PERF
        static int logCounter = 0;
        if (++logCounter % 60 == 0) {  // Log every 60 frames
            NSLog(@"[DISPATCH] Resolution: %dx%d | Pixels: %d | Mode: %s", 
                  cw, ch, cw*ch, cam.moving ? "MOVING" : "STILL");
        }
        #endif
        
        uploadCamera(cam, frameIndex, cw, ch);
        uploadDisk(frameIndex);
        uploadObjects(objs, frameIndex);
        
        // Optimization: Delay drawable acquisition until just before render pass
        // This reduces GPU/CPU synchronization stalls
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        
        // Optimization: Add labels for GPU debugging and profiling
        cb.label = @"BlackHoleFrame";
        
        __block dispatch_semaphore_t sema = inflight;
        // Optimization: Keep completion handler minimal for better performance
        [cb addCompletedHandler:^(id<MTLCommandBuffer> cbCompleted) {
            dispatch_semaphore_signal(sema);
            
            // Performance measurement - uncomment to see GPU timing
            #ifdef VERBOSE_PERF
            double gpuTime = (cbCompleted.GPUEndTime - cbCompleted.GPUStartTime) * 1000.0;
            NSLog(@"[GPU] Frame time: %.2f ms (%.1f FPS)", gpuTime, 1000.0/gpuTime);
            #endif
            (void)cbCompleted;  // Mark as used to suppress warning when VERBOSE_PERF is not defined
        }];

        // --- ASYNC gravity compute (runs in parallel on separate queue) ---
        // MULTITHREADING: Gravity runs on dedicated compute queue
        // This allows gravity to overlap with next frame's geodesic compute
        static int gravityFrameCounter = 0;
        bool syncGravityThisFrame = false;
        
        if (gravityEnabled) {
            uploadGravityData(objs, frameIndex);
            
            // Fire gravity compute on separate async queue
            id<MTLCommandBuffer> gravityCB = [gravityQueue commandBuffer];
            gravityCB.label = @"AsyncGravity";
            
            id<MTLComputeCommandEncoder> gce = [gravityCB computeCommandEncoder];
            gce.label = @"GravityCompute";
            dispatchGravity(gce, frameIndex);
            [gce endEncoding];
            
            // Only sync every 10th frame to reduce CPU/GPU stalls by 10x
            gravityFrameCounter++;
            if ((gravityFrameCounter % 10) == 0) {
                syncGravityThisFrame = true;
                id<MTLBlitCommandEncoder> syncBlit = [gravityCB blitCommandEncoder];
                [syncBlit synchronizeResource:gravityBuf[frameIndex]];
                [syncBlit endEncoding];
            }
            
            // Commit async - don't wait! Runs in parallel with main rendering
            [gravityCB commit];
            
            // Store command buffer for later sync
            static id<MTLCommandBuffer> lastGravityCB = nil;
            if (syncGravityThisFrame && lastGravityCB) {
                [lastGravityCB waitUntilCompleted];
                downloadGravityData(objs, frameIndex);
            }
            lastGravityCB = syncGravityThisFrame ? gravityCB : nil;
        }

        // --- geodesic compute pass ---
        id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
        // Optimization: Label encoder for GPU debugging
        ce.label = @"GeodesicCompute";
        dispatchCompute(ce, frameIndex, cw, ch);
        [ce endEncoding];

        // --- render pass ---
        // Optimization: Get drawable as late as possible to avoid stalls
        id<CAMetalDrawable> drawable = [layer nextDrawable];
        if (!drawable) {
            [cb commit];
            return;
        }

        MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
        // Render to MSAA texture, resolve to drawable
        rp.colorAttachments[0].texture = msaaColorTex;
        rp.colorAttachments[0].resolveTexture = drawable.texture;
        rp.colorAttachments[0].loadAction  = MTLLoadActionClear;
        rp.colorAttachments[0].storeAction = MTLStoreActionMultisampleResolve;
        rp.colorAttachments[0].clearColor = MTLClearColorMake(0,0,0,1);

        id<MTLRenderCommandEncoder> re = [cb renderCommandEncoderWithDescriptor:rp];
        re.label = @"RenderPass";
        
        // Render order matters (no depth buffer):
        // 1. Black hole raymarched image (full screen)
        // 2. Starfield overlay (additive blending)
        // 3. Grid lines overlay (alpha blending)
        drawFullscreen(re);
        drawStars(re, viewProj);
        drawGrid(re, viewProj, frameIndex);
        [re endEncoding];

        [cb presentDrawable:drawable];
        [cb commit];
        
        // Note: Gravity results are synced async in the gravity compute section above
        // No blocking wait here - main rendering continues immediately
        
        // Adaptive quality: Track frame time for dynamic adjustment
        adaptiveQuality.updateFrameTime();
        
        #ifdef ADAPTIVE_QUALITY
        // Log quality adjustments
        static int qualityLogCounter = 0;
        if (++qualityLogCounter % 120 == 0) {  // Every 2 seconds
            if (adaptiveQuality.shouldReduceQuality()) {
                NSLog(@"[ADAPTIVE] Frame time %.2fms > target %.2fms - consider reducing quality",
                      adaptiveQuality.currentFrameTimeMs, adaptiveQuality.targetFrameTimeMs);
            } else if (adaptiveQuality.shouldIncreaseQuality()) {
                NSLog(@"[ADAPTIVE] Frame time %.2fms < target %.2fms - could increase quality",
                      adaptiveQuality.currentFrameTimeMs, adaptiveQuality.targetFrameTimeMs);
            }
        }
        #endif
    }
};

// ============================================================================
// Public C++ Interface (CRTP Implementation)
// ============================================================================

MetalEngine::MetalEngine(int w, int h, int cw, int ch)
    : impl_(std::make_unique<Impl>(w, h, cw, ch)) {}

// C++11: Default destructor in .mm for unique_ptr with incomplete type
MetalEngine::~MetalEngine() = default;

// CRTP Implementation methods (called by base class)
GLFWwindow* MetalEngine::windowImpl() const { 
    return impl_->window; 
}

int MetalEngine::widthImpl() const { 
    return impl_->WIDTH; 
}

int MetalEngine::heightImpl() const { 
    return impl_->HEIGHT; 
}

int MetalEngine::computeWidthImpl() const { 
    return impl_->COMPUTE_WIDTH; 
}

int MetalEngine::computeHeightImpl() const { 
    return impl_->COMPUTE_HEIGHT; 
}

void MetalEngine::frameImpl(const Camera& cam,
                            std::vector<ObjectData>& objs,
                            const simd::float4x4& viewProj,
                            bool gravityEnabled) {
    impl_->frame(cam, objs, viewProj, gravityEnabled);
}

void MetalEngine::setQualityPresetImpl(int preset) {
    // Quality presets are baked into pipeline at creation time
    // Would need to rebuild pipeline to change - just log for now
    const char* presetName = preset == 0 ? "FAST" :
                             preset == 1 ? "MEDIUM" :
                             preset == 2 ? "CINEMATIC" : "UNKNOWN";
    NSLog(@"[MetalEngine] Quality preset: %s (requires pipeline rebuild)", presetName);
}
