# Black Hole Raymarcher

Real-time black hole visualization using GPU-accelerated geodesic raymarching on Apple Metal. Features physically-accurate Schwarzschild metric integration, accretion disk rendering, N-body gravity simulation, and procedurally-generated starfield.

![Platform](https://img.shields.io/badge/platform-macOS-blue) ![C++](https://img.shields.io/badge/C%2B%2B-20-blue) ![Metal](https://img.shields.io/badge/Metal-3.0-orange)

## Features

- **Geodesic Raymarching**: Schwarzschild metric integration with RK2 solver
- **Accretion Disk**: Spiral-patterned disk with Phong shading
- **Starfield**: 3D point cloud (~2000 stars) with additive blending
- **Grid Overlay**: Reference grid for spatial orientation
- **Anti-Aliasing**: 4× MSAA + FXAA-style post-processing
- **N-Body Gravity**: GPU-accelerated simulation (up to 16 objects)
- **Performance**: Dynamic resolution, triple buffering, async compute
- **Architecture**: CRTP design with modern C++20

## Building

**Prerequisites**: macOS 10.15+, Xcode Command Line Tools, CMake 3.21+, GLFW3, GLM

```bash
mkdir build && cd build
cmake ..
cmake --build .
./BlackHole3D
```

Build automatically compiles Metal shaders and code signs the executable.

## Controls

- **Mouse**: Left/Middle drag = rotate, Right hold = enable gravity, Scroll = zoom
- **Keyboard**: `G` = toggle gravity, `1/2/3` = quality preset info

## Performance

Apple Silicon (M1/M2): 30-50 FPS (still), 12-20 FPS (moving) at quarter-res compute.

Quality presets: FAST (~100 FPS, 3-4k steps), MEDIUM (~50 FPS, 6-8k steps, default), CINEMATIC (~20 FPS, 15-20k steps).

## Project Structure

```
src/
├── main.cpp              # Entry point
├── engine/               # RenderEngine (CRTP), MetalEngine
├── scene/                # Camera, Scene, InputHandler
├── shaders/              # All Metal shaders
└── resources/            # Build resources
```

## Known Issues

- **Quality presets**: Baked into pipeline (requires rebuild to change)
- **Metal API**: Some deprecated methods (warnings only)
- **Single backend**: Metal only (CRTP supports multiple backends)
- **Gravity sync**: Updates every 10th frame (slight delay)

## Future Improvements

- Dynamic quality preset switching
- Temporal Anti-Aliasing (TAA)
- Additional backends (OpenGL/Vulkan)
- Kerr metric (rotating black hole)
- UI improvements (FPS counter, preset selector)
