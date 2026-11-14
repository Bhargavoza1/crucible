# crucible

GPU physics engine for robot learning. CUDA C++ physics (**forge**), Vulkan renderer
(**flux**), and a GPU RL stack (**temper**). See the design document (v0.6).

Status: **Phase 1 done** (restructure, headless target, config, tests, Vulkan triangle).

## Requirements (Linux)

| Need | Package (Ubuntu 24.04) | Used by |
|---|---|---|
| CMake 3.24+, g++ | `cmake build-essential` | all |
| CUDA 12.x | NVIDIA CUDA Toolkit | forge kernels, tests_gpu |
| Vulkan 1.2+, glslc | `libvulkan-dev glslc vulkan-validationlayers` | flux (only with `BUILD_FLUX=ON`) |
| GLFW 3.3+ | `libglfw3-dev` | flux |
| GoogleTest | `libgtest-dev` (optional, else cloned from GitHub) | tests |

## Build

```bash
# Headless: no Vulkan needed (training, servers)
cmake -S . -B build -DBUILD_FLUX=OFF
cmake --build build -j

# Window mode: adds flux and sandbox_app
cmake -S . -B build -DBUILD_FLUX=ON
cmake --build build -j
```

Options:

| Option | Values | Meaning |
|---|---|---|
| `BUILD_FLUX` | OFF / ON | Build the Vulkan renderer and `sandbox_app` |
| `FORGE_CUDA` | AUTO / ON / OFF | AUTO uses CUDA if nvcc is found |
| `CMAKE_CUDA_ARCHITECTURES` | default 89 | RTX 4070 Ti |

If nvcc says your g++ is too new (CUDA 12.0 supports up to g++ 12), add
`-DCMAKE_CUDA_HOST_COMPILER=g++-12`.

## Test

```bash
cd build
ctest            # all tests; GPU tests are skipped when there is no GPU
ctest -LE gpu    # CPU tests only
ctest -L gpu     # GPU tests only
compute-sanitizer ./tests/tests_gpu   # now and then: memory errors and races
```

Run `ctest` before every commit.

## Run

```bash
./apps/train_app --task ant --num_envs 4096 --headless --save runs
./apps/train_app --config ../configs/example.txt --lr 1e-4
./apps/sandbox_app                     # window with the triangle
./apps/sandbox_app --frames 300        # exit after 300 frames
```

Flags: `--key value`, `--key=value`, or `--flag` (means true). Flags override
the config file. Unknown keys print a warning (catches typos like `lamda`).

## Layout

```
forge/      physics, CUDA only, never Vulkan
  math/       Vec3, Quat, Mat3: header-only, host + device (FORGE_HD)
  util/       config parser, CUDA_CHECK, DeviceBuffer, device info
flux/       Vulkan renderer: context, window/swapchain, pipeline, shaders
temper/     RL and VLA (Phase 7+)
  nn/         layers (CUTLASS Linear), activations, cuDNN conv
  optim/      Adam, grad clipping, LR schedules
  buffers/    rollout/ (on-policy), replay/ (off-policy)
  common/     GAE, obs normalizer, checkpoints
  algos/      ppo/, sac/, td3/, tqc/
  vla/
apps/       train_app (no flux), sandbox_app, vla_app (later)
configs/    example configs
tests/
  common/     helpers shared by cpu and gpu tests
  cpu/        mirrors the source tree, e.g. cpu/forge/math/test_quat.cpp
  gpu/        mirrors the source tree, files end in _gpu.cu
  data/       reference numbers (MuJoCo)
tools/      scripts, e.g. gen_mujoco_refs.py (Phase 5)
```

Libraries: `forge_cpu` (plain C++, the CPU answer keys), `forge` (forge_cpu + CUDA
kernels), `flux` (Vulkan). A `.cpp` goes in forge_cpu, a `.cu` goes in forge.
