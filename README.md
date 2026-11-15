# crucible

GPU physics engine for robot learning. CUDA C++ physics (**forge**), Vulkan renderer
(**flux**), and a GPU RL stack (**temper**). See the design document (v0.6).

Status: **Phase 6** done (constraint solver: joint limits, floor contacts with friction, PD
actuators, Newton; matched to MuJoCo; on the GPU one warp per env; shown in the sandbox). Phase 6a: joint limits, sphere-floor contacts with friction, PD actuators, Newton
solver, matched to MuJoCo.

## Requirements (Linux)

| Need | Package (Ubuntu 24.04) | Used by |
|---|---|---|
| CMake 3.24+, g++ | `cmake build-essential` | all |
| CUDA 12.x | NVIDIA CUDA Toolkit | forge kernels, tests_gpu |
| Vulkan 1.2+, glslc | `libvulkan-dev glslc vulkan-validationlayers` | flux (only with `BUILD_FLUX=ON`) |
| GLFW 3.3+ | `libglfw3-dev` | flux |
| GoogleTest | `libgtest-dev` (optional, else cloned from GitHub) | tests |
| tinyxml2 | `libtinyxml2-dev` (optional, else cloned from GitHub) | URDF reader |
| MuJoCo (Python) | `pip install mujoco` (only to regenerate test data) | tools/gen_mujoco_refs.py |

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
./apps/sandbox_app                     # 10,000 particles fall into a bin (GPU if found)
./apps/sandbox_app --scene rain        # free fall, no collisions
./apps/sandbox_app --envs 16 --particles 2000   # 16 envs, one bin each
./apps/sandbox_app --timeconst 0.008 --friction 0.2   # softer, more slippery contacts
./apps/sandbox_app --scene robot       # 16 hinge arms swinging into their joint limits
./apps/sandbox_app --scene robot --robot ../assets/robots/quadruped.urdf --floating true   # standing (floor + PD)
./apps/sandbox_app --scene robot --robot ../assets/robots/arm3.urdf     # slider on its rail, stops at its limits
./apps/sandbox_app --backend cpu       # CPU reference integrator
./apps/sandbox_app --interop off       # GPU physics, copy via host (Stage A)
./apps/sandbox_app --vsync false       # no 60 FPS cap, to compare speed
./apps/sandbox_app --frames 300        # exit after 300 frames
```

Drawing paths: **Stage B** (default with a GPU) shares memory between CUDA and
Vulkan, no copy. **Stage A** copies GPU -> host -> Vulkan; used by the CPU backend
and as a fallback. The sandbox prints which one it uses.

MuJoCo reference data (only when a test robot in assets/robots changes; the
tests read the saved files, so ctest never needs MuJoCo):

```bash
python3 tools/gen_mujoco_refs.py
```

Benchmark (headless, no real-time clock; settles a pile, then times substeps):

```bash
./benchmarks/bench_particles                        # 100k particles, both layouts
./benchmarks/bench_particles --particles 10000 --bin_half 1.0
./benchmarks/bench_robots --envs_list 256,1024,4096,16384   # env-steps per second
./benchmarks/bench_robots --constraints --envs_list 1024,4096,16384   # with contacts + PD
```

Profile (Nsight Systems):

```bash
nsys profile -t cuda,nvtx,vulkan -o phase3 ./apps/sandbox_app --frames 600 --reset_every 1000
nsys stats --report cuda_gpu_mem_time_sum phase3.nsys-rep
```

Flags: `--key value`, `--key=value`, or `--flag` (means true). Flags override
the config file. Unknown keys print a warning (catches typos like `lamda`).

## Layout

```
forge/      physics, CUDA only, never Vulkan
  math/       Vec3, Quat, Mat3: header-only, host + device (FORGE_HD)
  util/       config parser, CUDA_CHECK, DeviceBuffer, device info
  collision/  soft contact force (spring-damper + friction), planes, grid hash
  robots/     URDF reader -> RobotModel (joint tree, MuJoCo coordinates); PD actuators
  dynamics/   spatial algebra; FK, CRBA, RNEA, forward dynamics, Euler (FORGE_HD);
              batches of envs: CPU loop and GPU kernel (one thread per env)
  solver/     constraints (joint limits, sphere-plane contacts, MuJoCo soft model),
              Newton solver, constrained step (FORGE_HD); batches of envs: CPU loop
              and GPU kernels (one thread per env, or one warp per env: newton_warp.cuh)
  particles/  SoA state, free-flight integrators, steppers with collisions:
              step_particles_cpu (brute force answer key), ParticleStepperCpu
              and ParticleStepperGpu (uniform grid)
flux/       Vulkan renderer: context, memory, camera, window/swapchain,
            pipeline (one instanced draw), shaders
  interop/    CUDA-Vulkan shared memory + timeline semaphore (Linux, opaque fd)
temper/     RL and VLA (Phase 7+)
  nn/         layers (CUTLASS Linear), activations, cuDNN conv
  optim/      Adam, grad clipping, LR schedules
  buffers/    rollout/ (on-policy), replay/ (off-policy)
  common/     GAE, obs normalizer, checkpoints
  algos/      ppo/, sac/, td3/, tqc/
  vla/
apps/       train_app (no flux), sandbox_app (scenes: bin, rain, robot), vla_app (later)
configs/    example configs
tests/
  common/     helpers shared by cpu and gpu tests
  cpu/        mirrors the source tree, e.g. cpu/forge/math/test_quat.cpp
  gpu/        mirrors the source tree, files end in _gpu.cu
              (gpu/flux/ builds into tests_interop: needs CUDA + Vulkan)
  data/       reference numbers (MuJoCo)
benchmarks/ headless speed tests (bench_particles, bench_robots)
assets/     robots (URDF): arm (sandbox default), pendulum, arm3, quadruped, ball
tools/      gen_mujoco_refs.py: MuJoCo reference numbers for tests/data
```

Libraries: `forge_cpu` (plain C++, the CPU answer keys), `forge` (forge_cpu + CUDA
kernels), `flux` (Vulkan). A `.cpp` goes in forge_cpu, a `.cu` goes in forge.
