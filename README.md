# crucible

GPU physics engine for robot learning. CUDA C++ physics (**forge**), Vulkan renderer
(**flux**), and a GPU RL stack (**temper**). See the design document (v0.7).

Status: **Phase 7d** (ant task: walks forward with PPO). 7c: PPO on the GPU, matched to a PyTorch
baseline on cartpole (3 seeds). 7b: nn. 7a: Env API. Phase 6 done.

## Requirements (Linux)

| Need | Package (Ubuntu 24.04) | Used by |
|---|---|---|
| CMake 3.24+, g++ | `cmake build-essential` | all |
| CUDA 12.x | NVIDIA CUDA Toolkit | forge kernels, tests_gpu |
| Vulkan 1.2+, glslc | `libvulkan-dev glslc vulkan-validationlayers` | flux (only with `BUILD_FLUX=ON`) |
| GLFW 3.3+ | `libglfw3-dev` | flux |
| GoogleTest | `libgtest-dev` (optional, else cloned from GitHub) | tests |
| tinyxml2 | `libtinyxml2-dev` (optional, else cloned from GitHub) | URDF reader |
| CUTLASS 3.5.1 | nothing: CMake clones it (header-only) | temper nn (GEMM) |
| cuBLAS | part of the CUDA Toolkit | tests_gpu only (GEMM oracle) |
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
# Unitree G1, 12 leg joints (assets/robots/g1, BSD-3 from unitree_rl_gym): stands about 1 s under
# PD hold, then tips over (balance is the RL task's job); only the feet collide.
./apps/sandbox_app --scene robot --robot ../assets/robots/g1/g1_12dof.urdf --floating true --kp 100 --kv 3 \
    --force 100 --armature 0.01 --damping 0.001 --init_spread 0 --init_speed 0 --envs 9
./benchmarks/bench_robots --robot g1/g1_12dof --constraints --envs_list 1024,4096,16384
# Franka Panda (assets/robots/panda, Apache-2 from MuJoCo Menagerie) + a free cube (Phase 9a):
# arrows / W S move the hand (IK), Q E turn it, G grips, R resets.
./apps/sandbox_app --scene grasp
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
./apps/train_app --task cartpole --num_envs 4096 --bench_env 1000      # env steps/s, random actions
```

Train (doc 7.1). The run folder gets config.txt, log.csv, ckpt_*.bin, best.bin, final.bin:

```bash
./apps/train_app --config ../configs/cartpole.txt --save runs/
./apps/sandbox_app --task cartpole --load runs/cartpole_<date>_seed1/best.bin   # play mode: watch it
./apps/train_app --config ../configs/sac_cartpole.txt --save runs/              # SAC (algo = sac)
./apps/train_app --config ../configs/sac_ant.txt --save runs/                   # SAC on the ant
./apps/train_app --config ../configs/tqc_ant.txt --save runs/                   # TQC (algo = tqc)
./apps/train_app --config ../configs/g1_stand.txt --save runs/                  # G1 stands under pushes (Phase 8b)
./apps/sandbox_app --task g1_stand --load runs/g1_stand_<date>_seed1/best.bin --envs 9
./apps/train_app --config ../configs/g1_walk.txt --save runs/                   # G1 walks on velocity commands (Phase 8c)
./apps/sandbox_app --task g1_walk --load runs/g1_walk_<date>_seed1/best.bin --envs 1   # drive: W/S A/D Q/E, space, R
./apps/train_app --config ../configs/g1_walk_rough.txt --save runs/ --resume runs/g1_walk_<date>_seed1/best.bin  # terrain (8d)
./apps/train_app --config ../configs/go2_walk.txt --save runs/                  # Go2 walks on velocity commands (Phase 8e)
./apps/sandbox_app --task go2_walk --load runs/go2_walk_<date>_seed1/best.bin --envs 1  # drive as the G1
```

Baseline check (doc 5.7): the same PPO in PyTorch on the same cartpole (MuJoCo), then both curves
in one plot. Python env `crucible-tools` (torch, mujoco, matplotlib):

```bash
python3 tools/ppo_baseline.py --seed 1 --out runs_baseline/cartpole_seed1
python3 tools/ppo_baseline.py --task ant --iterations 120 --out runs_baseline/ant_seed1
python3 tools/sac_baseline.py --task cartpole --out runs_baseline/sac_cartpole_seed1
python3 tools/plot_log.py build/runs/cartpole_<date>_seed1=crucible runs_baseline/cartpole_seed1=pytorch -o cartpole.png
```

GEMM backend (every Linear layer): cuBLAS by default, `--gemm cutlass` for the
CUTLASS kernel. Compare their speed on our layer shapes:

```bash
./benchmarks/bench_gemm            # batches 1024 and 4096
```

Profile (Nsight Systems):

```bash
nsys profile -t cuda,nvtx,vulkan -o phase3 ./apps/sandbox_app --frames 600 --reset_every 1000
nsys stats --report cuda_gpu_mem_time_sum phase3.nsys-rep
```

Sandbox mouse: left drag = orbit, right or middle drag = pan, wheel = zoom. `--orbit_speed 0.15`
turns the camera by itself. The bottom-left corner shows the world axes (+X red, +Y green, +Z blue),
turning with the camera (`--corner_axes false` hides them); `--axes true` also draws them at
each env's origin.

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
  env/        Env API (doc 3.6), CPU and GPU backends; tasks/ (cartpole, ant)
  solver/     Solver interface (newton); constraints (joint limits, sphere-plane contacts, MuJoCo soft model),
              Newton solver, constrained step (FORGE_HD); batches of envs: CPU loop
              and GPU kernels (one thread per env, or one warp per env: newton_warp.cuh)
  particles/  SoA state, free-flight integrators, steppers with collisions:
              step_particles_cpu (brute force answer key), ParticleStepperCpu
              and ParticleStepperGpu (uniform grid)
flux/       Vulkan renderer: context, memory, camera, window/swapchain,
            pipeline (one instanced draw), shaders
  interop/    CUDA-Vulkan shared memory + timeline semaphore (Linux, opaque fd)
temper/     RL and VLA (Phase 7+)
  nn/         Tensor, GEMM (CUTLASS), Linear, ELU, MLP; CPU reference (double) for tests
  optim/      Adam, gradient-norm clipping (all on the GPU)
  algos/ppo/  PPO (Gaussian actor, critic, GAE, clipped loss) and its training loop
  common/     train_loop (any algorithm), checkpoint file, log.csv writer, observation normalizer
  buffers/    replay/ (off-policy ring buffer); PPO keeps its rollout buffer in ppo.cu for now
  algos/      algorithm.h (interface, doc 5.1), ppo/, sac/ (SAC and TQC); td3/ (later)
  vla/
apps/       train_app (no flux), sandbox_app (scenes: bin, rain, robot; --task: play mode), vla_app (later)
configs/    example configs
tests/
  common/     helpers shared by cpu and gpu tests
  cpu/        mirrors the source tree, e.g. cpu/forge/math/test_quat.cpp
  gpu/        mirrors the source tree, files end in _gpu.cu
              (gpu/flux/ builds into tests_interop: needs CUDA + Vulkan)
  data/       reference numbers (MuJoCo)
benchmarks/ headless speed tests (bench_particles, bench_robots, bench_gemm)
assets/     robots (URDF): arm (sandbox default), cartpole, ant, pendulum, arm3, quadruped, ball
tools/      gen_mujoco_refs.py: MuJoCo reference numbers for tests/data
            ppo_baseline.py, sac_baseline.py: PyTorch PPO / SAC / TQC on the same tasks; plot_log.py: curves
            sac_check_step.py: redo one temper SAC or TQC gradient step in PyTorch and compare
```

Libraries: `forge_cpu` (plain C++, the CPU answer keys), `forge` (forge_cpu + CUDA
kernels), `flux` (Vulkan). A `.cpp` goes in forge_cpu, a `.cu` goes in forge.
