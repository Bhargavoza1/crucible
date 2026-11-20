# crucible

GPU physics engine for robot learning. CUDA C++ physics (**forge**), Vulkan renderer
(**flux**), and a GPU RL stack (**temper**). See the design document (v0.7).

Status: **Phase 11 done** (OpenArm, both arms stack 3 cubes: red, then green, then blue): the
whole pipeline (IK expert -> VLA SFT -> VLA DAgger -> CNN + GRU student by DAgger distillation ->
RL fine-tune) ends with a 64 x 64 camera student that builds the tower in **~92%** of new starts
(see "Results" below). Phase 10: OpenArm pick-and-place, box / sphere / capsule contacts.
Phases 7-8: PPO on the GPU (cartpole, ant, G1 and Go2 walking). Phase 6 and earlier: physics core.

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
| zlib | `zlib1g-dev` | collect_app, vla_app (camera frames) |
| ONNX Runtime | release .tgz, `-DONNXRUNTIME_DIR=...` (optional) | vla_app, distill_app |

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
# PD hold, then tips over (balance is the RL task's job). Phase 10: drawn with its meshes, body and
# legs collide with the floor as capsules (tools/unitree_to_assets.py; not with the terrain heightfield).
./apps/sandbox_app --scene robot --robot ../assets/robots/g1/g1_12dof.urdf --floating true --kp 100 --kv 3 \
    --force 100 --armature 0.01 --damping 0.001 --init_spread 0 --init_speed 0 --envs 9
# Unitree Go2 (assets/robots/go2), the same: meshes and capsules
./apps/sandbox_app --scene robot --robot ../assets/robots/go2/go2.urdf --floating true --kp 20 --kv 0.5 \
    --force 100 --armature 0.01 --damping 0.1 --init_spread 0 --init_speed 0 --envs 9
./benchmarks/bench_robots --robot g1/g1_12dof --constraints --envs_list 1024,4096,16384
# OpenArm v2 arm + pinch gripper (assets/robots/openarm, Apache-2 from enactic/openarm_description,
# made by tools/openarm_to_urdf.py; Phase 10): capsule collisions (a URDF <cylinder> is a capsule),
# drawn with its meshes (<name>.cmesh next to the URDF; --skeleton true also shows the spheres)
./apps/sandbox_app --scene robot --robot ../assets/robots/openarm/openarm.urdf --floor true --kp 100 --kv 3 \
    --force 40 --armature 0.016 --damping 1 --envs 9
# --task play mode draws meshes too (panda_pick, g1_*, go2_*); --skeleton true adds the spheres
# both arms on the OpenArm body (our floor = the table in front of it)
./apps/sandbox_app --scene robot --robot ../assets/robots/openarm/openarm_bimanual.urdf --floor true --kp 100 \
    --kv 3 --force 40 --armature 0.016 --damping 1 --envs 4
# Franka Panda (assets/robots/panda, Apache-2 from MuJoCo Menagerie; meshes from Menagerie's visual
# geoms, Phase 10) + a free cube (Phase 9a):
# arrows / W S move the hand (IK), Q E turn it, G grips, R resets.
./apps/sandbox_app --scene grasp
# Pick task (Phase 9b): end-effector actions; the scripted expert, then recorded demos for imitation
./apps/sandbox_app --task panda_pick --expert true --envs 4
./apps/collect_app --config ../configs/panda_pick.txt --episodes 10000 --out demos/pick.bin
python ../tools/read_demos.py demos/pick.bin --npz demos/pick.npz
# Camera images (Phase 9c): front + wrist cameras, 64 x 64, every 2nd step, zlib in the file
./apps/collect_app --config ../configs/panda_pick.txt --episodes 10000 --images true --out demos/pick_img.bin
python ../tools/read_demos.py demos/pick_img.bin --png ep0.png      # what the cameras saw in episode 0
./tests/tests_flux                                                   # offscreen camera tests (any Vulkan device)
# Behavior cloning baseline in PyTorch (Phase 9d, Python env crucible-tools): images vs full state
python ../tools/bc_train.py demos/pick_img.bin --out bc_image.pt
python ../tools/bc_train.py demos/pick_img.bin --input state --out bc_state.pt
# The same behavior cloning in our engine (needs cuDNN), then the closed loop: the policy drives
# 500 envs with live camera images; prints its success rate and the expert's on the same starts
./apps/bc_app --demos demos/pick_img.bin --out bc_image.bin --eval_envs 500
./apps/bc_app --load bc_image.bin --eval_envs 500                    # closed loop only; failed runs -> bc_image_fail_<env>.png
# Phase 9e: three colored cubes, pick the named one ("pick the red cube"); demos with images
./apps/collect_app --config ../configs/panda_pick.txt --task panda_pick_color --episodes 5000 --images true --out demos/color_img.bin
python ../tools/read_demos.py demos/color_img.bin --png color_ep0.png
# small VLA (images + state + instruction -> action chunk by flow matching), PyTorch; then ONNX
pip install transformers onnx onnxruntime        # once, in crucible-tools
python ../tools/vla_train.py demos/color_img.bin --out vla.pt --onnx vla.onnx
# the VLA drives the robot in the engine (ONNX Runtime; configure with -DONNXRUNTIME_DIR=...)
./apps/vla_app --onnx vla.onnx --eval_envs 500                   # right cube / wrong cube / nothing
./apps/vla_app --onnx vla.onnx --eval_envs 500 --instruct other  # language test: name another cube
# Phase 10: the OpenArm Cell (enactic/openarm_mujoco, as openarm_mjlab's pick_place): the left arm
# starts at home, picks the cube (anywhere in its reach, any turn), puts it into the tray (anywhere
# in its reach), goes home, and the episode ends. Physics 0.002 s x 10 (configs/openarm_pick_place.txt).
./apps/sandbox_app --config ../configs/openarm_pick_place.txt --expert true --envs 9
./apps/collect_app --config ../configs/openarm_pick_place.txt --episodes 10000 --out demos/openarm.bin
# with the cell's 4 cameras (head, left wrist, ceiling, right wrist), 128 x 128, meshes (about 1.5 MB an episode)
./apps/collect_app --config ../configs/openarm_pick_place.txt --episodes 10000 --images true --image_size 128 --out demos/openarm_img.bin
python3 ../tools/read_demos.py demos/openarm_img.bin --png openarm_ep0.png
# OpenArm VLA (SFT): head + left wrist cameras, the arm's state, the instruction -> action chunks
python ../tools/vla_openarm.py demos/openarm_img.bin --out runs/oa_vla.pt --onnx runs/oa_vla.onnx
./apps/vla_app --task openarm_pick_place --onnx runs/oa_vla.onnx --eval_envs 500   # closed loop, vs the expert
# OpenArm VLA DAgger: the VLA drives (other starts: --eval_seed), the expert labels, train on demos + them
./apps/vla_app --task openarm_pick_place --onnx runs/oa_vla.onnx --eval_envs 2000 --eval_seed 2000 --record demos/oa_dagger1.bin
python ../tools/vla_openarm.py demos/openarm_img.bin demos/oa_dagger1.bin --init runs/oa_vla.pt --epochs 3 --out runs/oa_vla_d1.pt --onnx runs/oa_vla_d1.onnx
# OpenArm student: the VLA (teacher) distilled into a small CNN + GRU (temper), DAgger rounds; a
# decision every 3 steps: the student plans a 12-step chunk, runs 3 of it
./apps/distill_app --teacher runs/oa_vla12_d1.onnx --out runs/student.bin   # 12-step chunks, 3 cameras
./apps/distill_app --load runs/student.bin --rounds 0 --eval_envs 500            # closed loop only
# watch a camera policy in the sandbox (the status line counts the episodes that ended done)
./apps/sandbox_app --config ../configs/openarm_pick_place.txt --envs 9 --student runs/student.bin
./apps/sandbox_app --config ../configs/openarm_pick_place.txt --envs 9 --vla runs/oa_vla12_d1.onnx --exec 3
# Phase 11: both OpenArm arms build a tower on the gray spot: red, then green, then blue on top
# (cubes anywhere in reach, each arm stacks the cubes on its side). Physics 0.002 s x 10.
./apps/sandbox_app --config ../configs/openarm_stack.txt --expert true --envs 9
# 10 000 successful demos with 4 cameras (failed episodes are thrown away; ~48 GB at 128 x 128);
# the expert moves slowly near the cubes; with --push_prob / --slip_prob mistakes it shows recoveries
./apps/collect_app --config ../configs/openarm_stack.txt --episodes 10000 --only_success true --images true --image_size 128 --out demos/openarm_stack.bin
# stack VLA (head + both wrist cameras, both arms' joint positions -> 10 actions per step);
# --no_velocity: no joint velocities in the state (with them it copied its last motion);
# --keep_epochs: every epoch's checkpoint (_e<N>.pt); export one, pick the best by the closed loop
python ../tools/vla_openarm.py demos/openarm_stack.bin --no_velocity --keep_epochs --epochs 50 --val_every 50 --out runs/st_vla.pt --onnx runs/st_vla.onnx
python ../tools/vla_openarm.py --load runs/st_vla_e48.pt --onnx runs/st_vla_e48.onnx
./apps/vla_app --task openarm_stack --onnx runs/st_vla_e48.onnx --eval_envs 500
python ../tools/check_vla_start.py runs/st_vla_e48.onnx demos/openarm_stack.bin   # its first actions vs the expert's
# VLA DAgger rounds: the VLA drives new starts, the expert labels; train on demos + all rounds
./apps/vla_app --task openarm_stack --onnx runs/st_vla_e48.onnx --eval_envs 2000 --eval_seed 2000 --save_failures 0 --eval_expert false --record demos/st_dagger1.bin
python ../tools/vla_openarm.py demos/openarm_stack.bin demos/st_dagger1.bin --init runs/st_vla.pt --no_velocity --keep_epochs --epochs 2 --val_every 50 --out runs/st_vla_d1.pt --onnx runs/st_vla_d1.onnx
# stack student: the VLA distilled into the CNN + GRU (temper). The aux head guesses the cubes'
# positions from the images (so the CNN learns to see them); the teacher labels with zero flow
# noise (clean targets: with fresh noise the student learned timid averages). Every round is kept:
# pick the best by the eval, then go on from it with a fresh teacher round.
./apps/distill_app --task openarm_stack --teacher runs/st5_d3_e7.onnx --teacher_noise zero --exec 4 --teacher_envs 2000 --teacher_frames 300000 --dagger_frames 200000 --rounds 15 --train_steps 20000 --dagger_train_steps 4000 --keep_rounds true --out runs/st_student.bin --eval_envs 0
for f in runs/st_student_r*.bin; do echo "== $f"; ./apps/distill_app --task openarm_stack --teacher runs/st5_d3_e7.onnx --load $f --rounds 0 --eval_envs 500 2>&1 | grep "eval: student"; done
./apps/distill_app --task openarm_stack --teacher runs/st5_d3_e7.onnx --teacher_noise zero --load runs/st_student_r14.bin --teacher_round true --teacher_envs 2000 --teacher_frames 300000 --dagger_frames 200000 --rounds 30 --train_steps 4000 --dagger_train_steps 4000 --lr 1e-4 --keep_rounds true --out runs/st_student_z2.bin --eval_envs 0
# where it fails: record the eval (obs, the teacher's and the student's actions), then the npz
./apps/distill_app --task openarm_stack --teacher runs/st5_d3_e7.onnx --load runs/st_student_z2_r29.bin --rounds 0 --eval_envs 500 --record runs/st_track.bin
python ../tools/read_demos.py runs/st_track.bin --npz st_track.npz
# RL fine-tune of the student (PPO, the task's reward with the dense progress term; small steps:
# bigger ones broke the student). --start_mix: the training starts' curriculum (red placed, red +
# green placed, a cube at the edge; the eval keeps normal starts). Slow: ~2000 iterations to 92%
# (the 92% run went st_rl2 -> st_rl3 -> st_rl_cur, 512 envs). One run, 2048 envs (~16 s an iteration
# on an RTX 4070 Ti); --timing true adds a line per iteration with where the time went:
./apps/rl_app --load runs/st_student_z2_r29.bin --gamma 0.997 --iters 3000 --envs 2048 --lr 5e-6 --epochs 2 --target_kl 0.01 --init_std 0.05 --critic_warmup 20 --eval_every 10 --start_mix 0.2,0.25,0.05 --out runs/st_rl_2048.bin
./apps/rl_app --load runs/st_rl_best.bin --iters 0 --eval_envs 1000 --eval_seed 7000   # the honest number: new starts
./apps/sandbox_app --config ../configs/openarm_stack.txt --student runs/st_rl_best.bin --envs 1
# a video of it (needs ffmpeg): each frame moves the sim 1/50 s, so the video is real time
./apps/sandbox_app --config ../configs/openarm_stack.txt --student runs/st_rl_best.bin --envs 1 --video stack.mp4 --video_seconds 30 --width 1920 --height 1080 --corner_axes false --orbit_speed 0.1
# DAgger for the VLA: record the policy's states with the expert's actions, train on demos + them
./apps/vla_app --onnx vla.onnx --eval_envs 500 --record dagger1.bin
python ../tools/vla_train.py demos/color_img.bin dagger1.bin --init vla.pt --epochs 5 --out vla_d1.pt --onnx vla_d1.onnx
# DAgger (Phase 9d): the policy drives, the expert labels where it goes, retrain (4 rounds)
./apps/bc_app --demos demos/pick_img.bin --load bc_image.bin --epochs 0 --dagger_rounds 4 --out bc_dagger.bin --eval_envs 2000
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
./apps/train_app --config ../configs/g1_walk.txt --save runs/ --viz true         # same, watched live (V = drawing on / off)
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
# which kernels take the GPU time in RL (2 short iterations)
nsys profile -o rl_prof --force-overwrite true ./apps/rl_app --load runs/st_student_z2_r29.bin --iters 2 --envs 2048 --critic_warmup 1 --eval_envs 0 --out runs/timing_test.bin
nsys stats --report cuda_gpu_kern_sum rl_prof.nsys-rep | head -15
```

Inside the physics kernel (a debug build: `FORGE_PROFILE_SOLVER`, off by default): `rl_app
--timing true` then also prints the share of each phase of the warp solver (kinematics, mass
matrix, bias, Cholesky of M, contacts, Newton's H, Cholesky of H, line search, ...):

```bash
cmake build -DFORGE_PROFILE_SOLVER=ON && cmake --build build -j
./apps/rl_app --load runs/st_student_z2_r29.bin --iters 3 --envs 2048 --critic_warmup 1 --eval_envs 0 --timing true --out runs/timing_test.bin
cmake build -DFORGE_PROFILE_SOLVER=OFF && cmake --build build -j   # back, before training
```

Videos (needs ffmpeg): `--video out.mp4` in the task scene records every drawn frame at the
window's size (`--width`, `--height`); while recording, each frame moves the sim by exactly
`--video_speed / --video_fps` seconds (default 1/50: real time, smooth however slow the drawing),
`--video_speed 0.5` is slow motion, `--video_seconds N` stops after N seconds of video.

Sandbox mouse: left drag = orbit, right or middle drag = pan, wheel = zoom. `--orbit_speed 0.15`
turns the camera by itself. The bottom-left corner shows the world axes (+X red, +Y green, +Z blue),
turning with the camera (`--corner_axes false` hides them); `--axes true` also draws them at
each env's origin.

Flags: `--key value`, `--key=value`, or `--flag` (means true). Flags override
the config file. Unknown keys print a warning (catches typos like `lamda`).

## Results (Phase 11: two-arm cube stacking)

Done = the tower built on the spot (red, green on red, blue on green, within 1.8 cm) and both
hands back home. 500 or 1000 envs, the same evaluation starts for every row.

| Policy | Input | Done |
|---|---|---|
| Scripted IK expert | true state (cube poses) | 94% |
| VLA (flow matching, CLIP text), SFT 70 epochs | 3 cameras 128 x 128, joint positions | 55% |
| VLA after DAgger | same | 72% |
| CNN + GRU student, DAgger distillation (2 M parameters) | 3 cameras 64 x 64, joints | 51% |
| CNN + GRU student after RL (PPO + dense reward + start curriculum) | same | **92%** (new starts; 303 steps vs the expert's 440) |

What it took (each found by recording the policy and comparing it with its teacher):

- **Copycat**: with joint velocities in its state the VLA repeated its last motion -> `--no_velocity`.
- **Covariate shift**: slow expert near the cubes, push / slip recovery demos, then DAgger.
- **A blind student**: the CNN never learned to find the cubes (its moves were as good as zeros)
  -> an aux head that predicts the cube positions from the images (training only).
- **Noisy labels**: the VLA samples a new plan per call; the student's MSE learned the average
  (small, timid moves) -> the teacher labels with zero flow noise (it is as good: 71% vs 70%).
- **DAgger data**: keep more than the last round (`--dagger_frames`), keep every round's
  student and pick by evaluation (rounds differ by +-5%).
- **RL**: whole-network PPO broke the student at normal step sizes; small steps (lr 5e-6, KL
  0.01) and a dense, potential-based progress reward made it climb, slowly (~2000 iterations).
- **Selection bias**: the best of 200 evaluations on the same 500 starts said 95.4%; 3000 new
  starts say 92.3%.

## Speed (after Phase 11)

RL fine-tune of the stack student, 2048 envs x 3 cameras (64 x 64, drawn at 128 and halved),
RTX 4070 Ti. nsys first showed one kernel, the warp physics solver, at 84% of the GPU time.

| One iteration (32 decisions x 4 steps x 2048 envs) | Before | After |
|---|---|---|
| physics (step + download) | 12.0 s | ~6.5 s |
| halve the images (CPU) | 4.0 s | 0.5 s |
| rollout | 18 s | 9-10 s |
| whole iteration (with the PPO update) | 24-25 s | 16 s |

What changed; every step keeps the results bit-identical (same qpos on the CPU over 300 steps of
openarm_stack and g1_walk; the same RL run, the same evals: 65.4% after 40 iterations both times):

- capsule-box and sphere-box pairs skip the full test when their bounding spheres are apart
  (24 capsules x 3 cubes were all tested every step)
- profile (skyline) Cholesky of M: the sums start at each row's first nonzero
- the warp solver: the contact tests on all 32 lanes (slots in row order + a warp prefix sum),
  the mass matrix one column per lane, the Cholesky of M one block (arm, cube) per lane
- rl_app: the image halving on all CPU threads

Left (`FORGE_PROFILE_SOLVER` shares): contacts ~24%, Newton's H ~24%, Cholesky of H ~17%; the PPO
update (~6.5 s) is now as big as the physics.

## Adding a vision task to the pipeline

distill_app and rl_app take everything task-specific from `apps/common/vision_task.h`: sizes,
the done rule, the aux targets, the start curriculum, the camera rig. A new task:

1. its task file in `forge/env/tasks/` (physics, obs, reward, expert), registered with forge
2. a camera rig in `apps/collect_app/` if it needs a new scene
3. one entry in `make_vision_tasks()` (`apps/common/vision_task.cpp`)

(collect_app, vla_app and sandbox_app still have their own task code.)

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
            pipeline (one instanced draw), shaders; BatchCamera (policy images),
            video_writer (frames -> ffmpeg -> .mp4)
  interop/    CUDA-Vulkan shared memory + timeline semaphore (Linux, opaque fd)
temper/     RL and VLA (Phase 7+)
  nn/         Tensor, GEMM (cuBLAS; CUTLASS backend), Linear, ELU, MLP; CPU reference (double) for tests
  optim/      Adam, gradient-norm clipping (all on the GPU)
  algos/ppo/  PPO (Gaussian actor, critic, GAE, clipped loss) and its training loop
  common/     train_loop (any algorithm), checkpoint file, log.csv writer, observation normalizer
  buffers/    replay/ (off-policy ring buffer); PPO keeps its rollout buffer in ppo.cu for now
  algos/      algorithm.h (interface, doc 5.1), ppo/, sac/ (SAC and TQC); td3/ (later)
  vla/
apps/       train_app (no flux), sandbox_app (scenes: bin, rain, robot; --task: play mode, --video),
            collect_app (expert demos + camera images), bc_app, vla_app (ONNX VLA closed loop, DAgger
            records), distill_app (VLA -> CNN + GRU student), rl_app (PPO fine-tune of the student);
            common/vision_task: the pipeline's tasks (one entry per task)
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
            vla_openarm.py: OpenArm VLA training (PyTorch -> ONNX); check_vla_start.py: its actions vs
            the expert's; read_demos.py: demo files -> PNG / npz
```

Libraries: `forge_cpu` (plain C++, the CPU answer keys), `forge` (forge_cpu + CUDA
kernels), `flux` (Vulkan). A `.cpp` goes in forge_cpu, a `.cu` goes in forge.
