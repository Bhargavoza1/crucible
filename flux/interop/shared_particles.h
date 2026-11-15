#pragma once
// Zero-copy particle buffer shared by Vulkan and CUDA (doc 4.2, Stage B).
//
// Vulkan allocates ONE buffer with exportable memory (opaque fd, Linux only).
// CUDA imports it and gets a normal device pointer. The buffer has 5 regions,
// one per SoA field the renderer needs: px, py, pz, radius, env_id.
// The physics kernels write positions straight into it; Vulkan draws from it.
// No copy, and the data never leaves the GPU.
//
// Sync uses ONE timeline semaphore (its value only goes up). For frame k:
//
//   CUDA:   wait until value >= 2k      (Vulkan finished reading frame k-1)
//           run physics, write positions
//           signal 2k + 1
//   Vulkan: wait for 2k + 1 at the vertex input stage
//           draw
//           signal 2k + 2
//
// So the sync goes both ways: Vulkan never reads half-written positions, and
// CUDA never overwrites positions Vulkan is still drawing.
// Order per frame is fixed: cuda_wait_for_vulkan -> kernels -> cuda_signal_done
// -> Renderer::draw_frame (which calls vk_wait_value/vk_signal_value/next_frame).
#include <cstdint>

#include <cuda_runtime.h>
#include <vulkan/vulkan.h>

namespace flux {

class Context;

class SharedParticles {
public:
    enum Field : int { kPx = 0, kPy, kPz, kRadius, kEnvId, kNumFields };

    // Throws if the device cannot export memory or timeline semaphores, or if
    // CUDA cannot import them (for example, Vulkan and CUDA on different GPUs).
    SharedParticles(const Context& ctx, int capacity);
    ~SharedParticles();
    SharedParticles(const SharedParticles&) = delete;
    SharedParticles& operator=(const SharedParticles&) = delete;

    int capacity() const { return capacity_; }

    // ---- CUDA side: device pointers into the shared memory ----
    float* px() const { return field_ptr<float>(kPx); }
    float* py() const { return field_ptr<float>(kPy); }
    float* pz() const { return field_ptr<float>(kPz); }
    float* radius() const { return field_ptr<float>(kRadius); }
    int* env_id() const { return field_ptr<int>(kEnvId); }

    void cuda_wait_for_vulkan(cudaStream_t stream);  // waits for 2k (no-op for k = 0)
    void cuda_signal_done(cudaStream_t stream);      // signals 2k + 1

    // ---- Vulkan side ----
    VkBuffer buffer() const { return buffer_; }
    VkDeviceSize offset(Field f) const { return VkDeviceSize(f) * region_bytes(); }
    VkSemaphore timeline() const { return timeline_; }
    uint64_t vk_wait_value() const;    // 2k + 1. Throws if CUDA did not signal this frame.
    uint64_t vk_signal_value() const { return 2 * frame_ + 2; }
    void next_frame();                 // the renderer calls this after its submit
    uint64_t frame() const { return frame_; }

private:
    template <typename T>
    T* field_ptr(Field f) const {
        return reinterpret_cast<T*>(static_cast<char*>(cuda_ptr_) + offset(f));
    }
    VkDeviceSize region_bytes() const { return VkDeviceSize(capacity_) * 4; }
    void create_buffer();
    void create_timeline();

    const Context& ctx_;
    int capacity_ = 0;

    VkBuffer buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    VkDeviceSize alloc_size_ = 0;
    VkSemaphore timeline_ = VK_NULL_HANDLE;

    cudaExternalMemory_t cuda_mem_ = nullptr;
    void* cuda_ptr_ = nullptr;
    cudaExternalSemaphore_t cuda_sem_ = nullptr;

    uint64_t frame_ = 0;
    bool cuda_signaled_ = false;
};

}  // namespace flux
