// CUDA-Vulkan sharing tests (doc 4.2, Stage B). No window: a headless Context.
// Needs a CUDA device and a Vulkan driver that can export memory and timeline
// semaphores. Each test skips (with the reason) if the machine cannot do it.
//
// A broken sync usually HANGS instead of failing. Every wait here has a
// timeout; on timeout the test fails and the program exits at once.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <thread>

#include <gtest/gtest.h>

#include "flux/context.h"
#include "flux/interop/shared_particles.h"
#include "flux/memory.h"
#include "flux/vk_check.h"
#include "forge/util/cuda_check.h"
#include "forge/util/device_info.h"

using flux::SharedParticles;

namespace {

constexpr int kN = 1000;
constexpr double kTimeoutS = 5.0;

[[noreturn]] void die(const char* what) {
    std::fprintf(stderr, "TIMEOUT: %s. Sync is broken (a wait was never signaled).\n", what);
    std::fflush(stderr);
    std::_Exit(1);  // do not run destructors: they would wait on the stuck GPU work
}

void wait_stream(cudaStream_t s) {
    const auto t0 = std::chrono::steady_clock::now();
    while (cudaStreamQuery(s) == cudaErrorNotReady) {
        if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > kTimeoutS)
            die("CUDA stream did not finish");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    CUDA_CHECK(cudaGetLastError());
}

__global__ void fill_kernel(float* px, int* env_id, int n, float scale) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        px[i] = scale * float(i);
        env_id[i] = i;
    }
}

__global__ void add_one_kernel(float* px, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) px[i] += 1.0f;
}

// Headless Vulkan + CUDA setup shared by the tests.
class Interop : public ::testing::Test {
protected:
    void SetUp() override {
        auto uuid = forge::cuda_device_uuid(0);
        if (!uuid) GTEST_SKIP() << "no CUDA device";
        flux::ContextOptions opt;
        opt.validation = true;
        opt.interop = true;
        opt.required_uuid = *uuid;
        try {
            ctx_ = std::make_unique<flux::Context>(opt);
            shared_ = std::make_unique<SharedParticles>(*ctx_, kN);
        } catch (const std::exception& e) {
            GTEST_SKIP() << "interop not available: " << e.what();
        }
        CUDA_CHECK(cudaStreamCreate(&stream_));

        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex = ctx_->queue_family;
        VK_CHECK(vkCreateCommandPool(ctx_->device, &pci, nullptr, &pool_));
        readback_ = flux::create_host_buffer(*ctx_, VkDeviceSize(kN) * 4 * 2, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    }

    void TearDown() override {
        if (!ctx_) return;
        if (stream_) {
            wait_stream(stream_);
            cudaStreamDestroy(stream_);
        }
        vkDeviceWaitIdle(ctx_->device);
        flux::destroy(*ctx_, readback_);
        if (pool_) vkDestroyCommandPool(ctx_->device, pool_, nullptr);
        shared_.reset();
        ctx_.reset();
    }

    // Records commands with 'rec', submits them waiting for 'wait_value' on the
    // timeline and signaling 'signal_value', then waits (with timeout) on a fence.
    void vk_submit(const std::function<void(VkCommandBuffer)>& rec, uint64_t wait_value, uint64_t signal_value) {
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = pool_;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VkCommandBuffer cmd;
        VK_CHECK(vkAllocateCommandBuffers(ctx_->device, &ai, &cmd));
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
        rec(cmd);
        VK_CHECK(vkEndCommandBuffer(cmd));

        VkSemaphore tl = shared_->timeline();
        VkPipelineStageFlags stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        VkTimelineSemaphoreSubmitInfo ti{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
        ti.waitSemaphoreValueCount = 1;
        ti.pWaitSemaphoreValues = &wait_value;
        ti.signalSemaphoreValueCount = 1;
        ti.pSignalSemaphoreValues = &signal_value;
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.pNext = &ti;
        si.waitSemaphoreCount = 1;
        si.pWaitSemaphores = &tl;
        si.pWaitDstStageMask = &stage;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        si.signalSemaphoreCount = 1;
        si.pSignalSemaphores = &tl;

        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VkFence fence;
        VK_CHECK(vkCreateFence(ctx_->device, &fi, nullptr, &fence));
        VK_CHECK(vkQueueSubmit(ctx_->queue, 1, &si, fence));
        const VkResult r = vkWaitForFences(ctx_->device, 1, &fence, VK_TRUE, uint64_t(kTimeoutS * 1e9));
        if (r == VK_TIMEOUT) die("Vulkan submit did not finish");
        VK_CHECK(r);
        vkDestroyFence(ctx_->device, fence, nullptr);
        vkFreeCommandBuffers(ctx_->device, pool_, 1, &cmd);
    }

    // Copy px and env_id from the shared buffer into the host-visible readback buffer.
    void record_readback(VkCommandBuffer cmd) {
        const VkDeviceSize bytes = VkDeviceSize(kN) * 4;
        VkBufferCopy regions[2] = {{shared_->offset(SharedParticles::kPx), 0, bytes},
                                   {shared_->offset(SharedParticles::kEnvId), bytes, bytes}};
        vkCmdCopyBuffer(cmd, shared_->buffer(), readback_.buffer, 2, regions);
        // Make the copy visible to the CPU after the fence.
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0,
                             nullptr, 0, nullptr);
    }

    const float* readback_px() const { return static_cast<const float*>(readback_.mapped); }
    const int* readback_env() const { return static_cast<const int*>(readback_.mapped) + kN; }

    std::unique_ptr<flux::Context> ctx_;
    std::unique_ptr<SharedParticles> shared_;
    cudaStream_t stream_ = nullptr;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    flux::HostBuffer readback_;
};

constexpr int kBlock = 256;
constexpr int kGrid = (kN + kBlock - 1) / kBlock;

}  // namespace

TEST_F(Interop, VulkanPicksTheCudaGpu) {
    auto uuid = forge::cuda_device_uuid(0);
    ASSERT_TRUE(uuid);
    EXPECT_TRUE(std::equal(uuid->begin(), uuid->end(), ctx_->device_uuid.begin()))
        << "Vulkan uuid " << flux::uuid_hex(ctx_->device_uuid);
}

// The drawing direction: CUDA writes, signals; Vulkan waits, reads.
TEST_F(Interop, CudaWritesVulkanReads) {
    fill_kernel<<<kGrid, kBlock, 0, stream_>>>(shared_->px(), shared_->env_id(), kN, 0.5f);
    KERNEL_CHECK();
    shared_->cuda_signal_done(stream_);  // value 1
    vk_submit([&](VkCommandBuffer c) { record_readback(c); }, shared_->vk_wait_value(),
              shared_->vk_signal_value());  // waits 1, signals 2
    shared_->next_frame();

    for (int i = 0; i < kN; ++i) {
        ASSERT_EQ(readback_px()[i], 0.5f * float(i)) << "i=" << i;
        ASSERT_EQ(readback_env()[i], i) << "i=" << i;
    }
    // The reverse wait must also complete: CUDA waits for value 2.
    shared_->cuda_wait_for_vulkan(stream_);
    wait_stream(stream_);
}

// The other direction: Vulkan writes, signals; CUDA waits, reads.
TEST_F(Interop, VulkanWritesCudaReads) {
    shared_->cuda_signal_done(stream_);  // frame 0: CUDA has nothing to write
    const float value = 2.5f;
    uint32_t bits;
    std::memcpy(&bits, &value, 4);
    vk_submit(
        [&](VkCommandBuffer c) {
            vkCmdFillBuffer(c, shared_->buffer(), shared_->offset(SharedParticles::kPy), VkDeviceSize(kN) * 4, bits);
        },
        shared_->vk_wait_value(), shared_->vk_signal_value());
    shared_->next_frame();

    shared_->cuda_wait_for_vulkan(stream_);  // waits for value 2
    std::vector<float> py(kN);
    CUDA_CHECK(cudaMemcpyAsync(py.data(), shared_->py(), kN * 4, cudaMemcpyDeviceToHost, stream_));
    wait_stream(stream_);
    for (int i = 0; i < kN; ++i) ASSERT_EQ(py[i], value) << "i=" << i;
}

// The full per-frame loop, 100 times: CUDA waits, adds 1, signals; Vulkan waits,
// reads, signals. A missed signal hangs (caught by the timeout).
TEST_F(Interop, HundredFramesLoop) {
    CUDA_CHECK(cudaMemsetAsync(shared_->px(), 0, kN * 4, stream_));
    const int frames = 100;
    for (int f = 0; f < frames; ++f) {
        shared_->cuda_wait_for_vulkan(stream_);
        add_one_kernel<<<kGrid, kBlock, 0, stream_>>>(shared_->px(), kN);
        KERNEL_CHECK();
        shared_->cuda_signal_done(stream_);
        vk_submit([&](VkCommandBuffer c) { record_readback(c); }, shared_->vk_wait_value(),
                  shared_->vk_signal_value());
        shared_->next_frame();
        // Vulkan must see this frame's value: never last frame's, never half done.
        ASSERT_EQ(readback_px()[0], float(f + 1)) << "frame " << f;
        ASSERT_EQ(readback_px()[kN - 1], float(f + 1)) << "frame " << f;
    }
    EXPECT_EQ(shared_->frame(), uint64_t(frames));
}

// Calling the Vulkan side before CUDA signaled would hang the GPU forever.
// SharedParticles catches this mistake on the CPU instead.
TEST_F(Interop, DrawBeforeCudaSignalThrows) {
    EXPECT_THROW(shared_->vk_wait_value(), std::logic_error);
}
