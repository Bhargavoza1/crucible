#include "flux/interop/shared_particles.h"

#include <unistd.h>  // close()

#include <stdexcept>
#include <string>

#include "flux/context.h"
#include "flux/memory.h"
#include "flux/vk_check.h"
#include "forge/util/cuda_check.h"

namespace flux {

namespace {
constexpr VkExternalMemoryHandleTypeFlagBits kMemHandle = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
constexpr VkExternalSemaphoreHandleTypeFlagBits kSemHandle = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
}  // namespace

SharedParticles::SharedParticles(const Context& ctx, int capacity) : ctx_(ctx) {
    if (!ctx.interop_on) throw std::runtime_error("SharedParticles: Context was made without interop");
    if (capacity <= 0) throw std::invalid_argument("SharedParticles: capacity must be > 0");
    // Round up to 64 values, so every region starts on a 256-byte boundary
    // (good for coalesced CUDA access).
    capacity_ = (capacity + 63) / 64 * 64;
    create_buffer();
    create_timeline();
}

SharedParticles::~SharedParticles() {
    // Make sure neither side is still using the memory.
    cudaDeviceSynchronize();
    vkDeviceWaitIdle(ctx_.device);
    // CUDA side first: the mapped pointer is freed with cudaFree (CUDA docs).
    if (cuda_ptr_) cudaFree(cuda_ptr_);
    if (cuda_mem_) cudaDestroyExternalMemory(cuda_mem_);
    if (cuda_sem_) cudaDestroyExternalSemaphore(cuda_sem_);
    if (timeline_) vkDestroySemaphore(ctx_.device, timeline_, nullptr);
    if (buffer_) vkDestroyBuffer(ctx_.device, buffer_, nullptr);
    if (memory_) vkFreeMemory(ctx_.device, memory_, nullptr);
}

void SharedParticles::create_buffer() {
    const VkBufferUsageFlags usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                     VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    // Ask the driver first: can this kind of buffer be exported as an opaque fd?
    VkPhysicalDeviceExternalBufferInfo q{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO};
    q.usage = usage;
    q.handleType = kMemHandle;
    VkExternalBufferProperties qp{VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES};
    vkGetPhysicalDeviceExternalBufferProperties(ctx_.physical, &q, &qp);
    if (!(qp.externalMemoryProperties.externalMemoryFeatures & VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT))
        throw std::runtime_error("SharedParticles: this Vulkan device cannot export buffer memory as an fd");

    VkExternalMemoryBufferCreateInfo ext{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    ext.handleTypes = kMemHandle;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.pNext = &ext;
    bi.size = region_bytes() * kNumFields;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(ctx_.device, &bi, nullptr, &buffer_));

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(ctx_.device, buffer_, &req);
    alloc_size_ = req.size;

    // Dedicated allocation (one memory block for this one buffer): NVIDIA's
    // recommended way to share with CUDA. CUDA must then be told "dedicated" too.
    VkMemoryDedicatedAllocateInfo dedicated{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
    dedicated.buffer = buffer_;
    VkExportMemoryAllocateInfo export_info{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
    export_info.pNext = &dedicated;
    export_info.handleTypes = kMemHandle;
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.pNext = &export_info;
    ai.allocationSize = alloc_size_;
    ai.memoryTypeIndex = find_memory_type(ctx_, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(ctx_.device, &ai, nullptr, &memory_));
    VK_CHECK(vkBindBufferMemory(ctx_.device, buffer_, memory_, 0));

    // Export as a file descriptor and give it to CUDA.
    VkMemoryGetFdInfoKHR gi{VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
    gi.memory = memory_;
    gi.handleType = kMemHandle;
    int fd = -1;
    VK_CHECK(ctx_.vkGetMemoryFdKHR(ctx_.device, &gi, &fd));

    cudaExternalMemoryHandleDesc md{};
    md.type = cudaExternalMemoryHandleTypeOpaqueFd;
    md.handle.fd = fd;
    md.size = alloc_size_;
    md.flags = cudaExternalMemoryDedicated;
    const cudaError_t err = cudaImportExternalMemory(&cuda_mem_, &md);
    if (err != cudaSuccess) {
        close(fd);  // on success CUDA owns the fd; on failure we must close it
        cudaGetLastError();
        throw std::runtime_error(std::string("SharedParticles: CUDA cannot import Vulkan memory (") +
                                 cudaGetErrorString(err) + "). Are CUDA and Vulkan on the same GPU?");
    }

    cudaExternalMemoryBufferDesc bd{};
    bd.offset = 0;
    bd.size = region_bytes() * kNumFields;
    CUDA_CHECK(cudaExternalMemoryGetMappedBuffer(&cuda_ptr_, cuda_mem_, &bd));
}

void SharedParticles::create_timeline() {
    VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    type.initialValue = 0;

    // Ask the driver: can a timeline semaphore be exported as an opaque fd?
    VkPhysicalDeviceExternalSemaphoreInfo q{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO};
    q.pNext = &type;
    q.handleType = kSemHandle;
    VkExternalSemaphoreProperties qp{VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES};
    vkGetPhysicalDeviceExternalSemaphoreProperties(ctx_.physical, &q, &qp);
    if (!(qp.externalSemaphoreFeatures & VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT))
        throw std::runtime_error("SharedParticles: this Vulkan device cannot export timeline semaphores as an fd");

    VkExportSemaphoreCreateInfo export_info{VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO};
    export_info.handleTypes = kSemHandle;
    type.pNext = &export_info;
    VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    si.pNext = &type;
    VK_CHECK(vkCreateSemaphore(ctx_.device, &si, nullptr, &timeline_));

    VkSemaphoreGetFdInfoKHR gi{VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR};
    gi.semaphore = timeline_;
    gi.handleType = kSemHandle;
    int fd = -1;
    VK_CHECK(ctx_.vkGetSemaphoreFdKHR(ctx_.device, &gi, &fd));

    cudaExternalSemaphoreHandleDesc sd{};
    sd.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
    sd.handle.fd = fd;
    const cudaError_t err = cudaImportExternalSemaphore(&cuda_sem_, &sd);
    if (err != cudaSuccess) {
        close(fd);
        cudaGetLastError();
        throw std::runtime_error(std::string("SharedParticles: CUDA cannot import the timeline semaphore (") +
                                 cudaGetErrorString(err) + ")");
    }
}

void SharedParticles::cuda_wait_for_vulkan(cudaStream_t stream) {
    if (frame_ == 0) return;  // nothing drawn yet
    cudaExternalSemaphoreWaitParams wp{};
    wp.params.fence.value = 2 * frame_;
    CUDA_CHECK(cudaWaitExternalSemaphoresAsync(&cuda_sem_, &wp, 1, stream));
}

void SharedParticles::cuda_signal_done(cudaStream_t stream) {
    cudaExternalSemaphoreSignalParams sp{};
    sp.params.fence.value = 2 * frame_ + 1;
    CUDA_CHECK(cudaSignalExternalSemaphoresAsync(&cuda_sem_, &sp, 1, stream));
    cuda_signaled_ = true;
}

uint64_t SharedParticles::vk_wait_value() const {
    // Waiting for a value nobody will signal hangs the GPU forever. Catch it here.
    if (!cuda_signaled_)
        throw std::logic_error("SharedParticles: draw_frame called before cuda_signal_done for this frame");
    return 2 * frame_ + 1;
}

void SharedParticles::next_frame() {
    ++frame_;
    cuda_signaled_ = false;
}

}  // namespace flux
