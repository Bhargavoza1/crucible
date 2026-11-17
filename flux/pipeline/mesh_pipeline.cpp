#include "flux/pipeline/mesh_pipeline.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <stdexcept>

#include "flux/context.h"
#include "flux/pipeline/particle_pipeline.h"  // load_shader
#include "flux/vk_check.h"

namespace flux {

namespace {

struct MeshPush {
    Mat4 view_proj;
    float grid[4];
    int32_t info[4];
    float eye[4];
};
static_assert(sizeof(MeshPush) == 112, "must match mesh.vert");

}  // namespace

MeshDrawer::MeshDrawer(const Context& ctx, VkRenderPass render_pass, const std::string& shader_dir, int slots,
                       VkDescriptorSetLayout lighting, VkSampleCountFlagBits samples, VkRenderPass shadow_pass)
    : ctx_(ctx) {
    VkDevice dev = ctx.device;
    VkDescriptorSetLayoutBinding b{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dl.bindingCount = 1;
    dl.pBindings = &b;
    VK_CHECK(vkCreateDescriptorSetLayout(dev, &dl, nullptr, &set_layout_));
    VkPushConstantRange push{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(MeshPush)};
    const VkDescriptorSetLayout pipe_sets[2] = {set_layout_, lighting};  // set 0: poses, set 1: lighting
    VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    lci.setLayoutCount = 2;
    lci.pSetLayouts = pipe_sets;
    lci.pushConstantRangeCount = 1;
    lci.pPushConstantRanges = &push;
    VK_CHECK(vkCreatePipelineLayout(dev, &lci, nullptr, &layout_));
    pipeline_ = make_pipeline(render_pass, shader_dir, samples, false);
    if (shadow_pass != VK_NULL_HANDLE)
        shadow_pipeline_ = make_pipeline(shadow_pass, shader_dir, VK_SAMPLE_COUNT_1_BIT, true);

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, uint32_t(slots)};
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets = uint32_t(slots);
    pci.poolSizeCount = 1;
    pci.pPoolSizes = &ps;
    VK_CHECK(vkCreateDescriptorPool(dev, &pci, nullptr, &pool_));
    sets_.resize(slots);
    std::vector<VkDescriptorSetLayout> layouts(slots, set_layout_);
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = pool_;
    ai.descriptorSetCount = uint32_t(slots);
    ai.pSetLayouts = layouts.data();
    VK_CHECK(vkAllocateDescriptorSets(dev, &ai, sets_.data()));
    poses_.resize(slots);
}

VkPipeline MeshDrawer::make_pipeline(VkRenderPass pass, const std::string& shader_dir, VkSampleCountFlagBits samples,
                                     bool shadow) const {
    VkDevice dev = ctx_.device;
    VkShaderModule vert = load_shader(dev, shader_dir + "/mesh.vert.spv");
    VkShaderModule frag = shadow ? VK_NULL_HANDLE : load_shader(dev, shader_dir + "/mesh.frag.spv");
    VkPipelineShaderStageCreateInfo stages[2]{};
    for (int i = 0; i < 2; ++i) {
        stages[i].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[i].stage = i == 0 ? VK_SHADER_STAGE_VERTEX_BIT : VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[i].module = i == 0 ? vert : frag;
        stages[i].pName = "main";
    }
    VkVertexInputBindingDescription binding{0, sizeof(MeshVertex), VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription attrs[4] = {
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(MeshVertex, pos)},
        {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(MeshVertex, normal)},
        {2, 0, VK_FORMAT_R8G8B8A8_UNORM, offsetof(MeshVertex, rgba)},
        {3, 0, VK_FORMAT_R32_UINT, offsetof(MeshVertex, link)},
    };
    VkPipelineVertexInputStateCreateInfo vin{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vin.vertexBindingDescriptionCount = 1;
    vin.pVertexBindingDescriptions = &binding;
    vin.vertexAttributeDescriptionCount = 4;
    vin.pVertexAttributeDescriptions = attrs;
    VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport.viewportCount = 1;
    viewport.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1.0f;
    if (shadow) {  // slope-scaled depth bias against shadow acne
        raster.depthBiasEnable = VK_TRUE;
        raster.depthBiasConstantFactor = 1.5f;
        raster.depthBiasSlopeFactor = 2.0f;
    }
    VkPipelineMultisampleStateCreateInfo msaa{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    msaa.rasterizationSamples = samples;
    VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depth.depthTestEnable = VK_TRUE;
    depth.depthWriteEnable = VK_TRUE;
    depth.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    VkPipelineColorBlendAttachmentState blend_att{};
    blend_att.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                               VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blend.attachmentCount = shadow ? 0 : 1;
    blend.pAttachments = &blend_att;
    VkDynamicState dyn_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates = dyn_states;
    VkGraphicsPipelineCreateInfo ci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    ci.stageCount = shadow ? 1 : 2;
    ci.pStages = stages;
    ci.pVertexInputState = &vin;
    ci.pInputAssemblyState = &assembly;
    ci.pViewportState = &viewport;
    ci.pRasterizationState = &raster;
    ci.pMultisampleState = &msaa;
    ci.pDepthStencilState = &depth;
    ci.pColorBlendState = &blend;
    ci.pDynamicState = &dynamic;
    ci.layout = layout_;
    ci.renderPass = pass;
    VkPipeline p = VK_NULL_HANDLE;
    VK_CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &ci, nullptr, &p));
    vkDestroyShaderModule(dev, vert, nullptr);
    if (frag) vkDestroyShaderModule(dev, frag, nullptr);
    return p;
}

MeshDrawer::~MeshDrawer() {
    VkDevice dev = ctx_.device;
    for (auto& p : poses_) destroy(ctx_, p);
    if (vbuf_) vkDestroyBuffer(dev, vbuf_, nullptr);
    if (vmem_) vkFreeMemory(dev, vmem_, nullptr);
    if (pool_) vkDestroyDescriptorPool(dev, pool_, nullptr);
    if (pipeline_) vkDestroyPipeline(dev, pipeline_, nullptr);
    if (shadow_pipeline_) vkDestroyPipeline(dev, shadow_pipeline_, nullptr);
    if (layout_) vkDestroyPipelineLayout(dev, layout_, nullptr);
    if (set_layout_) vkDestroyDescriptorSetLayout(dev, set_layout_, nullptr);
}

void MeshDrawer::set_mesh(const RobotMesh& mesh, int lod) {
    if (mesh.lods.empty()) throw std::runtime_error("MeshDrawer: empty mesh");
    const auto& v = mesh.lods[size_t(std::min<int>(std::max(lod, 0), int(mesh.lods.size()) - 1))];
    VkDevice dev = ctx_.device;
    vkDeviceWaitIdle(dev);
    if (vbuf_) vkDestroyBuffer(dev, vbuf_, nullptr), vbuf_ = VK_NULL_HANDLE;
    if (vmem_) vkFreeMemory(dev, vmem_, nullptr), vmem_ = VK_NULL_HANDLE;
    links_ = mesh.num_links();
    vertex_count_ = uint32_t(v.size());
    if (v.empty()) return;
    const VkDeviceSize bytes = VkDeviceSize(v.size()) * sizeof(MeshVertex);
    // device-local vertex buffer, filled once through a host staging buffer (every
    // camera tile reads the whole mesh: it must not come over the PCIe bus each time)
    HostBuffer staging = create_host_buffer(ctx_, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    std::memcpy(staging.mapped, v.data(), bytes);
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = bytes;
    bi.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VK_CHECK(vkCreateBuffer(dev, &bi, nullptr, &vbuf_));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev, vbuf_, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = find_memory_type(ctx_, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(dev, &mai, nullptr, &vmem_));
    VK_CHECK(vkBindBufferMemory(dev, vbuf_, vmem_, 0));
    VkCommandPool cp;
    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.queueFamilyIndex = ctx_.queue_family;
    VK_CHECK(vkCreateCommandPool(dev, &cpi, nullptr, &cp));
    VkCommandBuffer cmd;
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = cp;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(dev, &cai, &cmd));
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &begin));
    VkBufferCopy copy{0, 0, bytes};
    vkCmdCopyBuffer(cmd, staging.buffer, vbuf_, 1, &copy);
    VK_CHECK(vkEndCommandBuffer(cmd));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    VK_CHECK(vkQueueSubmit(ctx_.queue, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(ctx_.queue));
    vkDestroyCommandPool(dev, cp, nullptr);
    destroy(ctx_, staging);
}

void MeshDrawer::ensure_pose_capacity(int slot, size_t bytes) {
    HostBuffer& p = poses_[size_t(slot)];
    if (p.buffer && p.size >= bytes) return;
    vkDeviceWaitIdle(ctx_.device);  // rare: the number of envs grew
    destroy(ctx_, p);
    p = create_host_buffer(ctx_, std::max<size_t>(bytes, 256), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    VkDescriptorBufferInfo info{p.buffer, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = sets_[size_t(slot)];
    w.dstBinding = 0;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w.pBufferInfo = &info;
    vkUpdateDescriptorSets(ctx_.device, 1, &w, 0, nullptr);
}

void MeshDrawer::set_poses(int slot, const float* poses, int num_envs) {
    const size_t bytes = size_t(num_envs) * links_ * 12 * sizeof(float);
    ensure_pose_capacity(slot, bytes);
    std::memcpy(poses_[size_t(slot)].mapped, poses, bytes);
}

void MeshDrawer::push_and_draw(VkCommandBuffer cmd, VkPipeline pipe, int slot, const Mat4& m, Vec3 eye, bool grid,
                               float spacing, int per_row, int rows, int first_env, int count) const {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout_, 0, 1, &sets_[size_t(slot)], 0, nullptr);
    MeshPush pc{m, {spacing, float(per_row), float(rows), grid ? 1.0f : 0.0f}, {first_env, links_, 0, 0},
                {eye.x, eye.y, eye.z, 0.0f}};
    vkCmdPushConstants(cmd, layout_, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);
    VkDeviceSize off = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &vbuf_, &off);
    vkCmdDraw(cmd, vertex_count_, uint32_t(count), 0, 0);
}

void MeshDrawer::draw(VkCommandBuffer cmd, int slot, const Mat4& view_proj, Vec3 eye, bool grid, float spacing,
                      int per_row, int rows, int first_env, int count, VkDescriptorSet lighting) const {
    if (!vertex_count_ || count <= 0 || !poses_[size_t(slot)].buffer) return;
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout_, 1, 1, &lighting, 0, nullptr);
    push_and_draw(cmd, pipeline_, slot, view_proj, eye, grid, spacing, per_row, rows, first_env, count);
}

void MeshDrawer::draw_shadow(VkCommandBuffer cmd, int slot, const Mat4& light_vp, bool grid, float spacing,
                             int per_row, int rows, int first_env, int count) const {
    if (!shadow_pipeline_ || !vertex_count_ || count <= 0 || !poses_[size_t(slot)].buffer) return;
    push_and_draw(cmd, shadow_pipeline_, slot, light_vp, Vec3{0, 0, 0}, grid, spacing, per_row, rows, first_env, count);
}

}  // namespace flux
