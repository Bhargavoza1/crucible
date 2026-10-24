#include "flux/context.h"

#include <cstring>
#include <stdexcept>
#include <string>

#include "flux/vk_check.h"

namespace flux {

const char* vk_result_name(VkResult r) {
    switch (r) {
        case VK_SUCCESS: return "VK_SUCCESS";
        case VK_NOT_READY: return "VK_NOT_READY";
        case VK_TIMEOUT: return "VK_TIMEOUT";
        case VK_SUBOPTIMAL_KHR: return "VK_SUBOPTIMAL_KHR";
        case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
        case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
        case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
        case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
        case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
        case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
        case VK_ERROR_SURFACE_LOST_KHR: return "VK_ERROR_SURFACE_LOST_KHR";
        case VK_ERROR_OUT_OF_DATE_KHR: return "VK_ERROR_OUT_OF_DATE_KHR";
        default: return "VK_ERROR_(other)";
    }
}

namespace {

const char* kValidationLayer = "VK_LAYER_KHRONOS_validation";

bool layer_available(const char* name) {
    uint32_t n = 0;
    vkEnumerateInstanceLayerProperties(&n, nullptr);
    std::vector<VkLayerProperties> props(n);
    vkEnumerateInstanceLayerProperties(&n, props.data());
    for (const auto& p : props)
        if (std::strcmp(p.layerName, name) == 0) return true;
    return false;
}

VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                              VkDebugUtilsMessageTypeFlagsEXT,
                                              const VkDebugUtilsMessengerCallbackDataEXT* data, void*) {
    const char* tag = (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) ? "ERROR" : "WARNING";
    std::fprintf(stderr, "[vulkan %s] %s\n", tag, data->pMessage);
    return VK_FALSE;
}

VkDebugUtilsMessengerCreateInfoEXT messenger_info() {
    VkDebugUtilsMessengerCreateInfoEXT ci{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    ci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                         VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    ci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                     VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                     VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    ci.pfnUserCallback = debug_callback;
    return ci;
}

const char* device_type_name(VkPhysicalDeviceType t) {
    switch (t) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return "discrete";
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "integrated";
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return "virtual";
        case VK_PHYSICAL_DEVICE_TYPE_CPU: return "cpu";
        default: return "other";
    }
}

}  // namespace

void Context::create_instance(const ContextOptions& opt) {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "crucible";
    app.pEngineName = "flux";
    app.apiVersion = VK_API_VERSION_1_2;  // doc 2.3: Vulkan 1.2+

    std::vector<const char*> exts = opt.instance_extensions;
    std::vector<const char*> layers;
    validation_on = opt.validation && layer_available(kValidationLayer);
    if (opt.validation && !validation_on)
        std::fprintf(stderr, "flux: validation layer not installed, running without it\n");
    if (validation_on) {
        layers.push_back(kValidationLayer);
        exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo = &app;
    ci.enabledExtensionCount = static_cast<uint32_t>(exts.size());
    ci.ppEnabledExtensionNames = exts.data();
    ci.enabledLayerCount = static_cast<uint32_t>(layers.size());
    ci.ppEnabledLayerNames = layers.data();
    VkDebugUtilsMessengerCreateInfoEXT dbg = messenger_info();
    if (validation_on) ci.pNext = &dbg;  // also catch errors during vkCreateInstance
    VK_CHECK(vkCreateInstance(&ci, nullptr, &instance));

    if (validation_on) {
        auto create = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
        if (create) VK_CHECK(create(instance, &dbg, nullptr, &messenger_));
    }
}

void Context::pick_physical_device() {
    uint32_t n = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &n, nullptr));
    if (n == 0) throw std::runtime_error("flux: no Vulkan device found");
    std::vector<VkPhysicalDevice> devs(n);
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &n, devs.data()));

    // Score: must have swapchain + a queue family with graphics and present.
    // Prefer a discrete GPU (the 4070 Ti), so a laptop iGPU is never picked by mistake.
    int best_score = -1;
    for (VkPhysicalDevice d : devs) {
        uint32_t en = 0;
        vkEnumerateDeviceExtensionProperties(d, nullptr, &en, nullptr);
        std::vector<VkExtensionProperties> ext(en);
        vkEnumerateDeviceExtensionProperties(d, nullptr, &en, ext.data());
        bool has_swapchain = false;
        for (const auto& e : ext)
            if (std::strcmp(e.extensionName, VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0) has_swapchain = true;
        if (!has_swapchain) continue;

        uint32_t qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(d, &qn, nullptr);
        std::vector<VkQueueFamilyProperties> qf(qn);
        vkGetPhysicalDeviceQueueFamilyProperties(d, &qn, qf.data());
        int family = -1;
        for (uint32_t i = 0; i < qn; ++i) {
            VkBool32 present = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(d, i, surface, &present);
            if ((qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && present) {
                family = static_cast<int>(i);
                break;
            }
        }
        if (family < 0) continue;

        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(d, &p);
        int score = (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) ? 100
                  : (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) ? 50 : 1;
        if (score > best_score) {
            best_score = score;
            physical = d;
            queue_family = static_cast<uint32_t>(family);
        }
    }
    if (!physical) throw std::runtime_error("flux: no Vulkan device can draw to this window");

    VkPhysicalDeviceIDProperties id{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    p2.pNext = &id;
    vkGetPhysicalDeviceProperties2(physical, &p2);
    std::memcpy(device_uuid, id.deviceUUID, VK_UUID_SIZE);

    std::string uuid;
    char hex[3];
    for (uint32_t i = 0; i < VK_UUID_SIZE; ++i) {
        std::snprintf(hex, sizeof(hex), "%02x", device_uuid[i]);
        uuid += hex;
    }
    std::printf("flux: %s (%s), Vulkan %u.%u, uuid %s, validation %s\n", p2.properties.deviceName,
                device_type_name(p2.properties.deviceType), VK_API_VERSION_MAJOR(p2.properties.apiVersion),
                VK_API_VERSION_MINOR(p2.properties.apiVersion), uuid.c_str(), validation_on ? "on" : "off");
}

void Context::create_device() {
    float priority = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = queue_family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;

    const char* exts[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    VkDeviceCreateInfo ci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    ci.queueCreateInfoCount = 1;
    ci.pQueueCreateInfos = &qci;
    ci.enabledExtensionCount = 1;
    ci.ppEnabledExtensionNames = exts;
    VK_CHECK(vkCreateDevice(physical, &ci, nullptr, &device));
    vkGetDeviceQueue(device, queue_family, 0, &queue);
}

Context::~Context() {
    if (device) vkDestroyDevice(device, nullptr);
    if (surface) vkDestroySurfaceKHR(instance, surface, nullptr);
    if (messenger_) {
        auto destroy = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
        if (destroy) destroy(instance, messenger_, nullptr);
    }
    if (instance) vkDestroyInstance(instance, nullptr);
}

}  // namespace flux
