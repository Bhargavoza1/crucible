#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "flux/window/window.h"

#include <stdexcept>

#include "flux/vk_check.h"

namespace flux {

namespace {
void glfw_error(int code, const char* msg) { std::fprintf(stderr, "GLFW error %d: %s\n", code, msg); }
}  // namespace

Window::Window(int width, int height, const std::string& title) {
    glfwSetErrorCallback(glfw_error);
    if (!glfwInit()) throw std::runtime_error("glfwInit failed (no display? try xvfb-run)");
    if (!glfwVulkanSupported()) throw std::runtime_error("GLFW: Vulkan loader not found");

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);  // Vulkan, not OpenGL
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    window_ = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
    if (!window_) throw std::runtime_error("glfwCreateWindow failed");
    glfwSetWindowUserPointer(window_, this);
    glfwSetFramebufferSizeCallback(window_, on_resize);
    glfwSetScrollCallback(window_, on_scroll);
}

Window::~Window() {
    if (window_) glfwDestroyWindow(window_);
    glfwTerminate();
}

bool Window::should_close() const { return glfwWindowShouldClose(window_); }
void Window::poll_events() { glfwPollEvents(); }
void Window::wait_events() { glfwWaitEvents(); }

void Window::framebuffer_size(int& w, int& h) const { glfwGetFramebufferSize(window_, &w, &h); }

bool Window::take_resized() {
    bool r = resized_;
    resized_ = false;
    return r;
}

void Window::on_resize(GLFWwindow* w, int, int) {
    static_cast<Window*>(glfwGetWindowUserPointer(w))->resized_ = true;
}

void Window::on_scroll(GLFWwindow* w, double, double dy) { static_cast<Window*>(glfwGetWindowUserPointer(w))->scroll_ += dy; }

MouseInput Window::take_mouse() {
    MouseInput m;
    glfwGetCursorPos(window_, &m.x, &m.y);
    m.left = glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    m.right = glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
    m.middle = glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;
    m.scroll = scroll_;
    scroll_ = 0;
    return m;
}

std::vector<const char*> Window::required_instance_extensions() {
    uint32_t n = 0;
    const char** ext = glfwGetRequiredInstanceExtensions(&n);
    if (!ext) throw std::runtime_error("GLFW: no Vulkan surface extensions available");
    return std::vector<const char*>(ext, ext + n);
}

VkSurfaceKHR Window::create_surface(VkInstance instance) const {
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VK_CHECK(glfwCreateWindowSurface(instance, window_, nullptr, &surface));
    return surface;
}

}  // namespace flux
