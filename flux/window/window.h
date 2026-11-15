#pragma once
// GLFW window for the sandbox. No OpenGL context: Vulkan only.
#include <cstdint>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

#include "flux/camera_control.h"

struct GLFWwindow;

namespace flux {

class Window {
public:
    Window(int width, int height, const std::string& title);
    ~Window();
    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    bool should_close() const;
    void poll_events();
    void wait_events();

    // Size in pixels (may differ from window size on HiDPI screens).
    void framebuffer_size(int& w, int& h) const;

    // Set by the resize callback. The renderer reads and clears it.
    bool take_resized();

    // Instance extensions GLFW needs for a surface (VK_KHR_surface + platform one).
    static std::vector<const char*> required_instance_extensions();

    VkSurfaceKHR create_surface(VkInstance instance) const;

    GLFWwindow* handle() const { return window_; }

    // Cursor, buttons, and the wheel steps since the last call (then reset to 0).
    MouseInput take_mouse();

private:
    static void on_resize(GLFWwindow* w, int width, int height);
    static void on_scroll(GLFWwindow* w, double dx, double dy);
    double scroll_ = 0;
    GLFWwindow* window_ = nullptr;
    bool resized_ = false;
};

}  // namespace flux
