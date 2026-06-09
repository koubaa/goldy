/**
 * Triangle example - animated colored triangle in a window (cross-platform).
 *
 * Uses GLFW for windowing and Goldy TaskGraph:
 * offscreen RenderTarget -> render_pass -> copy_render_target_to_swapchain -> present.
 *
 * Build: cmake --build build --target triangle
 */

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <goldy.hpp>

#include <GLFW/glfw3.h>

#if defined(_WIN32)
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#elif defined(__APPLE__)
#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3native.h>
#include <objc/message.h>
#include <objc/objc.h>
#include <objc/runtime.h>
#else
#define GLFW_EXPOSE_NATIVE_WAYLAND
#include <GLFW/glfw3native.h>
#endif

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace {

struct Vertex {
    float position[2];
    float color[4];
};

struct GpuState {
    goldy::Device device;
    goldy::Buffer vertex_buffer;
    goldy::ShaderModule shader;
    goldy::RenderPipeline pipeline;
    goldy::RenderTarget scene_rt;
    goldy::Surface surface;
    goldy::TaskGraph frame_graph;
    uint64_t frame_count = 0;
    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
};

goldy::RenderTarget make_scene_rt(const goldy::Device& device, const goldy::Surface& surface) {
    auto [width, height] = surface.size();
    width = std::max(width, 1u);
    height = std::max(height, 1u);
    return goldy::RenderTarget(device, width, height, surface.format());
}

goldy::Surface create_surface(const goldy::Device& device, GLFWwindow* window) {
#if defined(_WIN32)
    void* hwnd = glfwGetWin32Window(window);
    if (!hwnd) {
        throw std::runtime_error("glfwGetWin32Window failed");
    }
    return goldy::Surface(device, hwnd);
#elif defined(__APPLE__)
    void* ns_window = glfwGetCocoaWindow(window);
    if (!ns_window) {
        throw std::runtime_error("glfwGetCocoaWindow failed");
    }
    using MsgSendFn = id (*)(id, SEL);
    void* ns_view = reinterpret_cast<void*>(
        ((MsgSendFn)objc_msgSend)(reinterpret_cast<id>(ns_window), sel_registerName("contentView")));
    if (!ns_view) {
        throw std::runtime_error("NSWindow contentView is null");
    }
    return goldy::Surface(device, ns_view);
#else
    void* display = glfwGetWaylandDisplay();
    void* surface = glfwGetWaylandWindow(window);
    if (!display || !surface) {
        throw std::runtime_error(
            "Wayland handles unavailable — run under a Wayland session (Vulkan backend requires Wayland on Linux)");
    }
    return goldy::Surface(device, display, surface);
#endif
}

GpuState init_gpu(goldy::Device device, GLFWwindow* window) {
    const Vertex vertices[] = {
        {{0.0f, -0.5f}, {1.0f, 0.0f, 0.0f, 1.0f}},
        {{-0.5f, 0.5f}, {0.0f, 1.0f, 0.0f, 1.0f}},
        {{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f, 1.0f}},
    };

    goldy::Buffer vertex_buffer(
        device,
        std::span<const Vertex>(vertices),
        goldy::BufferKind::Scattered);

    goldy::Surface surface = create_surface(device, window);

    goldy::ShaderModule shader(device, goldy::ShaderModule::builtin_vertex_color_2d());

    GoldyVertexAttribute attributes[] = {
        {0, GOLDY_VERTEX_FORMAT_FLOAT32X2, 0},
        {1, GOLDY_VERTEX_FORMAT_FLOAT32X4, static_cast<uint32_t>(sizeof(float) * 2)},
    };

    GoldyRenderPipelineDesc pipeline_desc{};
    pipeline_desc.vertex_attributes = attributes;
    pipeline_desc.vertex_attribute_count = static_cast<uint32_t>(std::size(attributes));
    pipeline_desc.vertex_stride = sizeof(Vertex);
    pipeline_desc.topology = GOLDY_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    pipeline_desc.target_format = surface.format();
    pipeline_desc.depth_enabled = false;

    goldy::RenderPipeline pipeline(device, shader, shader, pipeline_desc);
    goldy::RenderTarget scene_rt = make_scene_rt(device, surface);

    return GpuState{
        std::move(device),
        std::move(vertex_buffer),
        std::move(shader),
        std::move(pipeline),
        std::move(scene_rt),
        std::move(surface),
        goldy::TaskGraph{},
        0,
        std::chrono::steady_clock::now(),
    };
}

void render_frame(GpuState& gpu) {
    const float t = std::sin(static_cast<float>(gpu.frame_count) * 0.02f) * 0.5f + 0.5f;
    const goldy::Color bg_color{
        0.1f + t * 0.1f,
        0.1f + t * 0.05f,
        0.2f + t * 0.1f,
        1.0f,
    };

    gpu.frame_graph.clear();

    {
        auto pass = gpu.frame_graph.render_pass("triangle", gpu.scene_rt);
        pass.bind_buffer(gpu.vertex_buffer, goldy::NodeAccess::Read)
            .clear(bg_color)
            .set_pipeline(gpu.pipeline)
            .set_vertex_buffer(0, gpu.vertex_buffer)
            .draw(0, 3);
    }

    const auto swapchain = gpu.frame_graph.declare_swapchain_output();
    gpu.frame_graph.copy_render_target_to_swapchain(gpu.scene_rt, swapchain);

    auto frame = gpu.surface.begin();
    frame = gpu.surface.submit_graph_to_frame(gpu.frame_graph, std::move(frame));
    gpu.surface.present(std::move(frame));

    ++gpu.frame_count;
}

void handle_resize(GpuState& gpu, GLFWwindow* window) {
    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window, &width, &height);
    if (width <= 0 || height <= 0) {
        return;
    }
    const auto w = static_cast<uint32_t>(width);
    const auto h = static_cast<uint32_t>(height);
    if (w == gpu.surface.width() && h == gpu.surface.height()) {
        return;
    }
    gpu.surface.resize(w, h);
    gpu.scene_rt = make_scene_rt(gpu.device, gpu.surface);
}

void print_perf(const GpuState& gpu) {
    const auto elapsed = std::chrono::steady_clock::now() - gpu.start_time;
    const auto sec = std::chrono::duration<double>(elapsed).count();
    if (sec > 0.0 && gpu.frame_count > 0) {
        const auto fps = static_cast<double>(gpu.frame_count) / sec;
        std::cout << "\nRendered " << gpu.frame_count << " frames in " << sec << "s ("
                  << fps << " FPS)\n";
    }
}

} // namespace

int main() {
    try {
        std::cout << "Goldy Triangle Window (C++ / GLFW)\n";
        std::cout << "==================================\n";
        std::cout << "TaskGraph: offscreen RT -> swapchain blit -> present\n";
        std::cout << "Press Escape or close the window to exit\n\n";

        if (!glfwInit()) {
            throw std::runtime_error("glfwInit failed");
        }

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        GLFWwindow* window = glfwCreateWindow(800, 600, "Goldy - Animated Triangle (C++)", nullptr, nullptr);
        if (!window) {
            glfwTerminate();
            throw std::runtime_error("glfwCreateWindow failed");
        }

        goldy::Instance instance;
        goldy::Device device = instance.request_adapter().request_device();
        GpuState gpu = init_gpu(std::move(device), window);

        while (!glfwWindowShouldClose(window)) {
            handle_resize(gpu, window);
            render_frame(gpu);

            if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
                glfwSetWindowShouldClose(window, GLFW_TRUE);
            }
            glfwPollEvents();
        }

        print_perf(gpu);
        glfwDestroyWindow(window);
        glfwTerminate();
        return 0;
    } catch (const goldy::Exception& e) {
        std::cerr << "Goldy error: " << e.what() << '\n';
        glfwTerminate();
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        glfwTerminate();
        return 1;
    }
}
