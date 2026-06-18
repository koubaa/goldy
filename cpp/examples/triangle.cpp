/**
 * Triangle example - colored triangle in a window (cross-platform).
 *
 * Uses GLFW for windowing and retained Scheme:
 * offscreen lease -> render_pass -> copy_to_present -> grant_present.
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

#include <algorithm>
#include <chrono>
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
    goldy::Context ctx;
    goldy::Device device;
    goldy::RetainedPool pool;
    goldy::Buffer vertex_buffer;
    goldy::ShaderModule shader;
    goldy::RenderPipeline pipeline;
    goldy::SwapchainPool swapchain;
    goldy::PresentLease screen;
    goldy::Scheme scheme;
    goldy::SchemeRenderTargetLease scene_rt;
    goldy::PresentGrant present;
    uint64_t frame_count = 0;
    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
};

goldy::SwapchainPool create_swapchain_pool(const goldy::Context& ctx, GLFWwindow* window) {
#if defined(_WIN32)
    void* hwnd = glfwGetWin32Window(window);
    if (!hwnd) {
        throw std::runtime_error("glfwGetWin32Window failed");
    }
    return goldy::SwapchainPool(ctx, hwnd);
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
    return goldy::SwapchainPool(ctx, ns_view);
#else
    void* display = glfwGetWaylandDisplay();
    void* surface = glfwGetWaylandWindow(window);
    if (!display || !surface) {
        throw std::runtime_error(
            "Wayland handles unavailable — run under a Wayland session (Vulkan backend requires Wayland on Linux)");
    }
    return goldy::SwapchainPool(ctx, display, surface);
#endif
}

goldy::PresentGrant record_scheme(
    goldy::Scheme& scheme,
    const goldy::RenderPipeline& pipeline,
    const goldy::Buffer& vertex_buffer,
    goldy::SchemeRenderTargetLease& scene_rt,
    const goldy::PresentLease& screen,
    const goldy::Color& bg_color) {
    {
        auto pass = scheme.render_pass("triangle", scene_rt);
        pass.with_field(vertex_buffer, 0, goldy::NodeAccess::Read)
            .clear(bg_color)
            .set_pipeline(pipeline)
            .set_vertex_buffer(0, vertex_buffer)
            .draw(0, 3);
    }
    scheme.copy_to_present(scene_rt, screen);
    return scheme.grant_present(screen);
}

GpuState init_gpu(goldy::Device device, GLFWwindow* window) {
    const Vertex vertices[] = {
        {{0.0f, -0.5f}, {1.0f, 0.0f, 0.0f, 1.0f}},
        {{-0.5f, 0.5f}, {0.0f, 1.0f, 0.0f, 1.0f}},
        {{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f, 1.0f}},
    };

    goldy::Context ctx(device);
    goldy::RetainedPool pool(device);
    goldy::Buffer vertex_buffer = pool.acquire_buffer_with_data(
        std::span<const Vertex>(vertices),
        goldy::BufferKind::Scattered);

    goldy::SwapchainPool swapchain = create_swapchain_pool(ctx, window);
    goldy::PresentLease screen = swapchain.lease();

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
    pipeline_desc.target_format = swapchain.format();
    pipeline_desc.depth_enabled = false;

    goldy::RenderPipeline pipeline(device, shader, shader, pipeline_desc);

    goldy::Scheme scheme(ctx);
    auto [width, height] = swapchain.size();
    width = std::max(width, 1u);
    height = std::max(height, 1u);
    goldy::SchemeRenderTargetLease scene_rt =
        scheme.lease_render_target(width, height, swapchain.format());
    const goldy::Color bg_color{0.1f, 0.1f, 0.2f, 1.0f};
    goldy::PresentGrant present =
        record_scheme(scheme, pipeline, vertex_buffer, scene_rt, screen, bg_color);

    return GpuState{
        std::move(ctx),
        std::move(device),
        std::move(pool),
        std::move(vertex_buffer),
        std::move(shader),
        std::move(pipeline),
        std::move(swapchain),
        std::move(screen),
        std::move(scheme),
        std::move(scene_rt),
        std::move(present),
        0,
        std::chrono::steady_clock::now(),
    };
}

void render_frame(GpuState& gpu) {
    auto submission = gpu.scheme.submit();
    gpu.present.consume(submission);
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
    if (w == gpu.swapchain.width() && h == gpu.swapchain.height()) {
        return;
    }
    gpu.swapchain.resize(w, h);

    GoldyRenderPipelineDesc pipeline_desc{};
    pipeline_desc.topology = GOLDY_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    pipeline_desc.target_format = gpu.swapchain.format();
    pipeline_desc.depth_enabled = false;
    GoldyVertexAttribute attributes[] = {
        {0, GOLDY_VERTEX_FORMAT_FLOAT32X2, 0},
        {1, GOLDY_VERTEX_FORMAT_FLOAT32X4, static_cast<uint32_t>(sizeof(float) * 2)},
    };
    pipeline_desc.vertex_attributes = attributes;
    pipeline_desc.vertex_attribute_count = static_cast<uint32_t>(std::size(attributes));
    pipeline_desc.vertex_stride = sizeof(Vertex);
    gpu.pipeline = goldy::RenderPipeline(gpu.device, gpu.shader, gpu.shader, pipeline_desc);

    gpu.scheme = goldy::Scheme(gpu.ctx);
    auto [new_w, new_h] = gpu.swapchain.size();
    new_w = std::max(new_w, 1u);
    new_h = std::max(new_h, 1u);
    gpu.scene_rt = gpu.scheme.lease_render_target(new_w, new_h, gpu.swapchain.format());

    const goldy::Color bg_color{0.1f, 0.1f, 0.2f, 1.0f};
    gpu.present = record_scheme(
        gpu.scheme, gpu.pipeline, gpu.vertex_buffer, gpu.scene_rt, gpu.screen, bg_color);
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
        std::cout << "Goldy Triangle Window (C++ / Scheme + Present)\n";
        std::cout << "=============================================\n";
        std::cout << "Scheme: offscreen lease -> copy_to_present -> grant_present\n";
        std::cout << "Press Escape or close the window to exit\n\n";

        if (!glfwInit()) {
            throw std::runtime_error("glfwInit failed");
        }

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        GLFWwindow* window = glfwCreateWindow(800, 600, "Goldy - Triangle (C++)", nullptr, nullptr);
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
