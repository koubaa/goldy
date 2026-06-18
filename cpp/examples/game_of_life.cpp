/**
 * Game of Life — hybrid Scheme in a window (compute ping-pong + render + present).
 *
 * Build: cmake --build build --target game_of_life
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
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr uint32_t GRID_WIDTH = 128;
constexpr uint32_t GRID_HEIGHT = 128;
constexpr size_t CELL_COUNT = static_cast<size_t>(GRID_WIDTH) * GRID_HEIGHT;
constexpr uint32_t WORKGROUPS_X = (GRID_WIDTH + 7) / 8;
constexpr uint32_t WORKGROUPS_Y = (GRID_HEIGHT + 7) / 8;

constexpr uint32_t SLOT_A = 0;
constexpr uint32_t SLOT_B = 1;

struct DisplayState {
    goldy::Scheme scheme;
    goldy::SchemeRenderTargetLease scene_rt;
    goldy::PresentGrant present;
};

struct GpuState {
    goldy::Context ctx;
    goldy::Device device;
    goldy::RetainedPool pool;
    goldy::Buffer cells;
    goldy::ShaderModule compute_shader;
    goldy::ShaderModule render_shader;
    goldy::ComputePipeline compute_pipeline;
    goldy::RenderPipeline render_pipeline;
    goldy::SwapchainPool swapchain;
    goldy::PresentLease screen;
    DisplayState display;
    bool use_buffer_a = true;
    uint64_t frame_count = 0;
    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point last_update = std::chrono::steady_clock::now();
};

std::string read_text_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open: " + path);
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::string find_shader(const char* name) {
    const std::vector<std::string> candidates = {
        std::string("shaders/") + name,
        std::string("../shaders/") + name,
        std::string("../../shaders/") + name,
        std::string("../../../shaders/") + name,
        std::string("../../../../shaders/") + name,
        std::string("../../../../../shaders/") + name,
        std::string("../../../../../../shaders/") + name,
    };
    for (const auto& path : candidates) {
        std::ifstream probe(path);
        if (probe.good()) {
            return read_text_file(path);
        }
    }
    throw std::runtime_error(std::string("Could not find shader: ") + name);
}

int demo_frame_limit() {
    if (const char* raw = std::getenv("GOLDY_DEMO_FRAMES")) {
        return std::max(1, std::atoi(raw));
    }
    return 0;
}

std::vector<uint32_t> create_initial_state() {
    std::vector<uint32_t> cells(CELL_COUNT, 0);

    const int gun[][2] = {
        {1, 5},   {1, 6},   {2, 5},   {2, 6},   {11, 5},  {11, 6},  {11, 7},  {12, 4},
        {12, 8},  {13, 3},  {13, 9},  {14, 3},  {14, 9},  {15, 6},  {16, 4},  {16, 8},
        {17, 5},  {17, 6},  {17, 7},  {18, 6},  {21, 3},  {21, 4},  {21, 5},  {22, 3},
        {22, 4},  {22, 5},  {23, 2},  {23, 6},  {25, 1},  {25, 2},  {25, 6},  {25, 7},
        {35, 3},  {35, 4},  {36, 3},  {36, 4},
    };

    constexpr int offset_x = 10;
    constexpr int offset_y = 10;
    for (const auto& pair : gun) {
        const int px = pair[0] + offset_x;
        const int py = pair[1] + offset_y;
        if (px >= 0 && py >= 0 && static_cast<uint32_t>(px) < GRID_WIDTH &&
            static_cast<uint32_t>(py) < GRID_HEIGHT) {
            cells[static_cast<size_t>(py) * GRID_WIDTH + static_cast<size_t>(px)] = 1;
        }
    }

    uint64_t rng = 42;
    for (int y = 60; y < 100; ++y) {
        for (int x = 60; x < 100; ++x) {
            rng = rng * 6364136223846793005ULL + 1ULL;
            if ((rng >> 32) % 4 == 0) {
                cells[static_cast<size_t>(y) * GRID_WIDTH + static_cast<size_t>(x)] = 1;
            }
        }
    }

    return cells;
}

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

void run_compute_step(
    const goldy::Context& ctx,
    const goldy::Buffer& cells,
    uint32_t read_slot,
    uint32_t write_slot,
    const goldy::ComputePipeline& pipeline) {
    goldy::Scheme scheme(ctx);
    {
        auto node = scheme.compute_node("game_of_life", pipeline);
        node.with_field(
            cells, read_slot, goldy::NodeAccess::Read, goldy::ResourceAccess::ReadWrite);
        node.with_field(
            cells, write_slot, goldy::NodeAccess::Write, goldy::ResourceAccess::Write);
        node.dispatch(WORKGROUPS_X, WORKGROUPS_Y, 1);
    }
    scheme.submit();
}

DisplayState record_display_scheme(
    const goldy::Context& ctx,
    const goldy::SwapchainPool& swapchain,
    const goldy::Buffer& cells,
    uint32_t current_slot,
    const goldy::RenderPipeline& render_pipeline,
    const goldy::PresentLease& screen) {
    goldy::Scheme scheme(ctx);
    auto [width, height] = swapchain.size();
    width = std::max(width, 1u);
    height = std::max(height, 1u);
    goldy::SchemeRenderTargetLease scene_rt =
        scheme.lease_render_target(width, height, swapchain.format());
    const uint32_t cells_idx =
        cells.unit_resource_index(current_slot, goldy::ResourceAccess::ReadWrite);
    {
        auto pass = scheme.render_pass("game_of_life_render", scene_rt);
        pass.with_field(cells, current_slot, goldy::NodeAccess::Read)
            .clear(goldy::Color::black())
            .set_pipeline(render_pipeline)
            .bind_resource_index(cells_idx)
            .draw_fullscreen();
    }
    scheme.copy_to_present(scene_rt, screen);
    goldy::PresentGrant present = scheme.grant_present(screen);
    return DisplayState{std::move(scheme), std::move(scene_rt), std::move(present)};
}

GpuState init_gpu(goldy::Device device, GLFWwindow* window) {
    const auto initial = create_initial_state();
    const std::vector<uint32_t> zeros(CELL_COUNT, 0);

    goldy::Context ctx(device);
    goldy::RetainedPool pool(device);
    auto record = pool.record();
    record.emplace_named("a", initial);
    record.emplace_named("b", zeros);
    goldy::Buffer cells = record.build(pool);

    goldy::ShaderModule compute_shader(device, find_shader("game_of_life.slang"));
    goldy::ShaderModule render_shader(device, find_shader("game_of_life_render.slang"));
    goldy::ComputePipeline compute_pipeline(device, compute_shader);

    goldy::SwapchainPool swapchain = create_swapchain_pool(ctx, window);
    goldy::PresentLease screen = swapchain.lease();

    GoldyRenderPipelineDesc render_desc{};
    render_desc.topology = GOLDY_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    render_desc.target_format = swapchain.format();
    render_desc.depth_enabled = false;
    goldy::RenderPipeline render_pipeline(device, render_shader, render_shader, render_desc);

    DisplayState display = record_display_scheme(
        ctx, swapchain, cells, SLOT_A, render_pipeline, screen);

    return GpuState{
        std::move(ctx),
        std::move(device),
        std::move(pool),
        std::move(cells),
        std::move(compute_shader),
        std::move(render_shader),
        std::move(compute_pipeline),
        std::move(render_pipeline),
        std::move(swapchain),
        std::move(screen),
        std::move(display),
        true,
        0,
        std::chrono::steady_clock::now(),
        std::chrono::steady_clock::now(),
    };
}

void render_frame(GpuState& gpu) {
    const auto now = std::chrono::steady_clock::now();
    const bool should_update =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - gpu.last_update).count() > 33;

    if (should_update) {
        gpu.last_update = now;

        const uint32_t read_slot = gpu.use_buffer_a ? SLOT_A : SLOT_B;
        const uint32_t write_slot = gpu.use_buffer_a ? SLOT_B : SLOT_A;

        run_compute_step(gpu.ctx, gpu.cells, read_slot, write_slot, gpu.compute_pipeline);
        gpu.use_buffer_a = !gpu.use_buffer_a;

        const uint32_t current_slot = gpu.use_buffer_a ? SLOT_A : SLOT_B;
        gpu.display = record_display_scheme(
            gpu.ctx, gpu.swapchain, gpu.cells, current_slot, gpu.render_pipeline, gpu.screen);
    }

    auto submission = gpu.display.scheme.submit();
    gpu.display.present.consume(submission);
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

    GoldyRenderPipelineDesc render_desc{};
    render_desc.topology = GOLDY_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    render_desc.target_format = gpu.swapchain.format();
    render_desc.depth_enabled = false;
    gpu.render_pipeline =
        goldy::RenderPipeline(gpu.device, gpu.render_shader, gpu.render_shader, render_desc);

    const uint32_t current_slot = gpu.use_buffer_a ? SLOT_A : SLOT_B;
    gpu.display = record_display_scheme(
        gpu.ctx, gpu.swapchain, gpu.cells, current_slot, gpu.render_pipeline, gpu.screen);
}

void print_perf(const GpuState& gpu) {
    const auto elapsed = std::chrono::steady_clock::now() - gpu.start_time;
    const auto sec = std::chrono::duration<double>(elapsed).count();
    if (sec > 0.0 && gpu.frame_count > 0) {
        const auto fps = static_cast<double>(gpu.frame_count) / sec;
        std::cout << "GOLDY_PERF: frames=" << gpu.frame_count << " elapsed=" << sec
                  << "s avg_fps=" << fps << '\n';
    }
}

} // namespace

int main() {
    try {
        std::cout << "Goldy Game of Life (C++ / Scheme + Present)\n";
        std::cout << "===========================================\n";
        std::cout << "Scheme: compute ping-pong + render + copy_to_present\n";
        std::cout << "Press Escape or close the window to exit\n\n";

        if (!glfwInit()) {
            throw std::runtime_error("glfwInit failed");
        }

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        GLFWwindow* window =
            glfwCreateWindow(800, 800, "Goldy - Game of Life (C++)", nullptr, nullptr);
        if (!window) {
            glfwTerminate();
            throw std::runtime_error("glfwCreateWindow failed");
        }

        goldy::Instance instance;
        goldy::Device device = instance.request_adapter().request_device();
        GpuState gpu = init_gpu(std::move(device), window);

        const int frame_limit = demo_frame_limit();

        while (!glfwWindowShouldClose(window)) {
            handle_resize(gpu, window);
            render_frame(gpu);

            if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
                glfwSetWindowShouldClose(window, GLFW_TRUE);
            }
            if (frame_limit > 0 && static_cast<int>(gpu.frame_count) >= frame_limit) {
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
