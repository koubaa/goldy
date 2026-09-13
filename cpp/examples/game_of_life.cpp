/**
 * Game of Life � two retained schemes, alternating resubmit.
 *
 * Ping-pong cell grids live in one retained record buffer (fields "a" / "b").
 * Orientation AB reads a and writes b; BA is the swap. Each is recorded once
 * (and again on resize). Simulation steps alternate which scheme submits. Idle
 * polls skip submit and leave the last present on the surface.
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
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr uint32_t GRID_WIDTH = 128;
constexpr uint32_t GRID_HEIGHT = 128;
constexpr size_t CELL_COUNT = static_cast<size_t>(GRID_WIDTH) * GRID_HEIGHT;
constexpr uint32_t WORKGROUPS_X = (GRID_WIDTH + 7) / 8;
constexpr uint32_t WORKGROUPS_Y = (GRID_HEIGHT + 7) / 8;

uint32_t field_unit(const char* name) {
    if (std::string_view(name) == "a") {
        return 0;
    }
    if (std::string_view(name) == "b") {
        return 1;
    }
    throw std::runtime_error(std::string("unknown field: ") + name);
}

struct GpuState {
    goldy::Context ctx;
    goldy::Device device;
    goldy::RetainedPool pool;
    goldy::Buffer cells;
    goldy::ShaderModule compute_shader;
    goldy::ShaderModule render_shader;
    goldy::ComputePipeline compute_pipeline;
    goldy::RenderPipeline render_pipeline;
    goldy::SurfaceExchange exchange;
    goldy::Scheme scheme_ab;
    goldy::Scheme scheme_ba;
    goldy::Transaction present_ab;
    goldy::Transaction present_ba;
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

goldy::SurfaceExchange create_surface_exchange(const goldy::Context& ctx, GLFWwindow* window) {
#if defined(_WIN32)
    void* hwnd = glfwGetWin32Window(window);
    if (!hwnd) {
        throw std::runtime_error("glfwGetWin32Window failed");
    }
    return goldy::SurfaceExchange(ctx, hwnd);
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
    return goldy::SurfaceExchange(ctx, ns_view);
#else
    void* display = glfwGetWaylandDisplay();
    void* surface = glfwGetWaylandWindow(window);
    if (!display || !surface) {
        throw std::runtime_error(
            "Wayland handles unavailable � run under a Wayland session (Vulkan backend requires Wayland on Linux)");
    }
    return goldy::SurfaceExchange(ctx, display, surface);
#endif
}

goldy::Transaction record_scheme(
    goldy::Scheme& scheme,
    const goldy::Buffer& cells,
    const char* read_field,
    const char* write_field,
    const goldy::ComputePipeline& compute_pipeline,
    const goldy::RenderPipeline& render_pipeline,
    const goldy::SchemeRenderTargetLease& scene_rt,
    goldy::SurfaceExchange& exchange) {
    goldy::Parcel read = cells.field(field_unit(read_field));
    goldy::Parcel write = cells.field(field_unit(write_field));
    {
        auto node = scheme.compute_node("game_of_life", compute_pipeline);
        node.with_parcel(read, goldy::NodeAccess::Read);
        node.with_parcel(write, goldy::NodeAccess::Overwrite);
        node.dispatch(WORKGROUPS_X, WORKGROUPS_Y, 1);
    }
    {
        auto pass = scheme.render_pass("game_of_life_render", scene_rt, goldy::TargetLoad::discard());
        pass.with_parcel(write, goldy::NodeAccess::Read)
            .set_pipeline(render_pipeline)
            .draw_fullscreen();
    }
    return exchange.bind_render_target(scheme, scene_rt);
}

struct BuiltSchemes {
    goldy::Scheme ab;
    goldy::Transaction present_ab;
    goldy::Scheme ba;
    goldy::Transaction present_ba;
};

std::pair<goldy::Scheme, goldy::Transaction> build_scheme(
    const goldy::Context& ctx,
    const goldy::Buffer& cells,
    const char* read_field,
    const char* write_field,
    const goldy::ComputePipeline& compute_pipeline,
    const goldy::RenderPipeline& render_pipeline,
    goldy::SurfaceExchange& exchange) {
    goldy::Scheme scheme(ctx);
    auto [width, height] = exchange.size();
    width = std::max(width, 1u);
    height = std::max(height, 1u);
    goldy::SchemeRenderTargetLease scene_rt =
        scheme.lease_render_target(width, height, exchange.format());
    goldy::Transaction present = record_scheme(
        scheme, cells, read_field, write_field, compute_pipeline, render_pipeline, scene_rt, exchange);
    return {std::move(scheme), std::move(present)};
}

BuiltSchemes build_schemes(
    const goldy::Context& ctx,
    const goldy::Buffer& cells,
    const goldy::ComputePipeline& compute_pipeline,
    const goldy::RenderPipeline& render_pipeline,
    goldy::SurfaceExchange& exchange) {
    auto ab = build_scheme(ctx, cells, "a", "b", compute_pipeline, render_pipeline, exchange);
    auto ba = build_scheme(ctx, cells, "b", "a", compute_pipeline, render_pipeline, exchange);
    return BuiltSchemes{
        std::move(ab.first),
        std::move(ab.second),
        std::move(ba.first),
        std::move(ba.second),
    };
}

void rerecord_orientations(GpuState& gpu) {
    BuiltSchemes schemes = build_schemes(
        gpu.ctx, gpu.cells, gpu.compute_pipeline, gpu.render_pipeline, gpu.exchange);
    gpu.scheme_ab = std::move(schemes.ab);
    gpu.present_ab = std::move(schemes.present_ab);
    gpu.scheme_ba = std::move(schemes.ba);
    gpu.present_ba = std::move(schemes.present_ba);
}

void step(GpuState& gpu) {
    goldy::Scheme& scheme = gpu.use_buffer_a ? gpu.scheme_ab : gpu.scheme_ba;
    goldy::Transaction& present = gpu.use_buffer_a ? gpu.present_ab : gpu.present_ba;
    auto submission = scheme.submit();
    present.claim(submission).consume();
    gpu.use_buffer_a = !gpu.use_buffer_a;
}

GpuState init_gpu(goldy::Device device, GLFWwindow* window) {
    const auto initial = create_initial_state();

    goldy::Context ctx(device);
    goldy::RetainedPool pool(device);
    auto record = pool.record();
    record.emplace_named("a", initial);
    record.emplace_named("b", initial);
    goldy::Buffer cells = record.build(pool);

    goldy::ShaderModule compute_shader(device, find_shader("game_of_life.slang"));
    goldy::ShaderModule render_shader(device, find_shader("game_of_life_render.slang"));
    goldy::ComputePipeline compute_pipeline(device, compute_shader);

    goldy::SurfaceExchange exchange = create_surface_exchange(ctx, window);

    GoldyRenderPipelineDesc render_desc{};
    render_desc.topology = GOLDY_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    render_desc.target_format = exchange.format();
    render_desc.depth_enabled = false;
    goldy::RenderPipeline render_pipeline(device, render_shader, render_shader, render_desc);

    BuiltSchemes schemes = build_schemes(ctx, cells, compute_pipeline, render_pipeline, exchange);

    return GpuState{
        std::move(ctx),
        std::move(device),
        std::move(pool),
        std::move(cells),
        std::move(compute_shader),
        std::move(render_shader),
        std::move(compute_pipeline),
        std::move(render_pipeline),
        std::move(exchange),
        std::move(schemes.ab),
        std::move(schemes.ba),
        std::move(schemes.present_ab),
        std::move(schemes.present_ba),
        true,
        0,
        std::chrono::steady_clock::now(),
        std::chrono::steady_clock::now(),
    };
}

void render_frame(GpuState& gpu) {
    const auto now = std::chrono::steady_clock::now();
    const bool should_step = gpu.frame_count == 0 ||
        std::chrono::duration_cast<std::chrono::milliseconds>(now - gpu.last_update).count() > 33;

    if (!should_step) {
        return;
    }

    gpu.last_update = now;
    step(gpu);
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
    if (w == gpu.exchange.width() && h == gpu.exchange.height()) {
        return;
    }
    gpu.exchange.resize(w, h);

    GoldyRenderPipelineDesc render_desc{};
    render_desc.topology = GOLDY_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    render_desc.target_format = gpu.exchange.format();
    render_desc.depth_enabled = false;
    gpu.render_pipeline =
        goldy::RenderPipeline(gpu.device, gpu.render_shader, gpu.render_shader, render_desc);

    rerecord_orientations(gpu);
    gpu.last_update = std::chrono::steady_clock::now() - std::chrono::milliseconds(34);
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
