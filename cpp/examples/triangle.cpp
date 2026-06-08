/**
 * Triangle example - render an animated colored triangle in an interactive window.
 *
 * Demonstrates the Surface API with task-graph submission via the stable C ABI:
 * offscreen RenderTarget -> render_pass -> copy_render_target_to_swapchain -> present.
 *
 * Window creation is platform-specific (Win32 / AppKit). Goldy surface creation uses
 * goldy_surface_create_win32 / goldy_surface_create_appkit only — no window toolkit
 * inside goldy.hpp.
 *
 * Build: cmake --build build --target triangle
 */

#include <goldy.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#elif defined(__APPLE__)
#import <Cocoa/Cocoa.h>
#endif

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

GpuState init_gpu(goldy::Device device, goldy::Surface surface) {
    const Vertex vertices[] = {
        {{0.0f, -0.5f}, {1.0f, 0.0f, 0.0f, 1.0f}},
        {{-0.5f, 0.5f}, {0.0f, 1.0f, 0.0f, 1.0f}},
        {{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f, 1.0f}},
    };

    goldy::Buffer vertex_buffer(
        device,
        std::span<const Vertex>(vertices),
        goldy::BufferKind::Scattered);

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

void print_perf(const GpuState& gpu) {
    const auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - gpu.start_time);
    const double fps = elapsed.count() > 0.0
        ? static_cast<double>(gpu.frame_count) / elapsed.count()
        : 0.0;
    std::cout << "GOLDY_PERF: frames=" << gpu.frame_count
              << " elapsed=" << elapsed.count() << "s avg_fps=" << fps << '\n';
}

#if defined(_WIN32)

struct WindowState {
    HWND hwnd = nullptr;
    std::optional<GpuState> gpu;
    bool running = true;
};

WindowState* g_window = nullptr;

void handle_resize(GpuState& gpu, uint32_t width, uint32_t height) {
    if (width == 0 || height == 0) {
        return;
    }
    gpu.surface.resize(width, height);
    gpu.scene_rt = make_scene_rt(gpu.device, gpu.surface);
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam) {
    switch (msg) {
    case WM_CLOSE:
        if (g_window) {
            g_window->gpu.reset();
            g_window->running = false;
        }
        DestroyWindow(hwnd);
        return 0;
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    case WM_KEYDOWN:
        if (wparam == VK_ESCAPE && g_window) {
            g_window->gpu.reset();
            g_window->running = false;
            DestroyWindow(hwnd);
        }
        return 0;
    case WM_SIZE:
        if (g_window && g_window->gpu && wparam != SIZE_MINIMIZED) {
            handle_resize(*g_window->gpu, LOWORD(lparam), HIWORD(lparam));
        }
        return 0;
    default:
        return DefWindowProc(hwnd, msg, wparam, lparam);
    }
}

int run_windowed_app(goldy::Device device) {
    HINSTANCE instance = GetModuleHandleW(nullptr);
    const wchar_t* class_name = L"GoldyTriangleWindow";

    WNDCLASSEXW wc{};
    wc.cbSize = sizeof(wc);
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = WndProc;
    wc.hInstance = instance;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.lpszClassName = class_name;
    RegisterClassExW(&wc);

    HWND hwnd = CreateWindowExW(
        0,
        class_name,
        L"Goldy - Animated Triangle (Surface API)",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        800,
        600,
        nullptr,
        nullptr,
        instance,
        nullptr);
    if (!hwnd) {
        throw std::runtime_error("CreateWindowEx failed");
    }

    WindowState window_state{.hwnd = hwnd};
    g_window = &window_state;

    ShowWindow(hwnd, SW_SHOW);
    UpdateWindow(hwnd);

    goldy::Surface surface(device, hwnd);
    window_state.gpu = init_gpu(std::move(device), std::move(surface));

    MSG msg{};
    while (window_state.running) {
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) {
                window_state.running = false;
                break;
            }
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
        if (!window_state.running || !window_state.gpu) {
            break;
        }
        render_frame(*window_state.gpu);
    }

    if (window_state.gpu) {
        print_perf(*window_state.gpu);
    }
    g_window = nullptr;
    return 0;
}

#elif defined(__APPLE__)

struct MacWindow {
    NSWindow* window = nil;
    std::optional<GpuState> gpu;
    bool running = true;
};

void handle_resize(GpuState& gpu, uint32_t width, uint32_t height) {
    if (width == 0 || height == 0) {
        return;
    }
    gpu.surface.resize(width, height);
    gpu.scene_rt = make_scene_rt(gpu.device, gpu.surface);
}

int run_windowed_app(goldy::Device device) {
    @autoreleasepool {
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];

        NSRect frame = NSMakeRect(0, 0, 800, 600);
        NSWindow* window = [[NSWindow alloc]
            initWithContentRect:frame
                      styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                                NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable
                        backing:NSBackingStoreBuffered
                          defer:NO];
        [window setTitle:@"Goldy - Animated Triangle (Surface API)"];
        [window center];
        [window makeKeyAndOrderFront:nil];

        MacWindow state{.window = window};
        void* ns_view = static_cast<void*>([window contentView]);
        goldy::Surface surface(device, ns_view);
        state.gpu = init_gpu(std::move(device), std::move(surface));

        while (state.running) {
            @autoreleasepool {
                NSEvent* event = [NSApp nextEventMatchingMask:NSEventMaskAny
                                                     untilDate:[NSDate dateWithTimeIntervalSinceNow:0]
                                                        inMode:NSDefaultRunLoopMode
                                                       dequeue:YES];
                if (event) {
                    if ([event type] == NSEventTypeKeyDown) {
                        NSString* chars = [event charactersIgnoringModifiers];
                        if ([chars isEqualToString:@"\e"]) {
                            state.gpu.reset();
                            state.running = false;
                            [window close];
                            break;
                        }
                    }
                    [NSApp sendEvent:event];
                }

                if (![window isVisible]) {
                    state.gpu.reset();
                    state.running = false;
                    break;
                }

                if (!state.running || !state.gpu) {
                    break;
                }

                NSRect bounds = [[window contentView] bounds];
                const uint32_t width = static_cast<uint32_t>(std::max<CGFloat>(bounds.size.width, 1.0));
                const uint32_t height = static_cast<uint32_t>(std::max<CGFloat>(bounds.size.height, 1.0));
                if (width != state.gpu->surface.width() || height != state.gpu->surface.height()) {
                    handle_resize(*state.gpu, width, height);
                }

                render_frame(*state.gpu);
            }
        }

        if (state.gpu) {
            print_perf(*state.gpu);
        }
    }
    return 0;
}

#else

int run_windowed_app(goldy::Device) {
    std::cerr << "triangle example requires Win32 or macOS for window/surface creation\n";
    return 1;
}

#endif

} // namespace

int main() {
    try {
        std::cout << "Goldy Surface API Example (C++)\n";
        std::cout << "================================\n";
        std::cout << "Rendering triangle via TaskGraph (offscreen RT -> swapchain blit)\n";
        std::cout << "Press Escape or close window to exit\n\n";

        goldy::Instance instance;
        goldy::Device device = instance.request_adapter().request_device();

        return run_windowed_app(std::move(device));
    } catch (const goldy::Exception& e) {
        std::cerr << "Goldy error: " << e.what() << '\n';
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
