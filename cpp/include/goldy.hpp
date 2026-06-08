/*
 * Goldy GPU Library - C++ RAII Wrapper
 *
 * LGPL-2.1-or-later - Copyright (c) 2025-2026 Mohamed Koubaa
 *
 * This header provides modern C++ RAII wrappers around the Goldy C API.
 * All resources are automatically cleaned up when they go out of scope.
 *
 * Usage:
 *   #include <goldy.hpp>
 *
 *   goldy::Instance instance;
 *   auto adapters = instance.enumerate_adapters();
 *   goldy::Device device = instance.create_device_for_adapter(adapters[0].id);
 *   goldy::RenderTarget target(device, 800, 600);
 *   // ...
 */

#ifndef GOLDY_HPP
#define GOLDY_HPP

#include "goldy.h"

#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <span>
#include <optional>

namespace goldy {

// =============================================================================
// Exception type
// =============================================================================

/**
 * @brief Exception thrown when a Goldy operation fails.
 */
class Exception : public std::runtime_error {
public:
    explicit Exception(const std::string& message)
        : std::runtime_error(message) {}

    /**
     * @brief Create an exception from the last error message.
     */
    static Exception from_last_error() {
        const char* error = goldy_get_last_error();
        if (error) {
            return Exception(error);
        }
        return Exception("Unknown error");
    }
};

// =============================================================================
// Forward declarations
// =============================================================================

class Instance;
class Device;
class Adapter;
class Buffer;
class ShaderModule;
class RenderPipeline;
class RenderTarget;
class Surface;
class TaskGraph;
class BufferPool;
class BufferView;
class ComputePipeline;
class ComputeEncoder;
class Texture;
class Sampler;

// =============================================================================
// Helper types
// =============================================================================

/**
 * @brief RGBA color with floating point components (0.0 - 1.0).
 */
struct Color {
    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
    float a = 1.0f;

    Color() = default;
    Color(float r, float g, float b, float a = 1.0f) : r(r), g(g), b(b), a(a) {}

    operator GoldyColor() const { return GoldyColor{r, g, b, a}; }

    // Common colors
    static Color black() { return {0.0f, 0.0f, 0.0f, 1.0f}; }
    static Color white() { return {1.0f, 1.0f, 1.0f, 1.0f}; }
    static Color red() { return {1.0f, 0.0f, 0.0f, 1.0f}; }
    static Color green() { return {0.0f, 1.0f, 0.0f, 1.0f}; }
    static Color blue() { return {0.0f, 0.0f, 1.0f, 1.0f}; }
    static Color cornflower_blue() { return {0.392f, 0.584f, 0.929f, 1.0f}; }
};

/**
 * @brief GPU device type (mirrors native Goldy adapter metadata).
 */
enum class DeviceType {
    DiscreteGpu = GOLDY_DEVICE_TYPE_DISCRETE_GPU,
    IntegratedGpu = GOLDY_DEVICE_TYPE_INTEGRATED_GPU,
    Cpu = GOLDY_DEVICE_TYPE_CPU,
    Other = GOLDY_DEVICE_TYPE_OTHER,
};

/**
 * @brief Power preference for adapter selection.
 */
enum class PowerPreference {
    None,
    LowPower,
    HighPerformance,
};

/**
 * @brief Options for Instance::request_adapter().
 */
struct RequestAdapterOptions {
    PowerPreference power_preference = PowerPreference::HighPerformance;
    bool force_fallback_adapter = false;
};

/**
 * @brief Descriptor for Adapter::request_device().
 */
struct DeviceDescriptor {
    std::optional<std::string> label;
};

/**
 * @brief Adapter information returned by Instance::enumerate_adapters().
 */
struct AdapterInfo {
    uint32_t id;
    DeviceType device_type;
    std::string name;
    std::string vendor;
};

// =============================================================================
// Access pattern enums (C++ friendly)
// =============================================================================

/**
 * @brief Data access pattern for buffers.
 */
enum class BufferKind {
    /// Any thread, any address, read/write (StructuredBuffer, RWStructuredBuffer).
    Scattered = 0,
    /// All threads same address, broadcast optimized (ConstantBuffer).
    Broadcast = 1,
};

/**
 * @brief Spatial access pattern for textures.
 */
enum class TextureKind {
    /// Hardware filtering between neighbors (Texture2D with sampler).
    Interpolated = 0,
    /// Direct 2D/3D indexing, no filtering (RWTexture2D).
    Direct = 1,
    /// Both UAV (storage/write) and SRV (sampled/read) access on the same texture.
    DirectInterpolated = 2,
};

/**
 * @brief Logical access declared on a task-graph node.
 */
enum class NodeAccess {
    Read = GOLDY_NODE_ACCESS_READ,
    Write = GOLDY_NODE_ACCESS_WRITE,
    ReadWrite = GOLDY_NODE_ACCESS_READ_WRITE,
};

enum class ResourceAccess {
    Read = GOLDY_RESOURCE_ACCESS_READ,
    Write = GOLDY_RESOURCE_ACCESS_WRITE,
    ReadWrite = GOLDY_RESOURCE_ACCESS_READ_WRITE,
};

/**
 * @brief Texture flags for copy and render operations.
 */
namespace TextureFlags {
    inline constexpr uint32_t None        = 0;
    inline constexpr uint32_t CopySrc     = 1 << 0;
    inline constexpr uint32_t CopyDst     = 1 << 1;
    inline constexpr uint32_t RenderTarget = 1 << 2;
}

} // namespace goldy

namespace goldy {

// =============================================================================
// Smart pointer deleters
// =============================================================================

namespace detail {

struct InstanceDeleter {
    void operator()(GoldyInstance* p) const { if (p) goldy_instance_destroy(p); }
};

struct DeviceDeleter {
    void operator()(GoldyDevice* p) const { if (p) goldy_device_destroy(p); }
};

struct BufferDeleter {
    void operator()(GoldyBuffer* p) const { if (p) goldy_buffer_destroy(p); }
};

struct BufferPoolDeleter {
    void operator()(GoldyBufferPool* p) const { if (p) goldy_buffer_pool_destroy(p); }
};

struct BufferViewDeleter {
    void operator()(GoldyBufferView* p) const { if (p) goldy_buffer_view_destroy(p); }
};

struct ShaderDeleter {
    void operator()(GoldyShaderModule* p) const { if (p) goldy_shader_destroy(p); }
};

struct RenderPipelineDeleter {
    void operator()(GoldyRenderPipeline* p) const { if (p) goldy_render_pipeline_destroy(p); }
};

struct RenderTargetDeleter {
    void operator()(GoldyRenderTarget* p) const { if (p) goldy_render_target_destroy(p); }
};

struct SurfaceDeleter {
    void operator()(GoldySurface* p) const { if (p) goldy_surface_destroy(p); }
};

struct TaskGraphDeleter {
    void operator()(GoldyTaskGraph* p) const { if (p) goldy_task_graph_destroy(p); }
};

struct ComputePipelineDeleter {
    void operator()(GoldyComputePipeline* p) const { if (p) goldy_compute_pipeline_destroy(p); }
};

struct ComputeEncoderDeleter {
    void operator()(GoldyComputeEncoder* p) const { if (p) goldy_compute_encoder_destroy(p); }
};

struct TextureDeleter {
    void operator()(GoldyTexture* p) const { if (p) goldy_texture_destroy(p); }
};

struct SamplerDeleter {
    void operator()(GoldySampler* p) const { if (p) goldy_sampler_destroy(p); }
};

inline void throw_on_result(GoldyResult result) {
    if (result != GOLDY_RESULT_OK) {
        throw Exception::from_last_error();
    }
}

} // namespace detail

// =============================================================================
// Instance
// =============================================================================

/**
 * @brief Entry point for the Goldy GPU library.
 *
 * An Instance provides access to GPU adapters and device creation.
 */
class Instance {
public:
    /**
     * @brief Create a new Goldy instance.
     * @throws Exception if creation fails.
     */
    Instance() {
        GoldyInstance* ptr = goldy_instance_create();
        if (!ptr) {
            throw Exception::from_last_error();
        }
        ptr_.reset(ptr);
    }

    Instance(const Instance&) = delete;
    Instance& operator=(const Instance&) = delete;
    Instance(Instance&&) = default;
    Instance& operator=(Instance&&) = default;

    /**
     * @brief Get the graphics backend type.
     */
    GoldyBackendType backend_type() const {
        return goldy_instance_backend_type(ptr_.get());
    }

    /**
     * @brief Enumerate available GPU adapters.
     */
    std::vector<AdapterInfo> enumerate_adapters() const {
        uint32_t count = goldy_instance_adapter_count(ptr_.get());
        std::vector<AdapterInfo> adapters;
        adapters.reserve(count);

        for (uint32_t i = 0; i < count; ++i) {
            GoldyAdapterInfo info{};
            if (goldy_instance_get_adapter(ptr_.get(), i, &info) == GOLDY_RESULT_OK) {
                adapters.push_back({
                    info.id,
                    static_cast<DeviceType>(info.device_type),
                    std::string(info.name),
                    std::string(info.vendor)
                });
            }
        }
        return adapters;
    }

    /**
     * @brief Request an adapter matching the given options (wgpu-style).
     * @throws Exception if no adapters are available.
     */
    Adapter request_adapter(const RequestAdapterOptions& opts = {});

    /**
     * @brief Create a device for a specific adapter.
     * @param adapter_id The adapter ID from enumerate_adapters().
     * @throws Exception if device creation fails.
     */
    Device create_device_for_adapter(uint32_t adapter_id);

    /**
     * @brief Get raw pointer (for advanced use).
     */
    GoldyInstance* get() const { return ptr_.get(); }

private:
    std::unique_ptr<GoldyInstance, detail::InstanceDeleter> ptr_;
};

// =============================================================================
// Device
// =============================================================================

/**
 * @brief A GPU device handle.
 *
 * Represents a connection to a GPU. Used to create resources.
 */
class Device {
public:
    Device() = default;

    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    Device(Device&&) = default;
    Device& operator=(Device&&) = default;

    /**
     * @brief Check if the device is valid.
     */
    bool is_valid() const {
        return ptr_ && goldy_device_is_valid(ptr_.get());
    }

    /**
     * @brief Get the adapter ID this device was created on.
     */
    uint32_t adapter_id() const {
        return goldy_device_adapter_id(ptr_.get());
    }

    /**
     * @brief Check if a shader library is registered.
     */
    bool has_library(std::string_view name) const {
        std::string name_str(name);
        return goldy_device_has_library(ptr_.get(), name_str.c_str());
    }

    /**
     * @brief Get raw pointer (for advanced use).
     */
    GoldyDevice* get() const { return ptr_.get(); }

private:
    friend class Instance;
    friend class Adapter;
    explicit Device(GoldyDevice* ptr) : ptr_(ptr) {}
    std::unique_ptr<GoldyDevice, detail::DeviceDeleter> ptr_;
};

// Instance methods that return Device
inline Device Instance::create_device_for_adapter(uint32_t adapter_id) {
    GoldyDevice* ptr = goldy_instance_create_device_for_adapter(ptr_.get(), adapter_id);
    if (!ptr) {
        throw Exception::from_last_error();
    }
    return Device(ptr);
}

// =============================================================================
// Adapter
// =============================================================================

/**
 * @brief A physical GPU adapter returned by Instance::request_adapter().
 */
class Adapter {
public:
    Adapter(const Adapter&) = delete;
    Adapter& operator=(const Adapter&) = delete;
    Adapter(Adapter&&) = default;
    Adapter& operator=(Adapter&&) = default;

    const AdapterInfo& get_info() const { return info_; }

    /**
     * @brief Create a logical Device on this adapter.
     */
    Device request_device(const DeviceDescriptor& desc = {}) const {
        (void)desc;
        GoldyDevice* ptr = goldy_instance_create_device_for_adapter(instance_, info_.id);
        if (!ptr) {
            throw Exception::from_last_error();
        }
        return Device(ptr);
    }

private:
    friend class Instance;
    Adapter(GoldyInstance* instance, AdapterInfo info)
        : instance_(instance), info_(std::move(info)) {}

    GoldyInstance* instance_;
    AdapterInfo info_;
};

inline Adapter Instance::request_adapter(const RequestAdapterOptions& opts) {
    auto adapters = enumerate_adapters();
    if (adapters.empty()) {
        throw Exception("No GPU adapters available");
    }

    const AdapterInfo* selected = nullptr;
    switch (opts.power_preference) {
    case PowerPreference::HighPerformance:
        for (const auto& a : adapters) {
            if (a.device_type == DeviceType::DiscreteGpu) {
                selected = &a;
                break;
            }
        }
        if (!selected) {
            for (const auto& a : adapters) {
                if (a.device_type == DeviceType::IntegratedGpu) {
                    selected = &a;
                    break;
                }
            }
        }
        if (!selected) {
            for (const auto& a : adapters) {
                if (a.device_type == DeviceType::Other) {
                    selected = &a;
                    break;
                }
            }
        }
        if (!selected) {
            selected = &adapters.front();
        }
        break;
    case PowerPreference::LowPower:
        for (const auto& a : adapters) {
            if (a.device_type == DeviceType::IntegratedGpu) {
                selected = &a;
                break;
            }
        }
        if (!selected) {
            for (const auto& a : adapters) {
                if (a.device_type == DeviceType::Cpu) {
                    selected = &a;
                    break;
                }
            }
        }
        if (!selected) {
            selected = &adapters.front();
        }
        break;
    case PowerPreference::None:
        selected = &adapters.front();
        break;
    }

    (void)opts.force_fallback_adapter;
    return Adapter(ptr_.get(), *selected);
}

// =============================================================================
// Buffer
// =============================================================================

/**
 * @brief A GPU buffer for data storage.
 *
 * Buffers hold data on the GPU with a specific access pattern:
 * - BufferKind::Scattered: Any thread can access any address (StructuredBuffer, RWStructuredBuffer)
 * - BufferKind::Broadcast: All threads read same address, broadcast optimized (ConstantBuffer)
 */
class Buffer {
public:
    /**
     * @brief Create an uninitialized buffer.
     * @param device The device to create the buffer on.
     * @param size Buffer size in bytes.
     * @param access Data access pattern.
     * @throws Exception if creation fails.
     */
    Buffer(const Device& device, uint64_t size, BufferKind access) {
        GoldyBuffer* ptr = goldy_buffer_create(device.get(), size, static_cast<GoldyBufferKind>(access));
        if (!ptr) {
            throw Exception::from_last_error();
        }
        ptr_.reset(ptr);
    }

    /**
     * @brief Create a buffer initialized with data.
     * @param device The device to create the buffer on.
     * @param data Initial data.
     * @param access Data access pattern.
     * @throws Exception if creation fails.
     */
    Buffer(const Device& device, std::span<const uint8_t> data, BufferKind access) {
        GoldyBuffer* ptr = goldy_buffer_create_with_data(
            device.get(), data.data(), data.size(), static_cast<GoldyBufferKind>(access));
        if (!ptr) {
            throw Exception::from_last_error();
        }
        ptr_.reset(ptr);
    }

    /**
     * @brief Create a buffer from typed data with the correct element stride.
     */
    template<typename T>
    Buffer(const Device& device, std::span<const T> data, BufferKind access) {
        GoldyBuffer* ptr = goldy_buffer_create_with_data_stride(
            device.get(),
            reinterpret_cast<const uint8_t*>(data.data()),
            data.size() * sizeof(T),
            static_cast<GoldyBufferKind>(access),
            static_cast<uint32_t>(sizeof(T)));
        if (!ptr) {
            throw Exception::from_last_error();
        }
        ptr_.reset(ptr);
    }

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&&) = default;
    Buffer& operator=(Buffer&&) = default;

    /**
     * @brief Write data to the buffer.
     * @param offset Byte offset to write at.
     * @param data Data to write.
     * @throws Exception if write fails.
     */
    void write(uint64_t offset, std::span<const uint8_t> data) {
        GoldyResult result = goldy_buffer_write(ptr_.get(), offset, data.data(), data.size());
        if (result != GOLDY_RESULT_OK) {
            throw Exception::from_last_error();
        }
    }

    /**
     * @brief Get buffer size in bytes.
     */
    uint64_t size() const { return goldy_buffer_size(ptr_.get()); }

    /**
     * @brief Get buffer's access pattern.
     */
    BufferKind access() const { return static_cast<BufferKind>(goldy_buffer_access(ptr_.get())); }

    /**
     * @brief Get raw pointer (for advanced use).
     */
    GoldyBuffer* get() const { return ptr_.get(); }

    uint32_t resource_index(ResourceAccess access) const {
        uint32_t idx = goldy_buffer_resource_index(ptr_.get(), static_cast<GoldyResourceAccess>(access));
        if (idx == UINT32_MAX) {
            throw Exception::from_last_error();
        }
        return idx;
    }

private:
    std::unique_ptr<GoldyBuffer, detail::BufferDeleter> ptr_;
};

// =============================================================================
// BufferPool / BufferView
// =============================================================================

class BufferPool {
public:
    BufferPool(const Device& device, uint64_t capacity) {
        GoldyBufferPool* ptr = goldy_buffer_pool_create(device.get(), capacity);
        if (!ptr) {
            throw Exception::from_last_error();
        }
        ptr_.reset(ptr);
    }

    BufferPool(const BufferPool&) = delete;
    BufferPool& operator=(const BufferPool&) = delete;
    BufferPool(BufferPool&&) = default;
    BufferPool& operator=(BufferPool&&) = default;

    BufferView alloc_u32(uint64_t count);

    void write_backing(uint64_t byte_offset, std::span<const uint8_t> data) {
        GoldyResult result = goldy_buffer_pool_write_backing(
            ptr_.get(), byte_offset, data.data(), data.size());
        if (result != GOLDY_RESULT_OK) {
            throw Exception::from_last_error();
        }
    }

    GoldyBufferPool* get() const { return ptr_.get(); }

private:
    std::unique_ptr<GoldyBufferPool, detail::BufferPoolDeleter> ptr_;
};

class BufferView {
public:
    BufferView() = default;

    explicit BufferView(GoldyBufferView* ptr) { ptr_.reset(ptr); }

    BufferView(const BufferView&) = delete;
    BufferView& operator=(const BufferView&) = delete;
    BufferView(BufferView&&) = default;
    BufferView& operator=(BufferView&&) = default;

    uint64_t offset() const { return goldy_buffer_view_offset(ptr_.get()); }

    uint32_t resource_index(ResourceAccess access) const {
        uint32_t idx = goldy_buffer_view_resource_index(ptr_.get(), static_cast<GoldyResourceAccess>(access));
        if (idx == UINT32_MAX) {
            throw Exception::from_last_error();
        }
        return idx;
    }

    GoldyBufferView* get() const { return ptr_.get(); }

private:
    std::unique_ptr<GoldyBufferView, detail::BufferViewDeleter> ptr_;
};

inline BufferView BufferPool::alloc_u32(uint64_t count) {
    GoldyBufferView* view = goldy_buffer_pool_alloc_u32(ptr_.get(), count);
    if (!view) {
        throw Exception::from_last_error();
    }
    return BufferView(view);
}

// =============================================================================
// ShaderModule
// =============================================================================

/**
 * @brief A compiled shader module.
 */
class ShaderModule {
public:
    /**
     * @brief Create a shader from Slang source code.
     * @param device The device to compile on.
     * @param source Slang shader source code.
     * @throws Exception if compilation fails.
     */
    ShaderModule(const Device& device, std::string_view source) {
        std::string source_str(source);
        GoldyShaderModule* ptr = goldy_shader_create(device.get(), source_str.c_str());
        if (!ptr) {
            throw Exception::from_last_error();
        }
        ptr_.reset(ptr);
    }

    ShaderModule(const ShaderModule&) = delete;
    ShaderModule& operator=(const ShaderModule&) = delete;
    ShaderModule(ShaderModule&&) = default;
    ShaderModule& operator=(ShaderModule&&) = default;

    /**
     * @brief Get built-in vertex color 2D shader source.
     */
    static const char* builtin_vertex_color_2d() {
        return goldy_shader_builtin_vertex_color_2d();
    }

    /**
     * @brief Get raw pointer (for advanced use).
     */
    GoldyShaderModule* get() const { return ptr_.get(); }

private:
    std::unique_ptr<GoldyShaderModule, detail::ShaderDeleter> ptr_;
};

// =============================================================================
// RenderPipeline
// =============================================================================

/**
 * @brief A graphics rendering pipeline.
 */
class RenderPipeline {
public:
    /**
     * @brief Create a render pipeline.
     * @param device The device.
     * @param vertex_shader The vertex shader.
     * @param fragment_shader The fragment shader.
     * @param desc Pipeline descriptor.
     * @throws Exception if creation fails.
     */
    RenderPipeline(const Device& device,
                   const ShaderModule& vertex_shader,
                   const ShaderModule& fragment_shader,
                   const GoldyRenderPipelineDesc& desc) {
        GoldyRenderPipeline* ptr = goldy_render_pipeline_create(
            device.get(), vertex_shader.get(), fragment_shader.get(), &desc);
        if (!ptr) {
            throw Exception::from_last_error();
        }
        ptr_.reset(ptr);
    }

    RenderPipeline(const RenderPipeline&) = delete;
    RenderPipeline& operator=(const RenderPipeline&) = delete;
    RenderPipeline(RenderPipeline&&) = default;
    RenderPipeline& operator=(RenderPipeline&&) = default;

    /**
     * @brief Get raw pointer (for advanced use).
     */
    GoldyRenderPipeline* get() const { return ptr_.get(); }

private:
    std::unique_ptr<GoldyRenderPipeline, detail::RenderPipelineDeleter> ptr_;
};

// =============================================================================
// RenderTarget
// =============================================================================

/**
 * @brief A GPU render target (offscreen framebuffer).
 */
class RenderTarget {
public:
    /**
     * @brief Create a render target without depth buffer.
     * @param device The device.
     * @param width Width in pixels.
     * @param height Height in pixels.
     * @param format Texture format.
     * @throws Exception if creation fails.
     */
    RenderTarget(const Device& device, uint32_t width, uint32_t height,
                 GoldyTextureFormat format = GOLDY_TEXTURE_FORMAT_RGBA8_UNORM) {
        GoldyRenderTarget* ptr = goldy_render_target_create(device.get(), width, height, format);
        if (!ptr) {
            throw Exception::from_last_error();
        }
        ptr_.reset(ptr);
    }

    /**
     * @brief Create a render target with depth buffer.
     */
    RenderTarget(const Device& device, uint32_t width, uint32_t height,
                 GoldyTextureFormat color_format, GoldyDepthFormat depth_format) {
        GoldyRenderTarget* ptr = goldy_render_target_create_with_depth(
            device.get(), width, height, color_format, depth_format);
        if (!ptr) {
            throw Exception::from_last_error();
        }
        ptr_.reset(ptr);
    }

    RenderTarget(const RenderTarget&) = delete;
    RenderTarget& operator=(const RenderTarget&) = delete;
    RenderTarget(RenderTarget&&) = default;
    RenderTarget& operator=(RenderTarget&&) = default;

    /**
     * @brief Get width in pixels.
     */
    uint32_t width() const { return goldy_render_target_width(ptr_.get()); }

    /**
     * @brief Get height in pixels.
     */
    uint32_t height() const { return goldy_render_target_height(ptr_.get()); }

    /**
     * @brief Get texture format.
     */
    GoldyTextureFormat format() const { return goldy_render_target_format(ptr_.get()); }

    /**
     * @brief Check if has depth buffer.
     */
    bool has_depth() const { return goldy_render_target_has_depth(ptr_.get()); }

    /**
     * @brief Get buffer size in bytes for CPU readback.
     */
    size_t buffer_size() const { return goldy_render_target_buffer_size(ptr_.get()); }

    /**
     * @brief Read rendered pixels to CPU memory.
     * @return Vector containing pixel data.
     * @throws Exception if readback fails.
     */
    std::vector<uint8_t> read_to_cpu() {
        std::vector<uint8_t> data(buffer_size());
        GoldyResult result = goldy_render_target_read_to_buffer(ptr_.get(), data.data(), data.size());
        if (result != GOLDY_RESULT_OK) {
            throw Exception::from_last_error();
        }
        return data;
    }

    /**
     * @brief Read rendered pixels to an existing buffer.
     * @param output Buffer to write to (must be at least buffer_size() bytes).
     * @throws Exception if readback fails.
     */
    void read_to_buffer(std::span<uint8_t> output) {
        GoldyResult result = goldy_render_target_read_to_buffer(ptr_.get(), output.data(), output.size());
        if (result != GOLDY_RESULT_OK) {
            throw Exception::from_last_error();
        }
    }

    /**
     * @brief Get raw pointer (for advanced use).
     */
    GoldyRenderTarget* get() const { return ptr_.get(); }

private:
    std::unique_ptr<GoldyRenderTarget, detail::RenderTargetDeleter> ptr_;
};

// =============================================================================
// Surface
// =============================================================================

/**
 * @brief An acquired swapchain frame.
 *
 * Consumed by Surface::present() or updated by Surface::submit_graph_to_frame().
 */
class SurfaceFrame {
public:
    SurfaceFrame() = default;
    SurfaceFrame(SurfaceFrame&& other) noexcept : ptr_(other.ptr_) { other.ptr_ = nullptr; }
    SurfaceFrame& operator=(SurfaceFrame&& other) noexcept {
        if (this != &other) {
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }
    SurfaceFrame(const SurfaceFrame&) = delete;
    SurfaceFrame& operator=(const SurfaceFrame&) = delete;

    GoldySurfaceFrame* get() const { return ptr_; }

private:
    friend class Surface;
    explicit SurfaceFrame(GoldySurfaceFrame* ptr) : ptr_(ptr) {}
    GoldySurfaceFrame* release() {
        GoldySurfaceFrame* p = ptr_;
        ptr_ = nullptr;
        return p;
    }

    GoldySurfaceFrame* ptr_ = nullptr;
};

/**
 * @brief A window swapchain surface.
 *
 * Created from platform window handles via goldy_surface_create_win32 /
 * goldy_surface_create_appkit / goldy_surface_create_wayland. Window toolkit
 * code stays in the application; this wrapper only wraps the stable C ABI.
 */
class Surface {
public:
    static constexpr bool is_supported() noexcept { return true; }

#if defined(_WIN32)
    Surface(const Device& device, void* hwnd) {
        GoldySurface* ptr = goldy_surface_create_win32(device.get(), hwnd);
        if (!ptr) {
            throw Exception::from_last_error();
        }
        ptr_.reset(ptr);
    }
#elif defined(__APPLE__)
    Surface(const Device& device, void* ns_view) {
        GoldySurface* ptr = goldy_surface_create_appkit(device.get(), ns_view);
        if (!ptr) {
            throw Exception::from_last_error();
        }
        ptr_.reset(ptr);
    }
#else
    Surface(const Device& device, void* wayland_display, void* wayland_surface) {
        GoldySurface* ptr = goldy_surface_create_wayland(
            device.get(), wayland_display, wayland_surface);
        if (!ptr) {
            throw Exception::from_last_error();
        }
        ptr_.reset(ptr);
    }
#endif

    Surface(const Surface&) = delete;
    Surface& operator=(const Surface&) = delete;
    Surface(Surface&&) = default;
    Surface& operator=(Surface&&) = default;

    std::pair<uint32_t, uint32_t> size() const {
        return {goldy_surface_width(ptr_.get()), goldy_surface_height(ptr_.get())};
    }

    uint32_t width() const { return goldy_surface_width(ptr_.get()); }
    uint32_t height() const { return goldy_surface_height(ptr_.get()); }

    GoldyTextureFormat format() const { return goldy_surface_format(ptr_.get()); }

    void resize(uint32_t width, uint32_t height) {
        detail::throw_on_result(goldy_surface_resize(ptr_.get(), width, height));
    }

    /**
     * @brief Begin the next frame (acquire swapchain image).
     */
    SurfaceFrame begin() {
        GoldySurfaceFrame* frame = goldy_surface_acquire(ptr_.get());
        if (!frame) {
            throw Exception::from_last_error();
        }
        return SurfaceFrame(frame);
    }

    /**
     * @brief Submit a recorded task graph to an acquired frame.
     */
    SurfaceFrame submit_graph_to_frame(TaskGraph& graph, SurfaceFrame frame);

    /**
     * @brief Present a frame to the screen (consumes the frame).
     */
    void present(SurfaceFrame frame);

    GoldySurface* get() const { return ptr_.get(); }

private:
    std::unique_ptr<GoldySurface, detail::SurfaceDeleter> ptr_;
};

// =============================================================================
// TaskGraph
// =============================================================================

/**
 * @brief Non-owning token from TaskGraph::declare_swapchain_output().
 *
 * Points at storage inside the parent graph; do not free. Pass to
 * copy_render_target_to_swapchain() only.
 */
class SwapchainOutput {
public:
    explicit SwapchainOutput(GoldySwapchainOutput* ptr) : ptr_(ptr) {}

    SwapchainOutput(const SwapchainOutput&) = default;
    SwapchainOutput& operator=(const SwapchainOutput&) = default;
    SwapchainOutput(SwapchainOutput&&) = default;
    SwapchainOutput& operator=(SwapchainOutput&&) = default;

    GoldySwapchainOutput* get() const { return ptr_; }

private:
    GoldySwapchainOutput* ptr_ = nullptr;
};

/**
 * @brief GPU task graph for render passes, compute nodes, and swapchain blits.
 */
class TaskGraph {
public:
    class RenderPass;

    TaskGraph() {
        GoldyTaskGraph* ptr = goldy_task_graph_create();
        if (!ptr) {
            throw Exception::from_last_error();
        }
        ptr_.reset(ptr);
    }

    TaskGraph(const TaskGraph&) = delete;
    TaskGraph& operator=(const TaskGraph&) = delete;
    TaskGraph(TaskGraph&&) = default;
    TaskGraph& operator=(TaskGraph&&) = default;

    void clear() {
        detail::throw_on_result(goldy_task_graph_clear(ptr_.get()));
    }

    void dispatch(const Device& device) {
        detail::throw_on_result(goldy_task_graph_dispatch(ptr_.get(), device.get()));
    }

    [[nodiscard]] SwapchainOutput declare_swapchain_output() {
        GoldySwapchainOutput* token = goldy_task_graph_declare_swapchain_output(ptr_.get());
        if (!token) {
            throw Exception::from_last_error();
        }
        return SwapchainOutput(token);
    }

    void copy_render_target_to_swapchain(const RenderTarget& src, const SwapchainOutput& swapchain) {
        detail::throw_on_result(goldy_task_graph_copy_render_target_to_swapchain(
            ptr_.get(), src.get(), swapchain.get()));
    }

    [[nodiscard]] RenderPass render_pass(const char* label, const RenderTarget& target);

    class ComputeNode;
    [[nodiscard]] ComputeNode compute_node(const char* label, const ComputePipeline& pipeline);

    void write_buffer(const Buffer& buffer, uint64_t offset, std::span<const uint8_t> data);

    GoldyTaskGraph* get() const { return ptr_.get(); }

private:
    friend class RenderPass;
    std::unique_ptr<GoldyTaskGraph, detail::TaskGraphDeleter> ptr_;
};

/**
 * @brief RAII scope for recording one offscreen render pass on a task graph.
 *
 * Calls render_pass_finish on destruction if finish() was not called explicitly.
 */
class TaskGraph::RenderPass {
public:
    RenderPass(TaskGraph& graph, const char* label, const RenderTarget& target)
        : graph_(graph) {
        detail::throw_on_result(goldy_task_graph_render_pass_begin(
            graph_.ptr_.get(), label, target.get()));
        active_ = true;
    }

    ~RenderPass() noexcept {
        if (active_) {
            // Do not throw from a destructor.
            goldy_task_graph_render_pass_finish(graph_.ptr_.get());
            active_ = false;
        }
    }

    RenderPass(const RenderPass&) = delete;
    RenderPass& operator=(const RenderPass&) = delete;
    RenderPass(RenderPass&&) = delete;
    RenderPass& operator=(RenderPass&&) = delete;

    RenderPass& bind_buffer(const Buffer& buffer, NodeAccess access) {
        detail::throw_on_result(goldy_task_graph_render_pass_bind_buffer(
            graph_.ptr_.get(), buffer.get(), static_cast<GoldyNodeAccess>(access)));
        return *this;
    }

    RenderPass& bind_buffer_view(const BufferView& view, NodeAccess access) {
        detail::throw_on_result(goldy_task_graph_render_pass_bind_buffer_view(
            graph_.ptr_.get(), view.get(), static_cast<GoldyNodeAccess>(access)));
        return *this;
    }

    RenderPass& bind_resource_index(uint32_t scattered_index) {
        const uint32_t pair[2] = {0, scattered_index};
        detail::throw_on_result(goldy_task_graph_render_pass_bind_resources_typed(
            graph_.ptr_.get(), pair, 1));
        return *this;
    }

    RenderPass& bind_resources(std::span<const Buffer* const> buffers) {
        if (buffers.empty()) {
            return *this;
        }
        std::vector<const GoldyBuffer*> ptrs;
        ptrs.reserve(buffers.size());
        for (const Buffer* buf : buffers) {
            ptrs.push_back(buf->get());
        }
        detail::throw_on_result(goldy_task_graph_render_pass_bind_resources(
            graph_.ptr_.get(), ptrs.data(), static_cast<uint32_t>(ptrs.size())));
        return *this;
    }

    RenderPass& clear(const Color& color) {
        detail::throw_on_result(goldy_task_graph_render_pass_clear(graph_.ptr_.get(), color));
        return *this;
    }

    RenderPass& clear_depth(float depth = 1.0f) {
        detail::throw_on_result(goldy_task_graph_render_pass_clear_depth(graph_.ptr_.get(), depth));
        return *this;
    }

    RenderPass& set_pipeline(const RenderPipeline& pipeline) {
        detail::throw_on_result(goldy_task_graph_render_pass_set_pipeline(
            graph_.ptr_.get(), pipeline.get()));
        return *this;
    }

    RenderPass& set_vertex_buffer(uint32_t slot, const Buffer& buffer) {
        detail::throw_on_result(goldy_task_graph_render_pass_set_vertex_buffer(
            graph_.ptr_.get(), slot, buffer.get()));
        return *this;
    }

    RenderPass& set_vertex_buffer(uint32_t slot, const Buffer& buffer, uint64_t offset) {
        detail::throw_on_result(goldy_task_graph_render_pass_set_vertex_buffer_offset(
            graph_.ptr_.get(), slot, buffer.get(), offset));
        return *this;
    }

    RenderPass& set_index_buffer(const Buffer& buffer, GoldyIndexFormat format) {
        detail::throw_on_result(goldy_task_graph_render_pass_set_index_buffer(
            graph_.ptr_.get(), buffer.get(), format));
        return *this;
    }

    RenderPass& draw(uint32_t first_vertex, uint32_t vertex_count,
                     uint32_t first_instance = 0, uint32_t instance_count = 1) {
        detail::throw_on_result(goldy_task_graph_render_pass_draw(
            graph_.ptr_.get(), first_vertex, vertex_count, first_instance, instance_count));
        return *this;
    }

    RenderPass& draw_indexed(uint32_t first_index, uint32_t index_count, int32_t base_vertex = 0,
                               uint32_t first_instance = 0, uint32_t instance_count = 1) {
        detail::throw_on_result(goldy_task_graph_render_pass_draw_indexed(
            graph_.ptr_.get(), first_index, index_count, base_vertex, first_instance, instance_count));
        return *this;
    }

    RenderPass& draw_fullscreen() {
        detail::throw_on_result(goldy_task_graph_render_pass_draw_fullscreen(graph_.ptr_.get()));
        return *this;
    }

    void finish() {
        if (!active_) {
            return;
        }
        detail::throw_on_result(goldy_task_graph_render_pass_finish(graph_.ptr_.get()));
        active_ = false;
    }

private:
    TaskGraph& graph_;
    bool active_ = false;
};

inline TaskGraph::RenderPass TaskGraph::render_pass(const char* label, const RenderTarget& target) {
    return RenderPass(*this, label, target);
}

/**
 * @brief RAII scope for recording one compute dispatch node on a task graph.
 */
class TaskGraph::ComputeNode {
public:
    ComputeNode(TaskGraph& graph, const char* label, const ComputePipeline& pipeline,
                uint32_t wg_x = 1, uint32_t wg_y = 1, uint32_t wg_z = 1)
        : graph_(graph), wg_x_(wg_x), wg_y_(wg_y), wg_z_(wg_z) {
        detail::throw_on_result(goldy_task_graph_compute_node_begin(
            graph_.ptr_.get(), label, pipeline.get()));
        active_ = true;
    }

    ~ComputeNode() noexcept {
        if (active_) {
            goldy_task_graph_compute_node_dispatch(graph_.ptr_.get(), wg_x_, wg_y_, wg_z_);
            active_ = false;
        }
    }

    ComputeNode(const ComputeNode&) = delete;
    ComputeNode& operator=(const ComputeNode&) = delete;
    ComputeNode(ComputeNode&&) = delete;
    ComputeNode& operator=(ComputeNode&&) = delete;

    ComputeNode& bind_buffer(const Buffer& buffer, NodeAccess access) {
        detail::throw_on_result(goldy_task_graph_compute_node_bind_buffer(
            graph_.ptr_.get(), buffer.get(), static_cast<GoldyNodeAccess>(access)));
        return *this;
    }

    ComputeNode& bind_buffer_view(const BufferView& view, NodeAccess access) {
        detail::throw_on_result(goldy_task_graph_compute_node_bind_buffer_view(
            graph_.ptr_.get(), view.get(), static_cast<GoldyNodeAccess>(access)));
        return *this;
    }

    ComputeNode& bind_resources_raw(std::span<const uint32_t> indices) {
        detail::throw_on_result(goldy_task_graph_compute_node_bind_resources_raw(
            graph_.ptr_.get(), indices.data(), static_cast<uint32_t>(indices.size())));
        return *this;
    }

    void dispatch(uint32_t x, uint32_t y = 1, uint32_t z = 1) {
        if (!active_) return;
        detail::throw_on_result(goldy_task_graph_compute_node_dispatch(
            graph_.ptr_.get(), x, y, z));
        active_ = false;
    }

private:
    TaskGraph& graph_;
    uint32_t wg_x_;
    uint32_t wg_y_;
    uint32_t wg_z_;
    bool active_ = false;
};

inline TaskGraph::ComputeNode TaskGraph::compute_node(const char* label, const ComputePipeline& pipeline) {
    return ComputeNode(*this, label, pipeline);
}

inline void TaskGraph::write_buffer(const Buffer& buffer, uint64_t offset, std::span<const uint8_t> data) {
    detail::throw_on_result(goldy_task_graph_write_buffer(
        ptr_.get(), buffer.get(), offset, data.data(), data.size()));
}

inline SurfaceFrame Surface::submit_graph_to_frame(TaskGraph& graph, SurfaceFrame frame) {
    GoldySurfaceFrame* raw = frame.release();
    detail::throw_on_result(
        goldy_surface_submit_graph_to_frame(ptr_.get(), graph.get(), raw));
    return SurfaceFrame(raw);
}

inline void Surface::present(SurfaceFrame frame) {
    detail::throw_on_result(goldy_surface_present(ptr_.get(), frame.release()));
}

// =============================================================================
// ComputePipeline
// =============================================================================

/**
 * @brief A compute shader pipeline.
 */
class ComputePipeline {
public:
    /**
     * @brief Create a compute pipeline.
     * @param device The device.
     * @param shader The compute shader.
     * @throws Exception if creation fails.
     */
    ComputePipeline(const Device& device, const ShaderModule& shader) {
        GoldyComputePipeline* ptr = goldy_compute_pipeline_create(device.get(), shader.get());
        if (!ptr) {
            throw Exception::from_last_error();
        }
        ptr_.reset(ptr);
    }

    ComputePipeline(const ComputePipeline&) = delete;
    ComputePipeline& operator=(const ComputePipeline&) = delete;
    ComputePipeline(ComputePipeline&&) = default;
    ComputePipeline& operator=(ComputePipeline&&) = default;

    /**
     * @brief Get raw pointer (for advanced use).
     */
    GoldyComputePipeline* get() const { return ptr_.get(); }

private:
    std::unique_ptr<GoldyComputePipeline, detail::ComputePipelineDeleter> ptr_;
};

// =============================================================================
// ComputeEncoder
// =============================================================================

/**
 * @brief Records compute commands.
 */
class ComputeEncoder {
public:
    /**
     * @brief Create a new compute encoder.
     */
    ComputeEncoder() {
        ptr_.reset(goldy_compute_encoder_create());
    }

    ComputeEncoder(const ComputeEncoder&) = delete;
    ComputeEncoder& operator=(const ComputeEncoder&) = delete;
    ComputeEncoder(ComputeEncoder&&) = default;
    ComputeEncoder& operator=(ComputeEncoder&&) = default;

    /**
     * @brief Set the compute pipeline.
     */
    void set_pipeline(const ComputePipeline& pipeline) {
        goldy_compute_encoder_set_pipeline(ptr_.get(), pipeline.get());
    }

    /**
     * @brief Dispatch compute workgroups.
     */
    void dispatch(uint32_t x, uint32_t y = 1, uint32_t z = 1) {
        goldy_compute_encoder_dispatch(ptr_.get(), x, y, z);
    }

    /**
     * @brief Execute the compute commands.
     * @param device The device to execute on.
     * @throws Exception if execution fails.
     */
    void execute(const Device& device) {
        GoldyResult result = goldy_compute_encoder_execute(ptr_.get(), device.get());
        if (result != GOLDY_RESULT_OK) {
            throw Exception::from_last_error();
        }
    }

    /**
     * @brief Get raw pointer (for advanced use).
     */
    GoldyComputeEncoder* get() const { return ptr_.get(); }

private:
    std::unique_ptr<GoldyComputeEncoder, detail::ComputeEncoderDeleter> ptr_;
};

// =============================================================================
// Texture
// =============================================================================

/**
 * @brief A GPU texture.
 *
 * Textures hold image data on the GPU with a specific spatial access pattern:
 * - TextureKind::Interpolated: Hardware filtering between neighbors (Texture2D with sampler)
 * - TextureKind::Direct: Direct 2D indexing, no filtering (RWTexture2D)
 * - TextureKind::DirectInterpolated: Both storage (UAV) and sampled (SRV) access
 */
class Texture {
public:
    /**
     * @brief Create a texture.
     * @param device The device.
     * @param width Width in pixels.
     * @param height Height in pixels.
     * @param format Texture format.
     * @param access Spatial access pattern.
     * @param flags Texture flags (copy operations, render target).
     * @throws Exception if creation fails.
     */
    Texture(const Device& device, uint32_t width, uint32_t height,
            GoldyTextureFormat format, TextureKind access, uint32_t flags = 0) {
        GoldyTexture* ptr = goldy_texture_create(device.get(), width, height, format,
            static_cast<GoldyTextureKind>(access), GoldyTextureFlags{flags});
        if (!ptr) {
            throw Exception::from_last_error();
        }
        ptr_.reset(ptr);
    }

    Texture(const Texture&) = delete;
    Texture& operator=(const Texture&) = delete;
    Texture(Texture&&) = default;
    Texture& operator=(Texture&&) = default;

    /**
     * @brief Get width in pixels.
     */
    uint32_t width() const { return goldy_texture_width(ptr_.get()); }

    /**
     * @brief Get height in pixels.
     */
    uint32_t height() const { return goldy_texture_height(ptr_.get()); }

    /**
     * @brief Get texture format.
     */
    GoldyTextureFormat format() const { return goldy_texture_format(ptr_.get()); }

    /**
     * @brief Get raw pointer (for advanced use).
     */
    GoldyTexture* get() const { return ptr_.get(); }

private:
    std::unique_ptr<GoldyTexture, detail::TextureDeleter> ptr_;
};

// =============================================================================
// Sampler
// =============================================================================

/**
 * @brief A texture sampler.
 */
class Sampler {
public:
    /**
     * @brief Create a sampler with default settings.
     * @param device The device.
     * @throws Exception if creation fails.
     */
    explicit Sampler(const Device& device) {
        GoldySampler* ptr = goldy_sampler_create_default(device.get());
        if (!ptr) {
            throw Exception::from_last_error();
        }
        ptr_.reset(ptr);
    }

    /**
     * @brief Create a sampler with custom settings.
     * @param device The device.
     * @param desc Sampler descriptor.
     * @throws Exception if creation fails.
     */
    Sampler(const Device& device, const GoldySamplerDesc& desc) {
        GoldySampler* ptr = goldy_sampler_create(device.get(), &desc);
        if (!ptr) {
            throw Exception::from_last_error();
        }
        ptr_.reset(ptr);
    }

    Sampler(const Sampler&) = delete;
    Sampler& operator=(const Sampler&) = delete;
    Sampler(Sampler&&) = default;
    Sampler& operator=(Sampler&&) = default;

    /**
     * @brief Get raw pointer (for advanced use).
     */
    GoldySampler* get() const { return ptr_.get(); }

private:
    std::unique_ptr<GoldySampler, detail::SamplerDeleter> ptr_;
};

// =============================================================================
// Utility functions
// =============================================================================

/**
 * @brief Get the last error message (if any).
 * @return Error message or empty string if no error.
 */
inline std::string get_last_error() {
    const char* error = goldy_get_last_error();
    return error ? std::string(error) : std::string();
}

/**
 * @brief Clear the last error message.
 */
inline void clear_error() {
    goldy_clear_error();
}

} // namespace goldy

#endif // GOLDY_HPP

