/*
 * Goldy GPU Library - C++ RAII Wrapper
 *
 * MIT License - Copyright (c) 2025 KOB3 Contributors
 *
 * This header provides modern C++ RAII wrappers around the Goldy C API.
 * All resources are automatically cleaned up when they go out of scope.
 *
 * Usage:
 *   #include <goldy.hpp>
 *
 *   goldy::Instance instance;
 *   goldy::Device device = instance.create_device(GOLDY_DEVICE_TYPE_DISCRETE_GPU);
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
class Buffer;
class ShaderModule;
class RenderPipeline;
class RenderTarget;
class CommandEncoder;
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
 * @brief Adapter information returned by Instance::enumerate_adapters().
 */
struct AdapterInfo {
    uint32_t id;
    GoldyDeviceType device_type;
    std::string name;
    std::string vendor;
};

// =============================================================================
// Access pattern enums (C++ friendly)
// =============================================================================

/**
 * @brief Data access pattern for buffers.
 */
enum class DataAccess {
    /// Any thread, any address, read/write (StructuredBuffer, RWStructuredBuffer).
    Scattered = 0,
    /// All threads same address, broadcast optimized (ConstantBuffer).
    Broadcast = 1,
};

/**
 * @brief Spatial access pattern for textures.
 */
enum class SpatialAccess {
    /// Hardware filtering between neighbors (Texture2D with sampler).
    Interpolated = 0,
    /// Direct 2D/3D indexing, no filtering (RWTexture2D).
    Direct = 1,
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

struct ShaderDeleter {
    void operator()(GoldyShaderModule* p) const { if (p) goldy_shader_destroy(p); }
};

struct RenderPipelineDeleter {
    void operator()(GoldyRenderPipeline* p) const { if (p) goldy_render_pipeline_destroy(p); }
};

struct RenderTargetDeleter {
    void operator()(GoldyRenderTarget* p) const { if (p) goldy_render_target_destroy(p); }
};

struct CommandEncoderDeleter {
    void operator()(GoldyCommandEncoder* p) const { if (p) goldy_encoder_destroy(p); }
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
                    info.device_type,
                    std::string(info.name),
                    std::string(info.vendor)
                });
            }
        }
        return adapters;
    }

    /**
     * @brief Create a device with preferred type.
     * @param preferred_type The preferred GPU type (discrete, integrated, etc.)
     * @throws Exception if device creation fails.
     */
    Device create_device(GoldyDeviceType preferred_type = GOLDY_DEVICE_TYPE_DISCRETE_GPU);

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
    explicit Device(GoldyDevice* ptr) : ptr_(ptr) {}
    std::unique_ptr<GoldyDevice, detail::DeviceDeleter> ptr_;
};

// Instance methods that return Device
inline Device Instance::create_device(GoldyDeviceType preferred_type) {
    GoldyDevice* ptr = goldy_instance_create_device(ptr_.get(), preferred_type);
    if (!ptr) {
        throw Exception::from_last_error();
    }
    return Device(ptr);
}

inline Device Instance::create_device_for_adapter(uint32_t adapter_id) {
    GoldyDevice* ptr = goldy_instance_create_device_for_adapter(ptr_.get(), adapter_id);
    if (!ptr) {
        throw Exception::from_last_error();
    }
    return Device(ptr);
}

// =============================================================================
// Buffer
// =============================================================================

/**
 * @brief A GPU buffer for data storage.
 *
 * Buffers hold data on the GPU with a specific access pattern:
 * - DataAccess::Scattered: Any thread can access any address (StructuredBuffer, RWStructuredBuffer)
 * - DataAccess::Broadcast: All threads read same address, broadcast optimized (ConstantBuffer)
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
    Buffer(const Device& device, uint64_t size, DataAccess access) {
        GoldyBuffer* ptr = goldy_buffer_create(device.get(), size, static_cast<GoldyDataAccess>(access));
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
    Buffer(const Device& device, std::span<const uint8_t> data, DataAccess access) {
        GoldyBuffer* ptr = goldy_buffer_create_with_data(
            device.get(), data.data(), data.size(), static_cast<GoldyDataAccess>(access));
        if (!ptr) {
            throw Exception::from_last_error();
        }
        ptr_.reset(ptr);
    }

    /**
     * @brief Create a buffer from typed data.
     */
    template<typename T>
    Buffer(const Device& device, std::span<const T> data, DataAccess access)
        : Buffer(device, std::span<const uint8_t>(
            reinterpret_cast<const uint8_t*>(data.data()),
            data.size() * sizeof(T)), access) {}

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
    DataAccess access() const { return static_cast<DataAccess>(goldy_buffer_access(ptr_.get())); }

    /**
     * @brief Get raw pointer (for advanced use).
     */
    GoldyBuffer* get() const { return ptr_.get(); }

private:
    std::unique_ptr<GoldyBuffer, detail::BufferDeleter> ptr_;
};

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
// CommandEncoder
// =============================================================================

/**
 * @brief Records rendering commands.
 */
class CommandEncoder {
public:
    /**
     * @brief Create a new command encoder.
     */
    CommandEncoder() {
        ptr_.reset(goldy_encoder_create());
    }

    CommandEncoder(const CommandEncoder&) = delete;
    CommandEncoder& operator=(const CommandEncoder&) = delete;
    CommandEncoder(CommandEncoder&&) = default;
    CommandEncoder& operator=(CommandEncoder&&) = default;

    /**
     * @brief Clear the color target.
     */
    void clear(const Color& color) {
        goldy_encoder_clear(ptr_.get(), color);
    }

    /**
     * @brief Clear the depth buffer.
     */
    void clear_depth(float depth = 1.0f) {
        goldy_encoder_clear_depth(ptr_.get(), depth);
    }

    /**
     * @brief Set the render pipeline.
     */
    void set_pipeline(const RenderPipeline& pipeline) {
        goldy_encoder_set_pipeline(ptr_.get(), pipeline.get());
    }

    /**
     * @brief Set a vertex buffer.
     */
    void set_vertex_buffer(uint32_t slot, const Buffer& buffer) {
        goldy_encoder_set_vertex_buffer(ptr_.get(), slot, buffer.get());
    }

    /**
     * @brief Set a vertex buffer with offset.
     */
    void set_vertex_buffer(uint32_t slot, const Buffer& buffer, uint64_t offset) {
        goldy_encoder_set_vertex_buffer_offset(ptr_.get(), slot, buffer.get(), offset);
    }

    /**
     * @brief Set an index buffer.
     */
    void set_index_buffer(const Buffer& buffer, GoldyIndexFormat format) {
        goldy_encoder_set_index_buffer(ptr_.get(), buffer.get(), format);
    }

    /**
     * @brief Bind resource slots for rendering.
     *
     * Pass the buffers whose indices should be bound to shader resource slots.
     * The indices are bound in order, so buffers[0] becomes slot 0,
     * buffers[1] becomes slot 1, etc.
     *
     * @param buffers Span of buffer pointers to bind to shader resource slots.
     */
    void bind_resources(std::span<const Buffer* const> buffers) {
        if (buffers.empty()) return;
        
        std::vector<const GoldyBuffer*> ptrs;
        ptrs.reserve(buffers.size());
        for (const auto* buf : buffers) {
            ptrs.push_back(buf->get());
        }
        goldy_encoder_bind_resources(ptr_.get(), ptrs.data(), static_cast<uint32_t>(ptrs.size()));
    }

    /**
     * @brief Bind a single buffer to a resource slot (convenience overload).
     */
    void bind_resources(const Buffer& buffer) {
        const GoldyBuffer* ptr = buffer.get();
        goldy_encoder_bind_resources(ptr_.get(), &ptr, 1);
    }

    /**
     * @brief Bind resource slots from an initializer list (convenience overload).
     */
    void bind_resources(std::initializer_list<const Buffer*> buffers) {
        bind_resources(std::span<const Buffer* const>{buffers.begin(), buffers.size()});
    }

    /**
     * @brief Draw primitives.
     */
    void draw(uint32_t vertex_count, uint32_t instance_count = 1,
              uint32_t first_vertex = 0, uint32_t first_instance = 0) {
        goldy_encoder_draw(ptr_.get(), first_vertex, vertex_count, first_instance, instance_count);
    }

    /**
     * @brief Draw indexed primitives.
     */
    void draw_indexed(uint32_t index_count, uint32_t instance_count = 1,
                      uint32_t first_index = 0, int32_t base_vertex = 0,
                      uint32_t first_instance = 0) {
        goldy_encoder_draw_indexed(ptr_.get(), first_index, index_count,
                                   base_vertex, first_instance, instance_count);
    }

    /**
     * @brief Release ownership of the underlying pointer.
     *
     * Used when passing to render target. After this call, the encoder is invalid.
     */
    GoldyCommandEncoder* release() { return ptr_.release(); }

    /**
     * @brief Get raw pointer (for advanced use).
     */
    GoldyCommandEncoder* get() const { return ptr_.get(); }

private:
    std::unique_ptr<GoldyCommandEncoder, detail::CommandEncoderDeleter> ptr_;
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
     * @brief Render commands to the target.
     *
     * This consumes the encoder.
     * @param encoder The command encoder (moved from).
     * @throws Exception if rendering fails.
     */
    void render(CommandEncoder encoder) {
        GoldyResult result = goldy_render_target_render(ptr_.get(), encoder.release());
        if (result != GOLDY_RESULT_OK) {
            throw Exception::from_last_error();
        }
    }

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
 * - SpatialAccess::Interpolated: Hardware filtering between neighbors (Texture2D with sampler)
 * - SpatialAccess::Direct: Direct 2D indexing, no filtering (RWTexture2D)
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
            GoldyTextureFormat format, SpatialAccess access, uint32_t flags = 0) {
        GoldyTexture* ptr = goldy_texture_create(device.get(), width, height, format,
            static_cast<GoldySpatialAccess>(access), GoldyTextureFlags{flags});
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

