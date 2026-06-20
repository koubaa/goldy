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
class Context;
class Adapter;
class RetainedPool;
class Buffer;
class Parcel;
class RecordBuilder;
class ShaderModule;
class RenderPipeline;
class RenderTarget;
class Surface;
class Scheme;
class SchemeRenderTargetLease;
class PresentGrant;
class PresentLease;
class SwapchainPool;
class ComputePipeline;
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

struct ContextDeleter {
    void operator()(GoldyContext* p) const { if (p) goldy_context_destroy(p); }
};

struct RetainedPoolDeleter {
    void operator()(GoldyRetainedPool* p) const { if (p) goldy_retained_pool_destroy(p); }
};

struct BufferDeleter {
    void operator()(GoldyBuffer* p) const { if (p) goldy_buffer_destroy(p); }
};

struct ParcelDeleter {
    void operator()(GoldyParcel* p) const { if (p) goldy_parcel_destroy(p); }
};

struct RecordBuilderDeleter {
    void operator()(GoldyRecordBuilder* p) const { if (p) goldy_record_builder_destroy(p); }
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

struct SchemeDeleter {
    void operator()(GoldyScheme* p) const { if (p) goldy_scheme_destroy(p); }
};

struct SchemeSubmissionDeleter {
    void operator()(GoldySchemeSubmission* p) const { if (p) goldy_scheme_submission_destroy(p); }
};

struct ReadGrantDeleter {
    void operator()(GoldyReadGrant* p) const { if (p) goldy_read_grant_destroy(p); }
};

struct PresentGrantDeleter {
    void operator()(GoldyPresentGrant* p) const { if (p) goldy_present_grant_destroy(p); }
};

struct SchemeRenderTargetLeaseDeleter {
    void operator()(GoldySchemeRenderTargetLease* p) const {
        if (p) goldy_scheme_render_target_lease_destroy(p);
    }
};

struct PresentLeaseDeleter {
    void operator()(GoldyPresentLease* p) const { if (p) goldy_present_lease_destroy(p); }
};

struct SwapchainPoolDeleter {
    void operator()(GoldySwapchainPool* p) const { if (p) goldy_swapchain_pool_destroy(p); }
};

struct ComputePipelineDeleter {
    void operator()(GoldyComputePipeline* p) const { if (p) goldy_compute_pipeline_destroy(p); }
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
// Context
// =============================================================================

/**
 * @brief Submission context for retained Scheme instances.
 */
class Context {
public:
    explicit Context(const Device& device) {
        GoldyContext* ptr = goldy_context_create(device.get());
        if (!ptr) {
            throw Exception::from_last_error();
        }
        ptr_.reset(ptr);
    }

    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    Context(Context&&) = default;
    Context& operator=(Context&&) = default;

    GoldyContext* get() const { return ptr_.get(); }

private:
    std::unique_ptr<GoldyContext, detail::ContextDeleter> ptr_;
};

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
// RetainedPool / Buffer / Parcel / RecordBuilder
// =============================================================================

/**
 * @brief Deed-governed pool for retained GPU buffers and texture parcels.
 */
class RetainedPool {
public:
    explicit RetainedPool(const Device& device) {
        GoldyRetainedPool* ptr = goldy_retained_pool_create(device.get());
        if (!ptr) {
            throw Exception::from_last_error();
        }
        ptr_.reset(ptr);
    }

    RetainedPool(const RetainedPool&) = delete;
    RetainedPool& operator=(const RetainedPool&) = delete;
    RetainedPool(RetainedPool&&) = default;
    RetainedPool& operator=(RetainedPool&&) = default;

    [[nodiscard]] RecordBuilder record();

    /**
     * @brief Acquire an uninitialized retained buffer.
     */
    [[nodiscard]] Buffer acquire_buffer(uint64_t size, BufferKind access, uint32_t element_stride = 1);

    /**
     * @brief Acquire a retained buffer initialized with typed data.
     */
    template<typename T>
    [[nodiscard]] Buffer acquire_buffer_with_data(std::span<const T> data, BufferKind access);

    template<typename T, typename Allocator>
    [[nodiscard]] Buffer acquire_buffer_with_data(const std::vector<T, Allocator>& data, BufferKind access);

    /**
     * @brief Acquire a retained buffer initialized with raw bytes and an explicit element stride.
     */
    [[nodiscard]] Buffer acquire_buffer_bytes(std::span<const uint8_t> data, BufferKind access,
                                              uint32_t element_stride = 1);

    [[nodiscard]] Parcel acquire_texture(uint32_t width, uint32_t height, GoldyTextureFormat format,
                                         GoldyTextureKind kind, GoldyTextureFlags flags,
                                         std::span<const uint8_t> init = {});

    GoldyRetainedPool* get() const { return ptr_.get(); }

private:
    std::unique_ptr<GoldyRetainedPool, detail::RetainedPoolDeleter> ptr_;
};

/**
 * @brief Opaque retained GPU parcel (texture parcels; buffer units use Buffer + unit index).
 */
class Parcel {
public:
    Parcel() = default;

    explicit Parcel(GoldyParcel* ptr) : ptr_(ptr) {}

    Parcel(const Parcel&) = delete;
    Parcel& operator=(const Parcel&) = delete;
    Parcel(Parcel&&) = default;
    Parcel& operator=(Parcel&&) = default;

    uint64_t byte_size() const { return goldy_parcel_byte_size(ptr_.get()); }

    GoldyParcel* get() const { return ptr_.get(); }

private:
    std::unique_ptr<GoldyParcel, detail::ParcelDeleter> ptr_;
};

/**
 * @brief Acquired retained GPU buffer (possibly partitioned into bindable units).
 */
class Buffer {
public:
    Buffer() = default;

    explicit Buffer(GoldyBuffer* ptr) : ptr_(ptr) {}

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&&) = default;
    Buffer& operator=(Buffer&&) = default;

    uint64_t byte_size() const { return goldy_buffer_byte_size(ptr_.get()); }

    uint32_t unit_count() const { return goldy_buffer_unit_count(ptr_.get()); }

    uint64_t unit_byte_size(uint32_t unit) const {
        return goldy_buffer_unit_byte_size(ptr_.get(), unit);
    }

    std::vector<uint8_t> unit_read_to_cpu(uint32_t unit, const Device& device) const {
        std::vector<uint8_t> output(unit_byte_size(unit));
        detail::throw_on_result(goldy_buffer_unit_read_to_cpu(
            ptr_.get(), unit, device.get(), output.data(), output.size()));
        return output;
    }

    /**
     * @brief Borrow one bindable unit as an owned parcel handle (source buffer must outlive GPU use).
     */
    [[nodiscard]] Parcel field(uint32_t unit) const;

    GoldyBuffer* get() const { return ptr_.get(); }

private:
    std::unique_ptr<GoldyBuffer, detail::BufferDeleter> ptr_;
};

/**
 * @brief Builder for a retained record buffer (one backing buffer, multiple sub-views).
 */
class RecordBuilder {
public:
    RecordBuilder() {
        GoldyRecordBuilder* ptr = goldy_record_builder_create();
        if (!ptr) {
            throw Exception::from_last_error();
        }
        ptr_.reset(ptr);
    }

    RecordBuilder(const RecordBuilder&) = delete;
    RecordBuilder& operator=(const RecordBuilder&) = delete;
    RecordBuilder(RecordBuilder&&) = default;
    RecordBuilder& operator=(RecordBuilder&&) = default;

    template<typename T>
    uint32_t emplace(std::span<const T> data) {
        return emplace_named(nullptr, data);
    }

    template<typename T>
    uint32_t emplace_named(const char* name, std::span<const T> data) {
        return emplace_named_bytes(
            name,
            std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(data.data()), data.size_bytes()),
            data.size(),
            static_cast<uint32_t>(sizeof(T)));
    }

    template<typename T, typename Allocator>
    uint32_t emplace(const std::vector<T, Allocator>& data) {
        return emplace(std::span<const T>(data.data(), data.size()));
    }

    template<typename T, typename Allocator>
    uint32_t emplace_named(const char* name, const std::vector<T, Allocator>& data) {
        return emplace_named(name, std::span<const T>(data.data(), data.size()));
    }

    uint32_t emplace_bytes(std::span<const uint8_t> data, uint64_t element_count, uint32_t element_stride) {
        return emplace_named_bytes(nullptr, data, element_count, element_stride);
    }

    uint32_t emplace_named_bytes(const char* name, std::span<const uint8_t> data,
                                 uint64_t element_count, uint32_t element_stride) {
        uint32_t slot = goldy_record_builder_emplace(
            ptr_.get(), name, data.data(), data.size(), element_count, element_stride);
        if (slot == UINT32_MAX) {
            throw Exception::from_last_error();
        }
        return slot;
    }

    uint32_t reserve(const char* name, uint64_t element_count, uint32_t element_stride) {
        uint32_t slot = goldy_record_builder_reserve(
            ptr_.get(), name, element_count, element_stride);
        if (slot == UINT32_MAX) {
            throw Exception::from_last_error();
        }
        return slot;
    }

    [[nodiscard]] Buffer build(RetainedPool& pool);

private:
    std::unique_ptr<GoldyRecordBuilder, detail::RecordBuilderDeleter> ptr_;
};

inline Parcel Buffer::field(uint32_t unit) const {
    GoldyParcel* parcel = goldy_buffer_field(ptr_.get(), unit);
    if (!parcel) {
        throw Exception::from_last_error();
    }
    return Parcel(parcel);
}

template<typename T>
inline Buffer RetainedPool::acquire_buffer_with_data(std::span<const T> data, BufferKind access) {
    GoldyBuffer* ptr = goldy_retained_pool_acquire_buffer(
        ptr_.get(),
        data.size_bytes(),
        static_cast<GoldyBufferKind>(access),
        static_cast<uint32_t>(sizeof(T)),
        reinterpret_cast<const uint8_t*>(data.data()),
        data.size_bytes());
    if (!ptr) {
        throw Exception::from_last_error();
    }
    return Buffer(ptr);
}

template<typename T, typename Allocator>
inline Buffer RetainedPool::acquire_buffer_with_data(const std::vector<T, Allocator>& data, BufferKind access) {
    return acquire_buffer_with_data(std::span<const T>(data.data(), data.size()), access);
}

inline Buffer RetainedPool::acquire_buffer_bytes(std::span<const uint8_t> data, BufferKind access,
                                                  uint32_t element_stride) {
    GoldyBuffer* ptr = goldy_retained_pool_acquire_buffer(
        ptr_.get(),
        data.size(),
        static_cast<GoldyBufferKind>(access),
        element_stride,
        data.data(),
        data.size());
    if (!ptr) {
        throw Exception::from_last_error();
    }
    return Buffer(ptr);
}

inline RecordBuilder RetainedPool::record() {
    return RecordBuilder{};
}

inline Buffer RetainedPool::acquire_buffer(uint64_t size, BufferKind access, uint32_t element_stride) {
    GoldyBuffer* ptr = goldy_retained_pool_acquire_buffer(
        ptr_.get(), size, static_cast<GoldyBufferKind>(access), element_stride, nullptr, 0);
    if (!ptr) {
        throw Exception::from_last_error();
    }
    return Buffer(ptr);
}

inline Parcel RetainedPool::acquire_texture(uint32_t width, uint32_t height, GoldyTextureFormat format,
                                            GoldyTextureKind kind, GoldyTextureFlags flags,
                                            std::span<const uint8_t> init) {
    GoldyParcel* ptr = goldy_retained_pool_acquire_texture(
        ptr_.get(), width, height, format, kind, flags,
        init.empty() ? nullptr : init.data(), init.size());
    if (!ptr) {
        throw Exception::from_last_error();
    }
    return Parcel(ptr);
}

inline Buffer RecordBuilder::build(RetainedPool& pool) {
    GoldyBuffer* buffer = goldy_record_builder_build(ptr_.release(), pool.get());
    if (!buffer) {
        throw Exception::from_last_error();
    }
    return Buffer(buffer);
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
 * Consumed by Surface::present().
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
     * @brief Present a frame to the screen (consumes the frame).
     */
    void present(SurfaceFrame frame);

    GoldySurface* get() const { return ptr_.get(); }

private:
    std::unique_ptr<GoldySurface, detail::SurfaceDeleter> ptr_;
};

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
// Scheme
// =============================================================================

/**
 * @brief Per-submission identity returned by Scheme::submit().
 */
class SchemeSubmission {
public:
    SchemeSubmission() = default;

    explicit SchemeSubmission(GoldySchemeSubmission* submission) : ptr_(submission) {}

    SchemeSubmission(const SchemeSubmission&) = delete;
    SchemeSubmission& operator=(const SchemeSubmission&) = delete;
    SchemeSubmission(SchemeSubmission&&) = default;
    SchemeSubmission& operator=(SchemeSubmission&&) = default;

    [[nodiscard]] uint64_t timeline_value() const {
        return goldy_scheme_submission_timeline_value(ptr_.get());
    }

    void wait(const Context& ctx) const {
        detail::throw_on_result(goldy_scheme_submission_wait(ctx.get(), ptr_.get()));
    }

    GoldySchemeSubmission* get() const { return ptr_.get(); }

private:
    std::unique_ptr<GoldySchemeSubmission, detail::SchemeSubmissionDeleter> ptr_;
};

/**
 * @brief Read easement grant recorded once via Scheme::grant_read().
 */
class ReadGrant {
public:
    ReadGrant() = default;

    explicit ReadGrant(GoldyReadGrant* grant) : ptr_(grant) {}

    ReadGrant(const ReadGrant&) = delete;
    ReadGrant& operator=(const ReadGrant&) = delete;
    ReadGrant(ReadGrant&&) = default;
    ReadGrant& operator=(ReadGrant&&) = default;

    [[nodiscard]] uint64_t byte_size() const {
        return goldy_read_grant_byte_size(ptr_.get());
    }

    [[nodiscard]] std::vector<uint8_t> consume(const SchemeSubmission& submission) const {
        std::vector<uint8_t> output(byte_size());
        detail::throw_on_result(goldy_read_grant_consume(
            ptr_.get(), submission.get(), output.data(), output.size()));
        return output;
    }

    GoldyReadGrant* get() const { return ptr_.get(); }

private:
    std::unique_ptr<GoldyReadGrant, detail::ReadGrantDeleter> ptr_;
};

/**
 * @brief Present easement grant recorded once via Scheme::grant_present().
 */
class PresentGrant {
public:
    PresentGrant() = default;

    explicit PresentGrant(GoldyPresentGrant* grant) : ptr_(grant) {}

    PresentGrant(const PresentGrant&) = delete;
    PresentGrant& operator=(const PresentGrant&) = delete;
    PresentGrant(PresentGrant&&) = default;
    PresentGrant& operator=(PresentGrant&&) = default;

    void consume(const SchemeSubmission& submission) const {
        detail::throw_on_result(goldy_present_grant_consume(ptr_.get(), submission.get()));
    }

    GoldyPresentGrant* get() const { return ptr_.get(); }

private:
    std::unique_ptr<GoldyPresentGrant, detail::PresentGrantDeleter> ptr_;
};

/**
 * @brief Stable render-target lease declared on a Scheme.
 */
class SchemeRenderTargetLease {
public:
    SchemeRenderTargetLease() = default;

    explicit SchemeRenderTargetLease(GoldySchemeRenderTargetLease* lease) : ptr_(lease) {}

    SchemeRenderTargetLease(const SchemeRenderTargetLease&) = delete;
    SchemeRenderTargetLease& operator=(const SchemeRenderTargetLease&) = delete;
    SchemeRenderTargetLease(SchemeRenderTargetLease&&) = default;
    SchemeRenderTargetLease& operator=(SchemeRenderTargetLease&&) = default;

    GoldySchemeRenderTargetLease* get() const { return ptr_.get(); }

private:
    friend class Scheme;
    std::unique_ptr<GoldySchemeRenderTargetLease, detail::SchemeRenderTargetLeaseDeleter> ptr_;
};

/**
 * @brief Stable present lease from a SwapchainPool.
 */
class PresentLease {
public:
    PresentLease() = default;

    explicit PresentLease(GoldyPresentLease* lease) : ptr_(lease) {}

    PresentLease(const PresentLease&) = delete;
    PresentLease& operator=(const PresentLease&) = delete;
    PresentLease(PresentLease&&) = default;
    PresentLease& operator=(PresentLease&&) = default;

    GoldyPresentLease* get() const { return ptr_.get(); }

private:
    std::unique_ptr<GoldyPresentLease, detail::PresentLeaseDeleter> ptr_;
};

/**
 * @brief Swapchain pool for present-on-scheme.
 */
class SwapchainPool {
public:
#if defined(_WIN32)
    SwapchainPool(const Context& ctx, void* hwnd, uint32_t depth = 3) {
        GoldySwapchainPool* ptr = goldy_swapchain_pool_create_win32(ctx.get(), hwnd, depth);
        if (!ptr) {
            throw Exception::from_last_error();
        }
        ptr_.reset(ptr);
    }
#elif defined(__APPLE__)
    SwapchainPool(const Context& ctx, void* ns_view, uint32_t depth = 3) {
        GoldySwapchainPool* ptr = goldy_swapchain_pool_create_appkit(ctx.get(), ns_view, depth);
        if (!ptr) {
            throw Exception::from_last_error();
        }
        ptr_.reset(ptr);
    }
#else
    SwapchainPool(const Context& ctx, void* wayland_display, void* wayland_surface, uint32_t depth = 3) {
        GoldySwapchainPool* ptr = goldy_swapchain_pool_create_wayland(
            ctx.get(), wayland_display, wayland_surface, depth);
        if (!ptr) {
            throw Exception::from_last_error();
        }
        ptr_.reset(ptr);
    }
#endif

    SwapchainPool(const SwapchainPool&) = delete;
    SwapchainPool& operator=(const SwapchainPool&) = delete;
    SwapchainPool(SwapchainPool&&) = default;
    SwapchainPool& operator=(SwapchainPool&&) = default;

    [[nodiscard]] PresentLease lease() {
        GoldyPresentLease* lease = goldy_swapchain_pool_lease(ptr_.get());
        if (!lease) {
            throw Exception::from_last_error();
        }
        return PresentLease{lease};
    }

    std::pair<uint32_t, uint32_t> size() const {
        return {goldy_swapchain_pool_width(ptr_.get()), goldy_swapchain_pool_height(ptr_.get())};
    }

    uint32_t width() const { return goldy_swapchain_pool_width(ptr_.get()); }
    uint32_t height() const { return goldy_swapchain_pool_height(ptr_.get()); }

    GoldyTextureFormat format() const { return goldy_swapchain_pool_format(ptr_.get()); }

    void resize(uint32_t width, uint32_t height) {
        detail::throw_on_result(goldy_swapchain_pool_resize(ptr_.get(), width, height));
    }

    GoldySwapchainPool* get() const { return ptr_.get(); }

private:
    std::unique_ptr<GoldySwapchainPool, detail::SwapchainPoolDeleter> ptr_;
};

/**
 * @brief Retained scheme bound to one Context.
 */
class Scheme {
public:
    class ComputeNode;
    class RenderPass;

    explicit Scheme(const Context& ctx) {
        GoldyScheme* ptr = goldy_scheme_create(ctx.get());
        if (!ptr) {
            throw Exception::from_last_error();
        }
        ptr_.reset(ptr);
    }

    Scheme(const Scheme&) = delete;
    Scheme& operator=(const Scheme&) = delete;
    Scheme(Scheme&&) = default;
    Scheme& operator=(Scheme&&) = default;

    uint32_t len() const { return goldy_scheme_len(ptr_.get()); }

    bool is_dirty() const { return goldy_scheme_is_dirty(ptr_.get()); }

    [[nodiscard]] ReadGrant grant_read(const Parcel& parcel) {
        GoldyReadGrant* grant = goldy_scheme_grant_read(ptr_.get(), parcel.get());
        if (!grant) {
            throw Exception::from_last_error();
        }
        return ReadGrant{grant};
    }

    [[nodiscard]] ReadGrant grant_read(const Buffer& buffer, uint32_t unit = 0) {
        return grant_read(buffer.field(unit));
    }

    [[nodiscard]] ReadGrant grant_read_texture(const Parcel& parcel) {
        GoldyReadGrant* grant = goldy_scheme_grant_read_texture(ptr_.get(), parcel.get());
        if (!grant) {
            throw Exception::from_last_error();
        }
        return ReadGrant{grant};
    }

    [[nodiscard]] SchemeRenderTargetLease lease_render_target(
        uint32_t width, uint32_t height, GoldyTextureFormat format,
        bool has_depth = false, GoldyDepthFormat depth_format = GOLDY_DEPTH_FORMAT_DEPTH24_PLUS) {
        GoldySchemeRenderTargetLease* lease = goldy_scheme_lease_render_target(
            ptr_.get(), width, height, format, has_depth, depth_format);
        if (!lease) {
            throw Exception::from_last_error();
        }
        return SchemeRenderTargetLease{lease};
    }

    void copy_to_texture(const SchemeRenderTargetLease& src, const Parcel& dst) {
        detail::throw_on_result(goldy_scheme_copy_to_texture(ptr_.get(), src.get(), dst.get()));
    }

    void copy_to_present(const SchemeRenderTargetLease& src, const PresentLease& dst) {
        detail::throw_on_result(goldy_scheme_copy_to_present(ptr_.get(), src.get(), dst.get()));
    }

    [[nodiscard]] PresentGrant grant_present(const PresentLease& lease) {
        GoldyPresentGrant* grant = goldy_scheme_grant_present(ptr_.get(), lease.get());
        if (!grant) {
            throw Exception::from_last_error();
        }
        return PresentGrant{grant};
    }

    [[nodiscard]] SchemeSubmission submit() {
        GoldySchemeSubmission* submission = nullptr;
        detail::throw_on_result(goldy_scheme_submit(ptr_.get(), &submission));
        return SchemeSubmission{submission};
    }

    [[nodiscard]] ComputeNode compute_node(const char* label, const ComputePipeline& pipeline);

    [[nodiscard]] RenderPass render_pass(const char* label, const SchemeRenderTargetLease& target);

    GoldyScheme* get() const { return ptr_.get(); }

private:
    friend class ComputeNode;
    friend class RenderPass;
    std::unique_ptr<GoldyScheme, detail::SchemeDeleter> ptr_;
};

/**
 * @brief RAII scope for recording one compute dispatch node on a scheme.
 */
class Scheme::ComputeNode {
public:
    ComputeNode(Scheme& scheme, const char* label, const ComputePipeline& pipeline,
                uint32_t wg_x = 1, uint32_t wg_y = 1, uint32_t wg_z = 1)
        : scheme_(scheme), wg_x_(wg_x), wg_y_(wg_y), wg_z_(wg_z) {
        detail::throw_on_result(goldy_scheme_compute_node_begin(
            scheme_.ptr_.get(), label, pipeline.get()));
        active_ = true;
    }

    ~ComputeNode() noexcept {
        if (active_) {
            goldy_scheme_compute_node_dispatch(scheme_.ptr_.get(), wg_x_, wg_y_, wg_z_);
            active_ = false;
        }
    }

    ComputeNode(const ComputeNode&) = delete;
    ComputeNode& operator=(const ComputeNode&) = delete;
    ComputeNode(ComputeNode&&) = delete;
    ComputeNode& operator=(ComputeNode&&) = delete;

    ComputeNode& with_parcel(const Parcel& parcel, NodeAccess node_access) {
        detail::throw_on_result(goldy_scheme_compute_node_with_parcel(
            scheme_.ptr_.get(), parcel.get(),
            static_cast<GoldyNodeAccess>(node_access)));
        return *this;
    }

    ComputeNode& with_field(const Buffer& buffer, uint32_t unit, NodeAccess node_access) {
        detail::throw_on_result(goldy_scheme_compute_node_with_field(
            scheme_.ptr_.get(), buffer.get(), unit,
            static_cast<GoldyNodeAccess>(node_access)));
        return *this;
    }

    ComputeNode& with_buffer(const Buffer& buffer, NodeAccess node_access) {
        return with_field(buffer, 0, node_access);
    }

    ComputeNode& with_param(uint32_t value) {
        detail::throw_on_result(goldy_scheme_compute_node_with_param(scheme_.ptr_.get(), value));
        return *this;
    }

    void dispatch(uint32_t x, uint32_t y = 1, uint32_t z = 1) {
        if (!active_) return;
        detail::throw_on_result(goldy_scheme_compute_node_dispatch(
            scheme_.ptr_.get(), x, y, z));
        active_ = false;
    }

private:
    Scheme& scheme_;
    uint32_t wg_x_;
    uint32_t wg_y_;
    uint32_t wg_z_;
    bool active_ = false;
};

inline Scheme::ComputeNode Scheme::compute_node(const char* label, const ComputePipeline& pipeline) {
    return ComputeNode(*this, label, pipeline);
}

/**
 * @brief RAII scope for recording one offscreen render pass on a scheme.
 */
class Scheme::RenderPass {
public:
    RenderPass(Scheme& scheme, const char* label, const SchemeRenderTargetLease& target)
        : scheme_(scheme) {
        detail::throw_on_result(goldy_scheme_render_pass_begin(
            scheme_.ptr_.get(), label, target.get()));
        active_ = true;
    }

    ~RenderPass() noexcept {
        if (active_) {
            goldy_scheme_render_pass_finish(scheme_.ptr_.get());
            active_ = false;
        }
    }

    RenderPass(const RenderPass&) = delete;
    RenderPass& operator=(const RenderPass&) = delete;
    RenderPass(RenderPass&&) = delete;
    RenderPass& operator=(RenderPass&&) = delete;

    RenderPass& with_parcel(const Parcel& parcel, NodeAccess access) {
        detail::throw_on_result(goldy_scheme_render_pass_with_parcel(
            scheme_.ptr_.get(), parcel.get(), static_cast<GoldyNodeAccess>(access)));
        return *this;
    }

    RenderPass& with_field(const Buffer& buffer, uint32_t unit, NodeAccess access) {
        detail::throw_on_result(goldy_scheme_render_pass_with_field(
            scheme_.ptr_.get(), buffer.get(), unit, static_cast<GoldyNodeAccess>(access)));
        return *this;
    }

    RenderPass& set_vertex_buffer(uint32_t slot, const Buffer& buffer, uint32_t unit = 0) {
        return set_vertex_buffer_parcel(slot, buffer.field(unit));
    }

    RenderPass& set_index_buffer(const Buffer& buffer, GoldyIndexFormat format, uint32_t unit = 0) {
        return set_index_buffer(buffer.field(unit), format);
    }

    RenderPass& clear(const Color& color) {
        detail::throw_on_result(goldy_scheme_render_pass_clear(scheme_.ptr_.get(), color));
        return *this;
    }

    RenderPass& clear_depth(float depth = 1.0f) {
        detail::throw_on_result(goldy_scheme_render_pass_clear_depth(scheme_.ptr_.get(), depth));
        return *this;
    }

    RenderPass& set_pipeline(const RenderPipeline& pipeline) {
        detail::throw_on_result(goldy_scheme_render_pass_set_pipeline(
            scheme_.ptr_.get(), pipeline.get()));
        return *this;
    }

    RenderPass& set_vertex_buffer_parcel(uint32_t slot, const Parcel& parcel) {
        detail::throw_on_result(goldy_scheme_render_pass_set_vertex_buffer_parcel(
            scheme_.ptr_.get(), slot, parcel.get()));
        return *this;
    }

    RenderPass& set_index_buffer(const Parcel& parcel, GoldyIndexFormat format) {
        detail::throw_on_result(goldy_scheme_render_pass_set_index_buffer(
            scheme_.ptr_.get(), parcel.get(), format));
        return *this;
    }

    RenderPass& draw(uint32_t first_vertex, uint32_t vertex_count,
                     uint32_t first_instance = 0, uint32_t instance_count = 1) {
        detail::throw_on_result(goldy_scheme_render_pass_draw(
            scheme_.ptr_.get(), first_vertex, vertex_count, first_instance, instance_count));
        return *this;
    }

    RenderPass& draw_indexed(uint32_t first_index, uint32_t index_count, int32_t base_vertex = 0,
                               uint32_t first_instance = 0, uint32_t instance_count = 1) {
        detail::throw_on_result(goldy_scheme_render_pass_draw_indexed(
            scheme_.ptr_.get(), first_index, index_count, base_vertex, first_instance, instance_count));
        return *this;
    }

    RenderPass& draw_fullscreen() {
        detail::throw_on_result(goldy_scheme_render_pass_draw_fullscreen(scheme_.ptr_.get()));
        return *this;
    }

    void finish() {
        if (!active_) {
            return;
        }
        detail::throw_on_result(goldy_scheme_render_pass_finish(scheme_.ptr_.get()));
        active_ = false;
    }

private:
    Scheme& scheme_;
    bool active_ = false;
};

inline Scheme::RenderPass Scheme::render_pass(const char* label, const SchemeRenderTargetLease& target) {
    return RenderPass(*this, label, target);
}


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

