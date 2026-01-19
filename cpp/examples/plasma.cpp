/**
 * Goldy C++ Example: Plasma (Fully Bindless)
 *
 * Demonstrates fully BINDLESS rendering:
 * - No BindGroup/BindGroupLayout needed
 * - Using set_push_constants() to pass buffer indices to shaders
 * - Time-based animation with uniform buffer updates
 *
 * This is the C++ equivalent of goldy/examples/plasma.rs
 */

#include <goldy.hpp>
#include <iostream>
#include <cstdint>
#include <array>
#include <fstream>
#include <cstring>

// Vertex data: position (x, y) + UV (u, v)
struct Vertex2DUv {
    float x, y;
    float u, v;
};

// Uniform buffer data (must match shader cbuffer layout)
struct Uniforms {
    float time;
};

// Plasma shader using bindless resource access
constexpr const char* PLASMA_SHADER = R"(
import goldy_exp;

// Uniform structure  
struct TimeUniforms {
    float time;
};

#ifdef __BINDLESS__

#if defined(__METAL__)
// Metal: Use ParameterBlock for argument buffer support
struct PlasmaResources {
    ConstantBuffer<TimeUniforms> uniforms;
};
ParameterBlock<PlasmaResources> gResources;
#define TIME gResources.uniforms.time

#elif defined(__SPIRV__)
// Vulkan: Use push constants for indices + descriptor array
import goldy_exp.bindless_indices;

// Global descriptor array of uniform buffers
[[vk::binding(1, 0)]] ConstantBuffer<TimeUniforms> g_UniformBuffers[];
#define TIME g_UniformBuffers[getBindlessIndex(0)].time

#elif defined(__HLSL__) || defined(__DX12__)
// DX12: Bindless via root constants + ResourceDescriptorHeap
cbuffer BindlessIndices : register(b0) {
    uint uniformsIndex;
};
#define TIME (*DescriptorHandle<ConstantBuffer<TimeUniforms>>(uint2(uniformsIndex, 0))).time

#endif

#else
// Traditional binding mode
[[vk::binding(0, 0)]]
ConstantBuffer<TimeUniforms> uniforms;
#define TIME uniforms.time
#endif

[shader("vertex")]
FullscreenVarying vs_main(FullscreenVertex input) {
    return vs_fullscreen(input);
}

[shader("fragment")]
float4 fs_main(FullscreenVarying input) : SV_Target {
    float2 uv = scale_uv(input.uv, 4.0);
    float t = TIME;
    
    // Classic plasma formula
    float v = sin(uv.x + t);
    v += sin(uv.y + t);
    v += sin(uv.x + uv.y + t);
    
    float cx = uv.x + 0.5 * sin(t / 3.0);
    float cy = uv.y + 0.5 * cos(t / 2.0);
    v += sin(sqrt(cx * cx + cy * cy + 1.0) + t);
    
    v = v / 2.0;
    
    // Use rainbow palette from goldy module
    return float4(rainbow(v), 1.0);
}
)";

// Simple PPM image writer (no external dependencies)
void write_ppm(const char* filename, uint32_t width, uint32_t height,
               const std::vector<uint8_t>& pixels) {
    std::ofstream file(filename, std::ios::binary);
    file << "P6\n" << width << " " << height << "\n255\n";
    
    // RGBA to RGB conversion
    for (size_t i = 0; i < pixels.size(); i += 4) {
        file.put(static_cast<char>(pixels[i]));     // R
        file.put(static_cast<char>(pixels[i + 1])); // G
        file.put(static_cast<char>(pixels[i + 2])); // B
    }
}

int main() {
    try {
        std::cout << "Goldy C++ Plasma Example (Fully Bindless)\n";
        std::cout << "==========================================\n\n";

        // Create instance and enumerate adapters
        goldy::Instance instance;
        std::cout << "Backend: ";
        switch (instance.backend_type()) {
            case GOLDY_BACKEND_TYPE_VULKAN: std::cout << "Vulkan\n"; break;
            case GOLDY_BACKEND_TYPE_DX12: std::cout << "DirectX 12\n"; break;
            case GOLDY_BACKEND_TYPE_METAL: std::cout << "Metal\n"; break;
            default: std::cout << "Unknown\n"; break;
        }
        std::cout << "\n";

        // Create device (prefer discrete GPU)
        goldy::Device device = instance.create_device(GOLDY_DEVICE_TYPE_DISCRETE_GPU);
        std::cout << "Created device on adapter " << device.adapter_id() << "\n";
        std::cout << "Has goldy_exp library: " << (device.has_library("goldy_exp") ? "yes" : "no") << "\n\n";

        // Fullscreen quad vertices (position + uv)
        std::array<Vertex2DUv, 6> vertices = {{
            // Triangle 1
            {-1.0f, -1.0f,  0.0f, 1.0f},  // Bottom-left
            { 1.0f, -1.0f,  1.0f, 1.0f},  // Bottom-right
            { 1.0f,  1.0f,  1.0f, 0.0f},  // Top-right
            // Triangle 2
            {-1.0f, -1.0f,  0.0f, 1.0f},  // Bottom-left
            { 1.0f,  1.0f,  1.0f, 0.0f},  // Top-right
            {-1.0f,  1.0f,  0.0f, 0.0f},  // Top-left
        }};

        // Create vertex buffer
        auto vertex_usage = goldy::BufferUsage::Vertex | goldy::BufferUsage::CopyDst;
        goldy::Buffer vertex_buffer(device,
            std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(vertices.data()),
                                     vertices.size() * sizeof(Vertex2DUv)),
            vertex_usage);
        std::cout << "Created vertex buffer: " << vertex_buffer.size() << " bytes\n";

        // Create uniform buffer for time
        auto uniform_usage = goldy::BufferUsage::Uniform | goldy::BufferUsage::CopyDst;
        goldy::Buffer uniform_buffer(device, sizeof(Uniforms), uniform_usage);
        std::cout << "Created uniform buffer: " << uniform_buffer.size() << " bytes\n";

        // Compile shader
        goldy::ShaderModule shader(device, PLASMA_SHADER);
        std::cout << "Compiled plasma shader (bindless)\n";

        // Create pipeline WITHOUT bind group layouts - fully bindless!
        std::array<GoldyVertexAttribute, 2> attributes = {{
            { 0, GOLDY_VERTEX_FORMAT_FLOAT32X2, 0 },                    // position
            { 1, GOLDY_VERTEX_FORMAT_FLOAT32X2, sizeof(float) * 2 },    // uv
        }};

        GoldyRenderPipelineDesc pipeline_desc{};
        pipeline_desc.vertex_attributes = attributes.data();
        pipeline_desc.vertex_attribute_count = static_cast<uint32_t>(attributes.size());
        pipeline_desc.vertex_stride = sizeof(Vertex2DUv);
        pipeline_desc.topology = GOLDY_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        pipeline_desc.target_format = GOLDY_TEXTURE_FORMAT_RGBA8_UNORM;
        pipeline_desc.depth_enabled = false;
        // No bind_group_layouts needed for bindless!
        pipeline_desc.bind_group_layouts = nullptr;
        pipeline_desc.bind_group_layout_count = 0;

        goldy::RenderPipeline pipeline(device, shader, shader, pipeline_desc);
        std::cout << "Created render pipeline (bindless - no bind groups)\n";

        // Create render target
        constexpr uint32_t WIDTH = 800;
        constexpr uint32_t HEIGHT = 600;
        goldy::RenderTarget target(device, WIDTH, HEIGHT, GOLDY_TEXTURE_FORMAT_RGBA8_UNORM);
        std::cout << "Created render target: " << target.width() << "x" << target.height() << "\n\n";

        // Render multiple frames to show animation
        constexpr int NUM_FRAMES = 10;
        std::cout << "Rendering " << NUM_FRAMES << " frames...\n";

        for (int frame_idx = 0; frame_idx < NUM_FRAMES; ++frame_idx) {
            // Update time uniform
            Uniforms uniforms;
            uniforms.time = frame_idx * 0.1f;  // Simulate time progression
            uniform_buffer.write(0, std::span<const uint8_t>(
                reinterpret_cast<const uint8_t*>(&uniforms), sizeof(uniforms)));

            // Render
            goldy::CommandEncoder encoder;
            encoder.clear(goldy::Color::black());
            encoder.set_pipeline(pipeline);
            // BINDLESS: Pass buffer indices via push constants instead of bind groups!
            encoder.set_push_constants(uniform_buffer);
            encoder.set_vertex_buffer(0, vertex_buffer);
            encoder.draw(6);  // 6 vertices = 2 triangles = fullscreen quad

            target.render(std::move(encoder));

            if (frame_idx % 3 == 0) {
                std::cout << "  Frame " << frame_idx << ": time=" << uniforms.time << "\n";
            }
        }

        // Read back and save final frame
        std::cout << "\nReading pixels...\n";
        auto pixels = target.read_to_cpu();
        std::cout << "Read " << pixels.size() << " bytes\n";

        // Save as PPM
        const char* output_file = "plasma.ppm";
        write_ppm(output_file, WIDTH, HEIGHT, pixels);
        std::cout << "Saved to " << output_file << "\n";

        std::cout << "\nDone!\n";
        return 0;

    } catch (const goldy::Exception& e) {
        std::cerr << "Goldy error: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
