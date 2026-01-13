/**
 * Goldy C++ Example: Triangle
 *
 * Demonstrates basic usage of the Goldy C++ API:
 * - Creating an instance and device
 * - Compiling a shader
 * - Creating buffers and pipelines
 * - Rendering to a target
 * - Reading back pixel data
 *
 * This is the C++ equivalent of goldy/examples/triangle.rs
 */

#include <goldy.hpp>
#include <iostream>
#include <cstdint>
#include <array>
#include <fstream>

// Vertex data: position (x, y) + color (r, g, b, a)
struct Vertex {
    float x, y;
    float r, g, b, a;
};

// Triangle shader in Slang
constexpr const char* TRIANGLE_SHADER = R"(
struct VertexInput {
    float2 position : POSITION;
    float4 color : COLOR;
};

struct VertexOutput {
    float4 position : SV_Position;
    float4 color : COLOR;
};

[shader("vertex")]
VertexOutput vs_main(VertexInput input) {
    VertexOutput output;
    output.position = float4(input.position, 0.0, 1.0);
    output.color = input.color;
    return output;
}

[shader("fragment")]
float4 fs_main(VertexOutput input) : SV_Target {
    return input.color;
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
        std::cout << "Goldy C++ Triangle Example\n";
        std::cout << "==========================\n\n";

        // Create instance and enumerate adapters
        goldy::Instance instance;
        std::cout << "Backend: ";
        switch (instance.backend_type()) {
            case GOLDY_BACKEND_TYPE_VULKAN: std::cout << "Vulkan\n"; break;
            case GOLDY_BACKEND_TYPE_DX12: std::cout << "DirectX 12\n"; break;
            case GOLDY_BACKEND_TYPE_METAL: std::cout << "Metal\n"; break;
            default: std::cout << "Unknown\n"; break;
        }

        auto adapters = instance.enumerate_adapters();
        std::cout << "\nAvailable adapters:\n";
        for (const auto& adapter : adapters) {
            std::cout << "  [" << adapter.id << "] " << adapter.name << "\n";
        }
        std::cout << "\n";

        // Create device (prefer discrete GPU)
        goldy::Device device = instance.create_device(GOLDY_DEVICE_TYPE_DISCRETE_GPU);
        std::cout << "Created device on adapter " << device.adapter_id() << "\n";
        std::cout << "Has goldy_exp library: " << (device.has_library("goldy_exp") ? "yes" : "no") << "\n\n";

        // Triangle vertices (clockwise)
        std::array<Vertex, 3> vertices = {{
            { 0.0f,  0.5f,  1.0f, 0.0f, 0.0f, 1.0f },  // Top - Red
            { 0.5f, -0.5f,  0.0f, 1.0f, 0.0f, 1.0f },  // Bottom right - Green
            {-0.5f, -0.5f,  0.0f, 0.0f, 1.0f, 1.0f },  // Bottom left - Blue
        }};

        // Create vertex buffer
        auto usage = goldy::BufferUsage::Vertex | goldy::BufferUsage::CopyDst;
        goldy::Buffer vertex_buffer(device,
            std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(vertices.data()),
                                     vertices.size() * sizeof(Vertex)),
            usage);
        std::cout << "Created vertex buffer: " << vertex_buffer.size() << " bytes\n";

        // Compile shader
        goldy::ShaderModule shader(device, TRIANGLE_SHADER);
        std::cout << "Compiled shader\n";

        // Create pipeline
        std::array<GoldyVertexAttribute, 2> attributes = {{
            { 0, GOLDY_VERTEX_FORMAT_FLOAT32X2, 0 },                    // position
            { 1, GOLDY_VERTEX_FORMAT_FLOAT32X4, sizeof(float) * 2 },    // color
        }};

        GoldyRenderPipelineDesc pipeline_desc{};
        pipeline_desc.vertex_attributes = attributes.data();
        pipeline_desc.vertex_attribute_count = static_cast<uint32_t>(attributes.size());
        pipeline_desc.vertex_stride = sizeof(Vertex);
        pipeline_desc.topology = GOLDY_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        pipeline_desc.target_format = GOLDY_TEXTURE_FORMAT_RGBA8_UNORM;
        pipeline_desc.depth_enabled = false;

        goldy::RenderPipeline pipeline(device, shader, shader, pipeline_desc);
        std::cout << "Created render pipeline\n";

        // Create render target
        constexpr uint32_t WIDTH = 800;
        constexpr uint32_t HEIGHT = 600;
        goldy::RenderTarget target(device, WIDTH, HEIGHT, GOLDY_TEXTURE_FORMAT_RGBA8_UNORM);
        std::cout << "Created render target: " << target.width() << "x" << target.height() << "\n\n";

        // Record and execute rendering commands
        std::cout << "Rendering...\n";
        {
            goldy::CommandEncoder encoder;
            encoder.clear(goldy::Color::cornflower_blue());
            encoder.set_pipeline(pipeline);
            encoder.set_vertex_buffer(0, vertex_buffer);
            encoder.draw(3);  // 3 vertices
            target.render(std::move(encoder));
        }

        // Read back and save
        std::cout << "Reading pixels...\n";
        auto pixels = target.read_to_cpu();
        std::cout << "Read " << pixels.size() << " bytes\n";

        // Save as PPM
        const char* output_file = "triangle.ppm";
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

