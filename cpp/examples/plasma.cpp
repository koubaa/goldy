/**
 * Goldy C++ Example: Plasma
 *
 * Demonstrates bindless rendering with vertex-less fullscreen triangle:
 * - Using set_push_constants() to pass buffer indices to shaders
 * - Time-based animation with uniform buffer updates
 * - No vertex buffer needed - geometry generated in vertex shader
 *
 * This is the C++ equivalent of goldy/examples/plasma.rs
 */

#include <goldy.hpp>
#include <iostream>
#include <cstdint>
#include <array>
#include <fstream>
#include <sstream>
#include <cstring>

// Uniform buffer data (must match shader cbuffer layout)
struct Uniforms {
    float time;
};

// Load shader from file
std::string load_shader(const char* name) {
    // Shaders are in ../../shaders/ relative to this example
    std::string path = std::string("../../shaders/") + name;
    std::ifstream file(path);
    if (!file.is_open()) {
        // Try from build directory (shaders may be in different relative path)
        path = std::string("../../../shaders/") + name;
        file.open(path);
    }
    if (!file.is_open()) {
        throw std::runtime_error("Could not open shader file: " + std::string(name));
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

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
        std::cout << "Goldy C++ Plasma Example\n";
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

        // Create uniform buffer for time (Broadcast = all threads read same address)
        goldy::Buffer uniform_buffer(device, sizeof(Uniforms), goldy::DataAccess::Broadcast);
        std::cout << "Created uniform buffer: " << uniform_buffer.size() << " bytes\n";

        // Load and compile shader from shared shaders directory
        std::string shader_src = load_shader("plasma.slang");
        goldy::ShaderModule shader(device, shader_src.c_str());
        std::cout << "Compiled plasma shader (from shaders/plasma.slang)\n";

        // Create pipeline - vertex-less (no vertex buffer needed)
        GoldyRenderPipelineDesc pipeline_desc{};
        pipeline_desc.vertex_attributes = nullptr;
        pipeline_desc.vertex_attribute_count = 0;
        pipeline_desc.vertex_stride = 0;
        pipeline_desc.topology = GOLDY_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        pipeline_desc.target_format = GOLDY_TEXTURE_FORMAT_RGBA8_UNORM;
        pipeline_desc.depth_enabled = false;

        goldy::RenderPipeline pipeline(device, shader, shader, pipeline_desc);
        std::cout << "Created render pipeline (vertex-less)\n";

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
            // Pass buffer indices via push constants
            encoder.set_push_constants(uniform_buffer);
            // Vertex-less fullscreen triangle: 3 vertices, no vertex buffer
            encoder.draw(3);

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
