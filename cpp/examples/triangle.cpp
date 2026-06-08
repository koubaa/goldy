/**
 * Triangle example - render a colored triangle to an offscreen target via TaskGraph.
 *
 * Writes triangle.ppm for visual verification (no window required).
 */

#include <goldy.hpp>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <vector>

namespace {

struct Vertex {
    float position[2];
    float color[4];
};

void write_ppm(const char* path, uint32_t width, uint32_t height, const std::vector<uint8_t>& rgba) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error(std::string("Failed to open ") + path);
    }
    file << "P6\n" << width << " " << height << "\n255\n";
    for (size_t i = 0; i + 3 < rgba.size(); i += 4) {
        file.put(static_cast<char>(rgba[i]));
        file.put(static_cast<char>(rgba[i + 1]));
        file.put(static_cast<char>(rgba[i + 2]));
    }
}

} // namespace

int main() {
    try {
        std::cout << "Goldy C++ triangle example (TaskGraph)\n";

        goldy::Instance instance;
        auto adapters = instance.enumerate_adapters();
        if (adapters.empty()) {
            std::cerr << "No GPU adapters found\n";
            return 1;
        }

        goldy::Device device = instance.create_device_for_adapter(adapters[0].id);
        std::cout << "Created device on adapter: " << adapters[0].name << "\n";

        constexpr uint32_t WIDTH = 800;
        constexpr uint32_t HEIGHT = 600;

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
        pipeline_desc.target_format = GOLDY_TEXTURE_FORMAT_RGBA8_UNORM;
        pipeline_desc.depth_enabled = false;

        goldy::RenderPipeline pipeline(device, shader, shader, pipeline_desc);
        goldy::RenderTarget target(device, WIDTH, HEIGHT, GOLDY_TEXTURE_FORMAT_RGBA8_UNORM);
        std::cout << "Created render target: " << target.width() << "x" << target.height() << "\n";

        goldy::TaskGraph graph;
        {
            auto pass = graph.render_pass("triangle", target);
            pass.bind_buffer(vertex_buffer, goldy::NodeAccess::Read)
                .clear(goldy::Color::cornflower_blue())
                .set_pipeline(pipeline)
                .set_vertex_buffer(0, vertex_buffer)
                .draw(0, 3);
        }

        std::cout << "Dispatching task graph...\n";
        graph.dispatch(device);

        std::cout << "Reading pixels...\n";
        auto pixels = target.read_to_cpu();
        std::cout << "Read " << pixels.size() << " bytes\n";

        constexpr const char* output_file = "triangle.ppm";
        write_ppm(output_file, WIDTH, HEIGHT, pixels);
        std::cout << "Saved to " << output_file << "\n";
        std::cout << "Done!\n";
        return 0;
    } catch (const goldy::Exception& e) {
        std::cerr << "Goldy error: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
