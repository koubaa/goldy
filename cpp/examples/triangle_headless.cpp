/**
 * Headless triangle — Scheme render pass + MemoryExchange withdraw (no GLFW).
 *
 * Mirrors python/examples/triangle_headless.py and ffi-client triangle_headless.
 * Used on headless Linux CI (lavapipe container has no Wayland display).
 *
 * Build: cmake --build build --target triangle_headless
 */

#include <goldy.hpp>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {

struct Vertex {
    float position[2];
    float color[4];
};

void write_ppm(const char* path, const std::vector<uint8_t>& rgba, uint32_t width, uint32_t height) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error(std::string("failed to open ") + path + " for writing");
    }
    out << "P6\n" << width << ' ' << height << "\n255\n";
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            const size_t i = (static_cast<size_t>(y) * width + x) * 4;
            out.put(static_cast<char>(rgba[i + 0]));
            out.put(static_cast<char>(rgba[i + 1]));
            out.put(static_cast<char>(rgba[i + 2]));
        }
    }
}

GoldyTextureFlags copy_readback_flags() {
    GoldyTextureFlags flags{};
    flags._0 = (1 << 0) | (1 << 1);
    return flags;
}

} // namespace

int main() {
    try {
        std::cout << "Goldy Triangle Headless (C++ / Scheme)\n";
        std::cout << "=====================================\n";

        goldy::Instance instance;
        goldy::Device device = instance.request_adapter().request_device();
        goldy::Context ctx(device);

        const Vertex vertices[] = {
            {{0.0f, -0.5f}, {1.0f, 0.0f, 0.0f, 1.0f}},
            {{-0.5f, 0.5f}, {0.0f, 1.0f, 0.0f, 1.0f}},
            {{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f, 1.0f}},
        };

        goldy::RetainedPool pool(device);
        goldy::Buffer vertex_buffer = pool.acquire_buffer_with_data(
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

        constexpr uint32_t kWidth = 64;
        constexpr uint32_t kHeight = 64;
        goldy::Texture readback = pool.acquire_texture(
            kWidth, kHeight, GOLDY_TEXTURE_FORMAT_RGBA8_UNORM,
            GOLDY_TEXTURE_KIND_DIRECT, copy_readback_flags());

        goldy::Scheme scheme(ctx);
        goldy::SchemeRenderTargetLease rt = scheme.lease_render_target(
            kWidth, kHeight, GOLDY_TEXTURE_FORMAT_RGBA8_UNORM);
        {
            auto pass = scheme.render_pass("triangle", rt, goldy::TargetLoad::clear(goldy::Color::black()));
            pass.with_field(vertex_buffer, 0, goldy::NodeAccess::Read)
                .set_pipeline(pipeline)
                .set_vertex_buffer(0, vertex_buffer)
                .draw(0, 3);
        }

        scheme.copy_to_texture(rt, readback);
        goldy::MemoryExchange memory(ctx);
        goldy::WithdrawTransaction withdraw = memory.bind_withdraw_texture(scheme, readback);
        goldy::SchemeSubmission submission = scheme.submit();
        goldy::WithdrawBytes bytes = withdraw.claim(submission).consume();
        const auto pixels = bytes.to_vector();

        const bool has_lit_pixel = std::any_of(pixels.begin(), pixels.end(), [](uint8_t b) { return b > 0; });
        if (!has_lit_pixel) {
            throw std::runtime_error("readback should contain lit triangle pixels");
        }

        write_ppm("triangle.ppm", pixels, kWidth, kHeight);
        std::cout << "Wrote triangle.ppm (" << pixels.size() << " bytes readback)\n";
        return 0;
    } catch (const goldy::Exception& e) {
        std::cerr << "Goldy error: " << e.what() << '\n';
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
