/**
 * Headless compute — Scheme compute dispatch + grant readback (no GLFW).
 *
 * Mirrors ffi-client/examples/compute_simple.rs.
 *
 * Build: cmake --build build --target compute_simple
 */

#include <goldy.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>

namespace {

constexpr const char* kComputeSrc = R"(
import goldy_exp;

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<float> data, ThreadId id) {
    uint idx = id.x;
    if (idx < 64u) {
        data[idx] = float(idx) * 2.0;
    }
}
)";

} // namespace

int main() {
    try {
        std::cout << "Goldy compute_simple (C++ / Scheme)\n";
        std::cout << "====================================\n";

        goldy::Instance instance;
        goldy::Device device = instance.request_adapter().request_device();

        std::array<float, 64> data{};
        goldy::RetainedPool pool(device);
        goldy::Buffer buffer = pool.acquire_buffer_with_data(
            std::span<const float>(data),
            goldy::BufferKind::Scattered);

        goldy::ShaderModule shader(device, kComputeSrc);
        goldy::ComputePipeline pipeline(device, shader);

        goldy::Context ctx(device);
        goldy::Scheme scheme(ctx);
        {
            auto node = scheme.compute_node("double", pipeline);
            node.with_buffer(buffer, goldy::NodeAccess::ReadWrite, goldy::ResourceAccess::Write);
            node.dispatch(1, 1, 1);
        }
        auto grant = scheme.grant_read(buffer);
        auto frame = scheme.submit();
        const auto bytes = grant.consume(frame);

        if (bytes.size() < 64 * sizeof(float)) {
            throw std::runtime_error("readback too small");
        }

        const auto* values = reinterpret_cast<const float*>(bytes.data());
        for (std::size_t i = 0; i < 64; ++i) {
            const float expected = static_cast<float>(i) * 2.0f;
            const float v = values[i];
            if (std::abs(v - expected) > 1e-4f) {
                throw std::runtime_error(
                    "index " + std::to_string(i) + ": expected " + std::to_string(expected) +
                    ", got " + std::to_string(v));
            }
        }

        std::cout << "Compute dispatch verified: data[i] == i * 2 for 64 elements.\n";
        return 0;
    } catch (const goldy::Exception& e) {
        std::cerr << "Goldy error: " << e.what() << '\n';
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
