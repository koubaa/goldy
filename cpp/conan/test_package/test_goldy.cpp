/**
 * Conan package test for Goldy
 *
 * Verifies that the package can be linked and basic functionality works.
 */

#include <goldy.hpp>
#include <iostream>

int main() {
    try {
        std::cout << "Testing Goldy C++ package...\n";

        // Create instance - this verifies the library loads correctly
        goldy::Instance instance;
        std::cout << "  Created instance: OK\n";

        // Check backend
        auto backend = instance.backend_type();
        std::cout << "  Backend type: ";
        switch (backend) {
            case GOLDY_BACKEND_TYPE_VULKAN: std::cout << "Vulkan"; break;
            case GOLDY_BACKEND_TYPE_DX12: std::cout << "DirectX 12"; break;
            case GOLDY_BACKEND_TYPE_METAL: std::cout << "Metal"; break;
            default: std::cout << "Unknown"; break;
        }
        std::cout << " OK\n";

        // Enumerate adapters
        auto adapters = instance.enumerate_adapters();
        std::cout << "  Found " << adapters.size() << " adapter(s): OK\n";

        if (adapters.empty()) {
            std::cout << "  Warning: No GPU adapters found\n";
        } else {
            // Try to create a device
            goldy::Device device = instance.create_device(GOLDY_DEVICE_TYPE_DISCRETE_GPU);
            std::cout << "  Created device: OK\n";
            std::cout << "  Device is valid: " << (device.is_valid() ? "yes" : "no") << "\n";
        }

        std::cout << "\nGoldy package test PASSED\n";
        return 0;

    } catch (const goldy::Exception& e) {
        std::cerr << "Goldy error: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}

