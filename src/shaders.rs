//! Slang shader source code - single source of truth for all shaders.
//!
//! All shaders are written in Slang and included at compile time from the
//! `shaders/` directory. Native examples use these directly with the Slang
//! compiler. Web demos expose these to JavaScript via wasm-bindgen, where
//! slang-wasm compiles them to WGSL at runtime.

/// Simple 2D vertex + fragment shader for colored vertices.
/// Used by: triangle, particles, starfield, bouncing_lines, spinning_cube, instancing, waveform
pub const VERTEX_COLOR_2D: &str = include_str!("../shaders/vertex_color_2d.slang");

/// Simple colored triangle shader (procedural vertices from vertex ID)
pub const TRIANGLE: &str = include_str!("../shaders/triangle.slang");

/// Digital clock shader (uses vertex coloring)
pub const DIGITAL_CLOCK: &str = include_str!("../shaders/digital_clock.slang");

/// Plasma effect shader (uses preprocessor-based platform selection)
pub const PLASMA: &str = include_str!("../shaders/plasma.slang");

/// Gradient effect shader
pub const GRADIENT: &str = include_str!("../shaders/gradient.slang");

/// Mandelbrot fractal shader
pub const MANDELBROT: &str = include_str!("../shaders/mandelbrot.slang");

/// Tunnel effect shader
pub const TUNNEL: &str = include_str!("../shaders/tunnel.slang");

/// Metaballs effect shader
pub const METABALLS: &str = include_str!("../shaders/metaballs.slang");

/// Checkerboard pattern shader
pub const CHECKERBOARD: &str = include_str!("../shaders/checkerboard.slang");

/// Starfield effect shader
pub const STARFIELD: &str = include_str!("../shaders/starfield.slang");

/// Particles (rain/snow) effect shader
pub const PARTICLES: &str = include_str!("../shaders/particles.slang");

/// Spinning cube wireframe shader
pub const SPINNING_CUBE: &str = include_str!("../shaders/spinning_cube.slang");

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify all shaders are non-empty and contain expected Slang syntax
    #[test]
    fn test_all_shaders_non_empty() {
        assert!(
            !VERTEX_COLOR_2D.is_empty(),
            "VERTEX_COLOR_2D shader is empty"
        );
        assert!(!TRIANGLE.is_empty(), "TRIANGLE shader is empty");
        assert!(!DIGITAL_CLOCK.is_empty(), "DIGITAL_CLOCK shader is empty");
        assert!(!PLASMA.is_empty(), "PLASMA shader is empty");
        assert!(!GRADIENT.is_empty(), "GRADIENT shader is empty");
        assert!(!MANDELBROT.is_empty(), "MANDELBROT shader is empty");
        assert!(!TUNNEL.is_empty(), "TUNNEL shader is empty");
        assert!(!METABALLS.is_empty(), "METABALLS shader is empty");
        assert!(!CHECKERBOARD.is_empty(), "CHECKERBOARD shader is empty");
        assert!(!STARFIELD.is_empty(), "STARFIELD shader is empty");
        assert!(!PARTICLES.is_empty(), "PARTICLES shader is empty");
        assert!(!SPINNING_CUBE.is_empty(), "SPINNING_CUBE shader is empty");
    }

    /// Verify shaders contain Slang shader entry point markers
    #[test]
    fn test_shaders_have_entry_points() {
        let shaders = [
            ("VERTEX_COLOR_2D", VERTEX_COLOR_2D),
            ("TRIANGLE", TRIANGLE),
            ("DIGITAL_CLOCK", DIGITAL_CLOCK),
            ("PLASMA", PLASMA),
            ("GRADIENT", GRADIENT),
            ("MANDELBROT", MANDELBROT),
            ("TUNNEL", TUNNEL),
            ("METABALLS", METABALLS),
            ("CHECKERBOARD", CHECKERBOARD),
            ("STARFIELD", STARFIELD),
            ("PARTICLES", PARTICLES),
            ("SPINNING_CUBE", SPINNING_CUBE),
        ];

        for (name, source) in shaders {
            assert!(
                source.contains("[shader(\"vertex\")]"),
                "{} missing vertex shader entry point",
                name
            );
            assert!(
                source.contains("[shader(\"fragment\")]"),
                "{} missing fragment shader entry point",
                name
            );
        }
    }

    /// Verify shaders contain vs_main and fs_main functions
    #[test]
    fn test_shaders_have_main_functions() {
        let shaders = [
            ("VERTEX_COLOR_2D", VERTEX_COLOR_2D),
            ("TRIANGLE", TRIANGLE),
            ("PLASMA", PLASMA),
        ];

        for (name, source) in shaders {
            assert!(
                source.contains("vs_main"),
                "{} missing vs_main function",
                name
            );
            assert!(
                source.contains("fs_main"),
                "{} missing fs_main function",
                name
            );
        }
    }

    /// Verify PLASMA structure
    #[test]
    fn test_plasma_structure() {
        // Plasma should use DescriptorHandle for DX12 and descriptor arrays for SPIRV/Metal
        assert!(
            PLASMA.contains("DescriptorHandle"),
            "PLASMA should use DescriptorHandle<T> for DX12"
        );
        assert!(
            PLASMA.contains("import goldy_exp"),
            "PLASMA should import goldy_exp module"
        );
    }

    /// Test that PLASMA compiles via Slang for all targets
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_plasma_compiles() {
        use crate::slang::{ShaderTarget, SlangCompiler};

        let compiler = SlangCompiler::new().expect("Failed to create Slang compiler");

        let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let shader_path = manifest_dir.join("shaders");
        let shader_path_str = shader_path.to_string_lossy();

        // Test SPIRV compilation (Vulkan) - needs __SPIRV__ define
        let spirv_defines = vec![("__SPIRV__", "1")];
        let result = compiler.compile_with_defines(
            PLASMA,
            ShaderTarget::Spirv,
            &[],
            &[&shader_path_str],
            &spirv_defines,
        );
        assert!(
            result.is_ok(),
            "PLASMA failed to compile for SPIRV: {:?}",
            result.err()
        );

        // Test DXIL compilation (DX12) - needs __DX12__ define
        // Only run on Windows since DXC compiler is not available on other platforms
        #[cfg(windows)]
        {
            let dxil_defines = vec![("__DX12__", "1")];
            let result = compiler.compile_with_defines(
                PLASMA,
                ShaderTarget::Dxil,
                &[],
                &[&shader_path_str],
                &dxil_defines,
            );
            assert!(
                result.is_ok(),
                "PLASMA failed to compile for DXIL: {:?}",
                result.err()
            );
        }

        // Test Metal compilation - needs __METAL__ define
        // Only run on macOS since Metal is Apple-only
        #[cfg(target_os = "macos")]
        {
            let metal_defines = vec![("__METAL__", "1")];
            let result = compiler.compile_with_defines(
                PLASMA,
                ShaderTarget::Metal,
                &[],
                &[&shader_path_str],
                &metal_defines,
            );
            assert!(
                result.is_ok(),
                "PLASMA failed to compile for Metal: {:?}",
                result.err()
            );
        }
    }
}
