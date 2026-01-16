//! Shader library system for reusable Slang modules.
//!
//! Shader libraries provide a way to package and distribute reusable Slang shader code.
//! Libraries are registered with a [`Device`](crate::Device) and are automatically available for
//! `import` statements in shaders.
//!
//! # Built-in Library
//!
//! Goldy ships with the `goldy` library, which is automatically registered:
//!
//! ```slang
//! import goldy;
//!
//! [shader("vertex")]
//! FullscreenVarying vs_main(FullscreenVertex input) {
//!     return vs_fullscreen(input);
//! }
//! ```
//!
//! # Custom Libraries
//!
//! You can create and register your own libraries:
//!
//! ```rust,ignore
//! use goldy::ShaderLibrary;
//!
//! let my_lib = ShaderLibrary::from_source("myutils", r#"
//!     module myutils;
//!     public float3 custom_effect(float2 uv) { return float3(uv, 0.5); }
//! "#);
//!
//! device.register_library(my_lib)?;
//! ```

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::Path;

/// A shader library containing reusable Slang modules.
///
/// Libraries are named collections of Slang source files that can be imported
/// by other shaders. The library name becomes the import name in Slang code.
///
/// # Example
///
/// ```rust
/// use goldy::ShaderLibrary;
///
/// // Create a simple single-module library
/// let lib = ShaderLibrary::from_source("mylib", r#"
///     module mylib;
///     public float3 my_func() { return float3(1, 0, 0); }
/// "#);
///
/// // The library can now be imported in shaders:
/// // import mylib;
/// // float3 color = my_func();
/// ```
#[derive(Debug, Clone)]
pub struct ShaderLibrary {
    name: String,
    /// Map of module path to source code
    /// The primary module uses the library name as key
    /// Sub-modules use paths like "libname/submodule"
    modules: HashMap<String, String>,
}

impl ShaderLibrary {
    /// Create a library from a single source string.
    ///
    /// This is the simplest way to create a library with a single module.
    /// The source should start with `module <name>;`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use goldy::ShaderLibrary;
    ///
    /// let lib = ShaderLibrary::from_source("effects", r#"
    ///     module effects;
    ///     public float3 glow(float intensity) { return float3(intensity, intensity, 0); }
    /// "#);
    /// ```
    pub fn from_source(name: &str, source: &str) -> Self {
        let mut modules = HashMap::new();
        modules.insert(name.to_string(), source.to_string());
        Self {
            name: name.to_string(),
            modules,
        }
    }

    /// Create a library from multiple embedded source strings.
    ///
    /// Use this for libraries with multiple modules (primary + sub-modules).
    /// The first entry should be the primary module with key equal to the library name.
    ///
    /// # Example
    ///
    /// ```rust
    /// use goldy::ShaderLibrary;
    ///
    /// let lib = ShaderLibrary::from_embedded("mylib", &[
    ///     ("mylib", "module mylib; __include \"mylib/utils\";"),
    ///     ("mylib/utils", "implementing mylib; public float helper() { return 1.0; }"),
    /// ]);
    /// ```
    pub fn from_embedded(name: &str, modules: &[(&str, &str)]) -> Self {
        let modules = modules
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        Self {
            name: name.to_string(),
            modules,
        }
    }

    /// Create a library from a directory on the filesystem.
    ///
    /// Scans the directory for `.slang` files and includes them as modules.
    /// The primary module should be `<name>.slang` in the directory.
    ///
    /// # Directory Structure
    ///
    /// ```text
    /// mylib/
    /// ├── mylib.slang        # Primary module: `module mylib;`
    /// └── mylib/
    ///     ├── utils.slang    # Sub-module: `implementing mylib;`
    ///     └── effects.slang  # Sub-module: `implementing mylib;`
    /// ```
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use goldy::ShaderLibrary;
    /// use std::path::Path;
    ///
    /// let lib = ShaderLibrary::from_directory("mylib", Path::new("shaders/mylib"))?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn from_directory(name: &str, path: &Path) -> Result<Self> {
        let mut modules = HashMap::new();

        // Read primary module
        let primary_path = path.join(format!("{}.slang", name));
        let primary_source = std::fs::read_to_string(&primary_path).with_context(|| {
            format!("Failed to read primary module: {}", primary_path.display())
        })?;
        modules.insert(name.to_string(), primary_source);

        // Read sub-modules from subdirectory
        let subdir = path.join(name);
        if subdir.is_dir() {
            Self::read_submodules(&mut modules, name, &subdir)?;
        }

        Ok(Self {
            name: name.to_string(),
            modules,
        })
    }

    fn read_submodules(
        modules: &mut HashMap<String, String>,
        prefix: &str,
        dir: &Path,
    ) -> Result<()> {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() && path.extension().is_some_and(|e| e == "slang") {
                let stem = path.file_stem().unwrap().to_string_lossy();
                let module_name = format!("{}/{}", prefix, stem);
                let source = std::fs::read_to_string(&path)
                    .with_context(|| format!("Failed to read module: {}", path.display()))?;
                modules.insert(module_name, source);
            } else if path.is_dir() {
                // Recursively read nested directories
                let dir_name = path.file_name().unwrap().to_string_lossy();
                let new_prefix = format!("{}/{}", prefix, dir_name);
                Self::read_submodules(modules, &new_prefix, &path)?;
            }
        }
        Ok(())
    }

    /// Get the library name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get all module sources in this library.
    pub fn modules(&self) -> &HashMap<String, String> {
        &self.modules
    }

    /// Get source for a specific module path.
    pub fn get_module(&self, path: &str) -> Option<&str> {
        self.modules.get(path).map(|s| s.as_str())
    }

    /// Check if this library contains a module.
    pub fn has_module(&self, path: &str) -> bool {
        self.modules.contains_key(path)
    }

    /// The built-in Goldy shader library (experimental).
    ///
    /// **⚠️ EXPERIMENTAL**: This library's API is unstable and may change
    /// significantly in future versions as we learn what abstractions work best.
    ///
    /// Provides common utilities for shader development:
    /// - Math functions: `hash`, `center_uv`, `scale_uv`, `PI`, `TAU`
    /// - Color utilities: `rainbow`, `palette`, `hsv_to_rgb`
    /// - Vertex formats: `FullscreenVertex`, `ColoredVertex`
    /// - Vertex shaders: `vs_fullscreen`, `vs_colored`
    ///
    /// # Example
    ///
    /// ```slang
    /// import goldy_exp;
    ///
    /// [shader("vertex")]
    /// FullscreenVarying vs_main(FullscreenVertex input) {
    ///     return vs_fullscreen(input);
    /// }
    ///
    /// [shader("fragment")]
    /// float4 fs_main(FullscreenVarying input) : SV_Target {
    ///     float2 uv = center_uv(input.uv);
    ///     return float4(rainbow(uv.x), 1.0);
    /// }
    /// ```
    pub fn goldy_experimental() -> Self {
        Self::from_embedded(
            "goldy_exp",
            &[
                ("goldy_exp", include_str!("../shaders/goldy_exp.slang")),
                (
                    "goldy_exp/math",
                    include_str!("../shaders/goldy_exp/math.slang"),
                ),
                (
                    "goldy_exp/color",
                    include_str!("../shaders/goldy_exp/color.slang"),
                ),
                (
                    "goldy_exp/vertex",
                    include_str!("../shaders/goldy_exp/vertex.slang"),
                ),
                (
                    "goldy_exp/bindless",
                    include_str!("../shaders/goldy_exp/bindless.slang"),
                ),
                (
                    "goldy_exp/bindless_indices",
                    include_str!("../shaders/goldy_exp/bindless_indices.slang"),
                ),
            ],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_source_creates_single_module() {
        let lib = ShaderLibrary::from_source("test", "module test; void foo() {}");

        assert_eq!(lib.name(), "test");
        assert_eq!(lib.modules().len(), 1);
        assert!(lib.has_module("test"));
        assert!(!lib.has_module("other"));
    }

    #[test]
    fn test_from_embedded_creates_multiple_modules() {
        let lib = ShaderLibrary::from_embedded(
            "mylib",
            &[
                ("mylib", "module mylib;"),
                ("mylib/sub", "implementing mylib;"),
            ],
        );

        assert_eq!(lib.name(), "mylib");
        assert_eq!(lib.modules().len(), 2);
        assert!(lib.has_module("mylib"));
        assert!(lib.has_module("mylib/sub"));
    }

    #[test]
    fn test_get_module_returns_source() {
        let lib = ShaderLibrary::from_source("test", "module test; float x = 1.0;");

        let source = lib.get_module("test").unwrap();
        assert!(source.contains("float x = 1.0"));
    }

    #[test]
    fn test_get_module_returns_none_for_missing() {
        let lib = ShaderLibrary::from_source("test", "module test;");

        assert!(lib.get_module("nonexistent").is_none());
    }

    #[test]
    fn test_goldy_library_has_expected_modules() {
        let lib = ShaderLibrary::goldy_experimental();

        assert_eq!(lib.name(), "goldy_exp");
        assert!(lib.has_module("goldy_exp"));
        assert!(lib.has_module("goldy_exp/math"));
        assert!(lib.has_module("goldy_exp/color"));
        assert!(lib.has_module("goldy_exp/vertex"));
        assert!(lib.has_module("goldy_exp/bindless"));
        assert!(lib.has_module("goldy_exp/bindless_indices"));
    }

    #[test]
    fn test_goldy_library_modules_are_valid() {
        let lib = ShaderLibrary::goldy_experimental();

        // Primary module should have module declaration
        let primary = lib.get_module("goldy_exp").unwrap();
        assert!(primary.contains("module goldy_exp;"));

        // Sub-modules should have implementing declaration
        let math = lib.get_module("goldy_exp/math").unwrap();
        assert!(math.contains("implementing goldy_exp;"));

        let color = lib.get_module("goldy_exp/color").unwrap();
        assert!(color.contains("implementing goldy_exp;"));

        let vertex = lib.get_module("goldy_exp/vertex").unwrap();
        assert!(vertex.contains("implementing goldy_exp;"));
    }
}
