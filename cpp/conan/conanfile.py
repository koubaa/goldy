from conan import ConanFile
from conan.tools.cmake import CMake, CMakeToolchain, cmake_layout
from conan.tools.files import copy, get, rmdir
from conan.tools.scm import Git
from conan.errors import ConanInvalidConfiguration
import os
import subprocess


class GoldyConan(ConanFile):
    name = "goldy"
    version = "0.1.0"
    license = "MIT"
    author = "KOB3 Contributors"
    url = "https://github.com/koubaa/goldy"
    homepage = "https://github.com/koubaa/goldy"
    description = "Modern GPU library with Slang shader support"
    topics = ("gpu", "graphics", "vulkan", "rendering", "slang")
    
    settings = "os", "compiler", "build_type", "arch"
    options = {
        "shared": [True, False],
    }
    default_options = {
        "shared": True,  # goldy-ffi is always a dynamic library
    }
    
    exports_sources = "cpp/*", "ffi/*", "src/*", "Cargo.toml", "Cargo.lock", "LICENSE"
    
    def validate(self):
        # Currently only Windows x64 is supported
        if self.settings.os != "Windows":
            raise ConanInvalidConfiguration(
                f"goldy currently only supports Windows, not {self.settings.os}"
            )
        if self.settings.arch != "x86_64":
            raise ConanInvalidConfiguration(
                f"goldy currently only supports x86_64, not {self.settings.arch}"
            )
        if not self.options.shared:
            raise ConanInvalidConfiguration(
                "goldy only supports shared library builds"
            )
    
    def layout(self):
        cmake_layout(self)
    
    def build_requirements(self):
        # Cargo is required to build the Rust FFI library
        # Users must have Rust installed: https://rustup.rs
        pass
    
    def source(self):
        # For conan-center, sources would be downloaded
        # For local development, sources are exported
        pass
    
    def generate(self):
        tc = CMakeToolchain(self)
        tc.variables["GOLDY_BUILD_FROM_SOURCE"] = True
        tc.variables["GOLDY_BUILD_EXAMPLES"] = False
        tc.generate()
    
    def _build_rust_ffi(self):
        """Build the goldy-ffi library using Cargo."""
        self.output.info("Building goldy-ffi with Cargo...")
        
        # Determine profile
        if self.settings.build_type == "Debug":
            profile_args = []
            target_subdir = "debug"
        else:
            profile_args = ["--release"]
            target_subdir = "release"
        
        # Run cargo build
        cargo_cmd = ["cargo", "build", "--package", "goldy-ffi"] + profile_args
        
        self.run(" ".join(cargo_cmd), cwd=self.source_folder)
        
        return os.path.join(self.source_folder, "target", target_subdir)
    
    def build(self):
        # First build the Rust FFI library
        self._build_rust_ffi()
        
        # CMake is only used if we want examples, skip for library-only build
        # The library is header-only C++ wrapper + prebuilt native lib
    
    def package(self):
        # Copy license
        copy(self, "LICENSE", src=self.source_folder,
             dst=os.path.join(self.package_folder, "licenses"))
        
        # Copy headers
        copy(self, "*.h", src=os.path.join(self.source_folder, "cpp", "include"),
             dst=os.path.join(self.package_folder, "include"))
        copy(self, "*.hpp", src=os.path.join(self.source_folder, "cpp", "include"),
             dst=os.path.join(self.package_folder, "include"))
        
        # Copy native library
        if self.settings.build_type == "Debug":
            target_subdir = "debug"
        else:
            target_subdir = "release"
        
        target_dir = os.path.join(self.source_folder, "target", target_subdir)
        
        if self.settings.os == "Windows":
            copy(self, "goldy_ffi.dll", src=target_dir,
                 dst=os.path.join(self.package_folder, "bin"))
            copy(self, "goldy_ffi.dll.lib", src=target_dir,
                 dst=os.path.join(self.package_folder, "lib"))
            # Rename the import library
            old_path = os.path.join(self.package_folder, "lib", "goldy_ffi.dll.lib")
            new_path = os.path.join(self.package_folder, "lib", "goldy_ffi.lib")
            if os.path.exists(old_path):
                os.rename(old_path, new_path)
        elif self.settings.os == "Linux":
            copy(self, "libgoldy_ffi.so", src=target_dir,
                 dst=os.path.join(self.package_folder, "lib"))
        elif self.settings.os == "Macos":
            copy(self, "libgoldy_ffi.dylib", src=target_dir,
                 dst=os.path.join(self.package_folder, "lib"))
    
    def package_info(self):
        self.cpp_info.set_property("cmake_file_name", "goldy")
        self.cpp_info.set_property("cmake_target_name", "goldy::goldy")
        
        # Header-only component
        self.cpp_info.components["goldy-headers"].includedirs = ["include"]
        
        # Native library component
        self.cpp_info.components["goldy-ffi"].libs = ["goldy_ffi"]
        self.cpp_info.components["goldy-ffi"].set_property("cmake_target_name", "goldy::ffi")
        
        # Main target depends on both
        self.cpp_info.components["goldy-headers"].requires = ["goldy-ffi"]
        
        # Runtime requirements
        self.cpp_info.set_property("cmake_find_mode", "config")
        
        if self.settings.os == "Windows":
            # DLL needs to be in PATH or next to executable
            self.runenv_info.append_path("PATH", os.path.join(self.package_folder, "bin"))

