from conan import ConanFile
from conan.tools.files import copy, get, download
from conan.errors import ConanInvalidConfiguration
import os


class GoldyConan(ConanFile):
    name = "goldy"
    version = "0.1.0"
    license = "MIT"
    author = "KOB3 Contributors"
    url = "https://github.com/koubaa/goldy"
    homepage = "https://github.com/koubaa/goldy"
    description = "Modern GPU library with Slang shader support"
    topics = ("gpu", "graphics", "vulkan", "rendering", "slang")
    
    # Architecture:
    #   - goldy.hpp: Header-only C++ wrapper (built from source)
    #   - goldy_ffi: Pre-built native library with stable C ABI
    #
    # This is the standard approach for C++ packages wrapping non-C++ libraries
    # (similar to CUDA, Vulkan SDK, Intel MKL, etc.)
    
    settings = "os", "compiler", "build_type", "arch"
    options = {
        "shared": [True, False],
    }
    default_options = {
        "shared": True,  # goldy-ffi is always a dynamic library
    }
    
    # Pre-built binary URLs and SHA256 hashes per platform
    _binary_info = {
        ("Windows", "x86_64"): {
            "url": "https://github.com/koubaa/goldy/releases/download/v0.1.0/goldy_ffi-windows-x64.zip",
            "sha256": "7abd86b278756c8c73171ef2df4f248667e0fbcaab5f23dc2521ca569bead7a9",
        },
        ("Linux", "x86_64"): {
            "url": "https://github.com/koubaa/goldy/releases/download/v0.1.0/goldy_ffi-linux-x64.tar.gz",
            "sha256": "ccbc4152b4867c3bc5261ae56a046b087d76bf51063424799365711d52f49c01",
        },
        ("Macos", "x86_64"): {
            "url": "https://github.com/koubaa/goldy/releases/download/v0.1.0/goldy_ffi-macos-x64.tar.gz",
            "sha256": "632a12c2daf6cceb2cb4b5306e049abd613f6a309d62a329421acaa34070d7ce",
        },
        ("Macos", "armv8"): {
            "url": "https://github.com/koubaa/goldy/releases/download/v0.1.0/goldy_ffi-macos-arm64.tar.gz",
            "sha256": "53bdf0a288f39f6044d18851d9c41afdab35260929f64665cacac0630003aa34",
        },
    }
    
    def validate(self):
        key = (str(self.settings.os), str(self.settings.arch))
        if key not in self._binary_info:
            raise ConanInvalidConfiguration(
                f"goldy does not support {self.settings.os} {self.settings.arch}. "
                f"Supported: Windows x86_64, Linux x86_64, macOS x86_64/arm64"
            )
        if not self.options.shared:
            raise ConanInvalidConfiguration(
                "goldy only supports shared library builds"
            )
    
    def source(self):
        # Download source for headers and license
        get(self, 
            url=f"https://github.com/koubaa/goldy/archive/refs/tags/v{self.version}.tar.gz",
            strip_root=True)
    
    def build(self):
        # Download pre-built native library
        key = (str(self.settings.os), str(self.settings.arch))
        binary_info = self._binary_info[key]
        
        self.output.info(f"Downloading pre-built goldy_ffi for {key}...")
        
        # Determine filename from URL
        filename = binary_info["url"].split("/")[-1]
        
        download(self, 
                 url=binary_info["url"], 
                 filename=filename)
        
        # Extract the archive
        if filename.endswith(".zip"):
            import zipfile
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall("binary")
        else:
            import tarfile
            with tarfile.open(filename, 'r:gz') as tar_ref:
                tar_ref.extractall("binary")
    
    def package(self):
        # Copy license
        copy(self, "LICENSE", src=self.source_folder,
             dst=os.path.join(self.package_folder, "licenses"))
        
        # Copy headers from source
        copy(self, "*.h", src=os.path.join(self.source_folder, "cpp", "include"),
             dst=os.path.join(self.package_folder, "include"))
        copy(self, "*.hpp", src=os.path.join(self.source_folder, "cpp", "include"),
             dst=os.path.join(self.package_folder, "include"))
        
        # Copy native library from pre-built binary
        binary_dir = os.path.join(self.build_folder, "binary")
        
        if self.settings.os == "Windows":
            copy(self, "goldy_ffi.dll", src=os.path.join(binary_dir, "lib"),
                 dst=os.path.join(self.package_folder, "bin"))
            copy(self, "goldy_ffi.dll.lib", src=os.path.join(binary_dir, "lib"),
                 dst=os.path.join(self.package_folder, "lib"))
            # Rename the import library
            old_path = os.path.join(self.package_folder, "lib", "goldy_ffi.dll.lib")
            new_path = os.path.join(self.package_folder, "lib", "goldy_ffi.lib")
            if os.path.exists(old_path):
                os.rename(old_path, new_path)
        elif self.settings.os == "Linux":
            copy(self, "libgoldy_ffi.so", src=os.path.join(binary_dir, "lib"),
                 dst=os.path.join(self.package_folder, "lib"))
        elif self.settings.os == "Macos":
            copy(self, "libgoldy_ffi.dylib", src=os.path.join(binary_dir, "lib"),
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
