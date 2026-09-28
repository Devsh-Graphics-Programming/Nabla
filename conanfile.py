from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMakeConfigDeps, CMake, cmake_layout
from conan.tools.files import copy

class NablaConan(ConanFile):
    name = "Nabla"
    required_conan_version = ">=2.32"

    settings = "os", "arch", "compiler", "build_type"
    generators = "CMakeConfigDeps", "CMakeToolchain"

    # What this package depends on
    requires = (
        "zlib/[>=1.3.2]",
        "bzip2/1.0.8",
        "lz4/[>1.9.0]",
        "libpng/[>1.6.50]",
        "libjpeg-turbo/[>3.1.2]",
        "freetype/[>2.14.1]",
        "openexr/[>3.4.5]",
        "imath/[>3.2.1]",
        "nlohmann_json/[>3.10.4]",
        "simdjson/[>1.0.0]",

        "onetbb/2023.1.0",
        "onedpl/2022.3.0",
        
        "argparse/[>3.0]",
        "expat/[>=2.5.0]",
        "blake3/1.8.5",
        "libdeflate/[>=1.21]",
        "portable-file-dialogs/[>=0.1.0]",
        "greg7mdp-gtl/1.2.0",
    )

    # What tools are needed to build this package
    tool_requires = (
        "cmake/[>=3.31]",
    )

    # Default options for this package
    default_options = {
        "zlib/*:shared": False,
        "bzip2/*:shared": False,
        "lz4/*:shared": False,
        "libpng/*:shared": False,
        "libjpeg-turbo/*:shared": False,
        "freetype/*:shared": False,
        "openexr/*:shared": False,
        "imath/*:shared": False,
        "simdjson/*:shared": False,
        "onetbb/*:shared": False,
        "onetbb/*:tbbmalloc": True,
        "onetbb/*:tbbproxy": True,
        "expat/*:shared": False,
        "blake3/*:shared": False,
        "libdeflate/*:shared": False,
    }

    def layout(self):
        cmake_layout(self)

    def requirements(self):
        self.requires("boost/[>=1.91.0]", options={"shared": False, "without_test": True, "without_cobalt": True})
        
        # glm and dependencies
        self.requires("glm/[>=1.0.3]", options={"shared": False}, override=True)
        self.requires("gli/cci.20210515")

        # imgui libs all have related dependencies
        self.requires("imgui/1.91.4", options={"shared": False}, force=True)
        self.requires("implot/0.17", options={"shared": False})
        self.requires("imguizmo/cci.20231114", options={"shared": False})

        # vulkan libs all have related dependencies
        self.requires("vulkan-headers/1.4.357.0")
        self.requires("volk/1.4.357.0")
        self.requires("glslang/1.4.357.0", options={"shared": False})
        self.requires("shaderc/2026.4", options={"shared": False})
        self.requires("spirv-tools/1.4.357.0", options={"shared": False})
        self.requires("spirv-headers/1.4.357.0")
        self.requires("spirv-cross/1.4.357.0", options={"shared": False, "glsl": True, "reflect": True})

    def configure(self):
        # force libtiff to build against libjpeg-turbo instead of standard libjpeg (as a dependency of openexr)
        self.options["libtiff"].jpeg = "libjpeg-turbo"

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()