from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMakeConfigDeps, CMake, cmake_layout
from conan.tools.files import copy

class NablaConan(ConanFile):
    name = "Nabla"
    required_conan_version = ">=2.32"

    settings = "os", "arch", "compiler", "build_type"

    tool_requires = (
        "cmake/[>=3.31]",
    )

    def layout(self):
        cmake_layout(self)

    def requirements(self):
        self.requires("argparse/[>3.0]")
        self.requires("blake3/1.8.5", options={"shared": False})
        self.requires("boost/[>=1.91.0]", options={"shared": False, "without_test": True, "without_cobalt": True})
        self.requires("bzip2/1.0.8", options={"shared": False})
        self.requires("freetype/[>2.14.1]", options={"shared": False})
        self.requires("greg7mdp-gtl/1.2.0")
        self.requires("imath/[>3.2.1]", options={"shared": False})
        self.requires("libdeflate/[>=1.21]", options={"shared": False})
        self.requires("expat/[>=2.5.0]", options={"shared": False})
        self.requires("libjpeg-turbo/[>3.1.2]", options={"shared": False})
        self.requires("libpng/[>1.6.50]", options={"shared": False})
        self.requires("lz4/[>1.9.0]", options={"shared": False})
        self.requires("nlohmann_json/[>3.10.4]")
        self.requires("onedpl/2022.3.0")
        self.requires("onetbb/2023.1.0", options={"shared": False, "tbbmalloc": True, "tbbproxy": True})
        self.requires("openexr/[>3.4.5]", options={"shared": False})
        self.requires("portable-file-dialogs/[>=0.1.0]")
        self.requires("simdjson/[>1.0.0]", options={"shared": False})
        self.requires("zlib/[>=1.3.2]", options={"shared": False})

        
        # glm and dependencies
        self.requires("glm/[>=1.0.3]", options={"shared": False}, override=True)
        self.requires("gli/cci.20210515")

        # imgui libs all have related dependencies
        # TODO: use submodules for now, we need our recipe for imgui to set custom config in IMGUI_USER_CONFIG
        # self.requires("imgui/1.91.8", options={"shared": False, "enable_test_engine": True}, force=True)
        # self.requires("implot/0.17", options={"shared": False})
        # self.requires("imguizmo/cci.20231114", options={"shared": False})

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

    def generate(self):
        deps = CMakeConfigDeps(self)
        deps.generate()

        tc = CMakeToolchain(self)
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()