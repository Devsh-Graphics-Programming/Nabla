import json
from conan import ConanFile
from conan.tools.cmake import cmake_layout, CMakeToolchain, CMakeDeps

class VeritasProject(ConanFile):
    settings = "os", "compiler", "build_type", "arch"

    default_options = {
        "libyuv/*:with_jpeg":False
    }

    def requirements(self):
        self.requires("libavif/1.4.1")

    def layout(self):
        cmake_layout(self)

    def generate(self):
        deps = CMakeDeps(self)
        deps.configuration_types = ["Debug", "Release", "RelWithDebInfo"]
        deps.build_context_activated = ["Release"]
        deps.generate()
        
        tc = CMakeToolchain(self)
        # Redirect Conan's preset output so CMake doesn't load it as the default User file
        tc.user_presets_path = ""
        tc.generate()