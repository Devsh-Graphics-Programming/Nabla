# Copyright (c) 2019 DevSH Graphics Programming Sp. z O.O.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

add_subdirectory(01.HelloWorld EXCLUDE_FROM_ALL)
add_subdirectory(03.GPU_Mesh EXCLUDE_FROM_ALL)
add_subdirectory(04.BuiltinNodes EXCLUDE_FROM_ALL)
add_subdirectory(05.SpherePoints EXCLUDE_FROM_ALL)
add_subdirectory(06.MeshLoaders EXCLUDE_FROM_ALL)
add_subdirectory(07.HardwareSkinning EXCLUDE_FROM_ALL)
add_subdirectory(08.HardwareInstancing EXCLUDE_FROM_ALL)
add_subdirectory(09.ColorSpaceTest EXCLUDE_FROM_ALL)
add_subdirectory(10.AllocPerfTest EXCLUDE_FROM_ALL)
add_subdirectory(11.RayCastCollision EXCLUDE_FROM_ALL)
add_subdirectory(12.Splines EXCLUDE_FROM_ALL)
if (IRR_BUILD_CEGUI)
add_subdirectory(13.BRDF_Explorer EXCLUDE_FROM_ALL)
endif()
add_subdirectory(14.MSAA EXCLUDE_FROM_ALL)
add_subdirectory(15.LayeredCubeMapShadows EXCLUDE_FROM_ALL)
add_subdirectory(16.OrderIndependentTransparencyZoo EXCLUDE_FROM_ALL)
if (IRR_BUILD_BULLET)
add_subdirectory(17.SimpleBulletIntegration EXCLUDE_FROM_ALL)
endif()
add_subdirectory(19.SIMDmatrixMultiplication EXCLUDE_FROM_ALL)
add_subdirectory(23.Autoexposure EXCLUDE_FROM_ALL)
add_subdirectory(24.BufferSubDataAlignTest EXCLUDE_FROM_ALL)
add_subdirectory(25.Blur EXCLUDE_FROM_ALL)
add_subdirectory(26.MultidrawIndirectVSCPUCull EXCLUDE_FROM_ALL)
add_subdirectory(27.PointCloud EXCLUDE_FROM_ALL)
add_subdirectory(28.UBOBenchmark EXCLUDE_FROM_ALL)
add_subdirectory(29.SSBOAtomicsBenchmark EXCLUDE_FROM_ALL)
add_subdirectory(30.UBOBenchmarkV2 EXCLUDE_FROM_ALL)
add_subdirectory(31.SkinningDataBenchmark EXCLUDE_FROM_ALL)
add_subdirectory(32.MultiThreadedRefCounting EXCLUDE_FROM_ALL)
add_subdirectory(33.Draw3DLine EXCLUDE_FROM_ALL)
add_subdirectory(34.AddressAllocatorTraitsTest EXCLUDE_FROM_ALL)
