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
add_subdirectory(02.ComputeShader EXCLUDE_FROM_ALL)
add_subdirectory(03.GPU_Mesh EXCLUDE_FROM_ALL)
# free space
add_subdirectory(05.IrrlichtBaWTutorialExample EXCLUDE_FROM_ALL)
add_subdirectory(06.MeshLoaders EXCLUDE_FROM_ALL)
#add_subdirectory(07.HardwareSkinning EXCLUDE_FROM_ALL)
#add_subdirectory(08.HardwareInstancing EXCLUDE_FROM_ALL)
add_subdirectory(09.ColorSpaceTest EXCLUDE_FROM_ALL)
#add_subdirectory(10.AllocPerfTest EXCLUDE_FROM_ALL)
#add_subdirectory(11.RayCastCollision EXCLUDE_FROM_ALL)
#add_subdirectory(12.Splines EXCLUDE_FROM_ALL)
if (IRR_BUILD_CEGUI)
	#add_subdirectory(13.BRDF_Explorer EXCLUDE_FROM_ALL)
endif()
#add_subdirectory(14.MSAA EXCLUDE_FROM_ALL) # TODO: Ressurect with Sponza or something?
#add_subdirectory(15.LayeredCubeMapShadows EXCLUDE_FROM_ALL) # TODO: Ressurect with Sponza or something?
#add_subdirectory(16.OrderIndependentTransparencyZoo EXCLUDE_FROM_ALL)
if (IRR_BUILD_BULLET)
	#add_subdirectory(17.SimpleBulletIntegration EXCLUDE_FROM_ALL) # Needs example 08 to come back first
endif()
if (IRR_BUILD_MITSUBA_LOADER)
	add_subdirectory(18.MitsubaLoader EXCLUDE_FROM_ALL)
endif()
add_subdirectory(19.SIMDmatrixMultiplication EXCLUDE_FROM_ALL)
add_subdirectory(20.Megatexture EXCLUDE_FROM_ALL)
add_subdirectory(21.DynamicTextureIndexing EXCLUDE_FROM_ALL) # TODO: revive
if (IRR_BUILD_MITSUBA_LOADER AND IRR_BUILD_RADEON_RAYS)
	#add_subdirectory(22.RaytracedAO EXCLUDE_FROM_ALL)
endif()
add_subdirectory(23.Autoexposure EXCLUDE_FROM_ALL)
#add_subdirectory(24.BufferSubDataAlignTest EXCLUDE_FROM_ALL) # TODO: @Anastazluk or @Hazardu
#add_subdirectory(25.Blur EXCLUDE_FROM_ALL)
add_subdirectory(26.MultidrawIndirectVSCPUCull EXCLUDE_FROM_ALL)
#add_subdirectory(27.PointCloud EXCLUDE_FROM_ALL)
if (IRR_BUILD_MITSUBA_LOADER AND IRR_BUILD_OPTIX)
	#add_subdirectory(28.OptiXPathTracing EXCLUDE_FROM_ALL)
endif()
add_subdirectory(29.SpecializationConstants EXCLUDE_FROM_ALL)
#add_subdirectory(31.SkinningDataBenchmark EXCLUDE_FROM_ALL)
add_subdirectory(32.MultiThreadedRefCounting EXCLUDE_FROM_ALL) # TODO: @Anastazluk make this into example 04
add_subdirectory(33.Draw3DLine EXCLUDE_FROM_ALL)
add_subdirectory(34.AddressAllocatorTraitsTest EXCLUDE_FROM_ALL)
add_subdirectory(35.GeometryCreator EXCLUDE_FROM_ALL)
add_subdirectory(36.CUDAInterop EXCLUDE_FROM_ALL)
#add_subdirectory(37.OptiXTriangle EXCLUDE_FROM_ALL)
add_subdirectory(38.EXRSplit EXCLUDE_FROM_ALL)
if (IRR_BUILD_MITSUBA_LOADER AND IRR_BUILD_OPTIX)
	add_subdirectory(39.DenoiserTonemapper EXCLUDE_FROM_ALL)
endif()
add_subdirectory(40.GLITest EXCLUDE_FROM_ALL)
#add_subdirectory(41.VisibilityBuffer EXCLUDE_FROM_ALL)
add_subdirectory(42.EnvmapLookup EXCLUDE_FROM_ALL)
add_subdirectory(43.AllocatorTest EXCLUDE_FROM_ALL) #TODO: @Anastazluk merge example 43,34 into ex 10
add_subdirectory(44.LevelCurveExtraction EXCLUDE_FROM_ALL)
add_subdirectory(45.BRDFEvalTest EXCLUDE_FROM_ALL)
add_subdirectory(46.SamplingValidation EXCLUDE_FROM_ALL)
