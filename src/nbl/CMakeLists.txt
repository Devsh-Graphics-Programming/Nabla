# Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
# Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
# This file is part of the "Nabla Engine".
# For conditions of distribution and use, see copyright notice in nabla.h

include(ProcessorCount)

if(UNIX AND NOT ANDROID)
	# TODO: Now we require the `-dev` package or maybe ship headers ourselves?
	find_package(X11 REQUIRED)
	set(CMAKE_THREAD_PREFER_PTHREAD 1)
	find_package(Threads REQUIRED)
endif()

# set default install prefix
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
	set(PLATFORM NOTFOUND)
	if (WIN32)
		if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
			set(PLATFORM win64-clang)
		elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
			set(PLATFORM win64-gcc)
		elseif (MSVC)
			set(PLATFORM win64-msvc)
		endif()
	elseif (ANDROID)
		if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
			set(PLATFORM android-clang)
		elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
			set(PLATFORM android-gcc)
		endif()
	elseif (UNIX AND NOT APPLE)
		if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
			set(PLATFORM linux-clang)
		elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
			set(PLATFORM linux-gcc)
		endif()
	endif()
	
	if ("${PLATFORM}" STREQUAL "NOTFOUND")
		message(FATAL_ERROR "Unsupported compiler!")
	endif()
	
	set(CMAKE_INSTALL_PREFIX "${NBL_ROOT_PATH}/install/${PLATFORM}" CACHE PATH "Install path" FORCE)
endif()

# include utility macros/functions
include(common)

#[[ Loaders and writers compile options available to edit by user
	All revelant _NBL_COMPILE_WITH will be there]]
option(_NBL_COMPILE_WITH_MTL_LOADER_ "Compile with MTL Loader" OFF) #default off until Material Compiler 2
option(_NBL_COMPILE_WITH_OBJ_LOADER_ "Compile with OBJ Loader" OFF) #default off until Material Compiler 2
#option(_NBL_COMPILE_WITH_OBJ_WRITER_ "Compile with OBJ Writer" ON) uncomment when writer exists
option(_NBL_COMPILE_WITH_STL_LOADER_ "Compile with STL Loader" OFF) #default off until reimplemented
option(_NBL_COMPILE_WITH_STL_WRITER_ "Compile with STL Writer" OFF) #default off until reimplemented
option(_NBL_COMPILE_WITH_PLY_LOADER_ "Compile with PLY Loader" ON)
option(_NBL_COMPILE_WITH_PLY_WRITER_ "Compile with PLY Writer" OFF) #default off until reimplemented
option(_NBL_COMPILE_WITH_JPG_LOADER_ "Compile with JPG Loader" ON)
option(_NBL_COMPILE_WITH_JPG_WRITER_ "Compile with JPG Writer" ON)
option(_NBL_COMPILE_WITH_PNG_LOADER_ "Compile with PNG Loader" ON)
option(_NBL_COMPILE_WITH_PNG_WRITER_ "Compile with PNG Writer" ON)
option(_NBL_COMPILE_WITH_TGA_LOADER_ "Compile with TGA Loader" ON)
option(_NBL_COMPILE_WITH_TGA_WRITER_ "Compile with TGA Writer" ON)
option(_NBL_COMPILE_WITH_OPENEXR_LOADER_ "Compile with OpenEXR Loader" ON)
option(_NBL_COMPILE_WITH_OPENEXR_WRITER_ "Compile with OpenEXR Writer" ON)
option(_NBL_COMPILE_WITH_GLI_LOADER_ "Compile with GLI Loader" ON)
option(_NBL_COMPILE_WITH_GLI_WRITER_ "Compile with GLI Writer" ON)
option(_NBL_COMPILE_WITH_GLTF_LOADER_ "Compile with GLTF Loader" OFF) # TMP OFF COMPILE ERRORS ON V143 ON MASTER
option(_NBL_COMPILE_WITH_GLTF_WRITER_ "Compile with GLTF Writer" OFF) # TMP OFF COMPILE ERRORS ON V143 ON MASTER
set(_NBL_EG_PRFNT_LEVEL 0 CACHE STRING "EasterEgg Profanity Level")

option(NBL_EXPLICIT_MODULE_LOAD_LOG "Enable Runtime logs for external dynamic module loading" OFF)
option(NBL_CPACK_NO_BUILD_DIRECTORY_MODULES "Disable searching for modules in developer's build directory in install mode" OFF)

add_subdirectory(device)

if(NBL_BUILD_ANDROID)
	set(NBL_BUILD_MITSUBA_LOADER OFF CACHE BOOL "Android doesn't need this loader, if you want it, pay us.'" FORCE)
endif()

if(NBL_FAST_MATH)
	set(__NBL_FAST_MATH 1) # this could be in BUILD_INTERFACE, the feature is header only and currently install consumer is forced to use what we used at configure time with no option to toggle it
	# it should be called _NBL_COMPILE_WITH_FAST_MATH to let user known the lib itself was compiled with the feature and __NBL_FAST_MATH should be separate thing to let control it with install build
endif()

if(NBL_TARGET_ARCH_ARM)
	# this should be PRIVATE or at least scoped to build interface only (as currently)
	# list(APPEND PUBLIC_DEFINITIONS $<BUILD_INTERFACE:_NBL_TARGET_ARCH_ARM_>) # TODO: uncomment in the future
endif()

set(CONFIG_DIRECOTORY "${CMAKE_CURRENT_BINARY_DIR}/include/nbl/config")
set(CONFIG_OUTPUT "${CONFIG_DIRECOTORY}/BuildConfigOptions.h")
configure_file("${NBL_ROOT_PATH}/include/nbl/config/BuildConfigOptions.h.in" "${CONFIG_DIRECOTORY}/.int/BuildConfigOptions.h.conf")
file(GENERATE OUTPUT "${CONFIG_OUTPUT}" INPUT "${CONFIG_DIRECOTORY}/.int/BuildConfigOptions.h.conf")
nbl_install_file_spec("${CONFIG_OUTPUT}" nbl/config)

if (NBL_COMPILE_WITH_CUDA)
	message(STATUS "Building with CUDA interop")
	set(_NBL_COMPILE_WITH_CUDA_ ${NBL_COMPILE_WITH_CUDA})
	if (NBL_BUILD_OPTIX)
		set(_NBL_BUILD_OPTIX_ ${NBL_BUILD_OPTIX})
	endif()
endif()

# => TODO: clean
# a little bit of globbing for headers never hurt anyone
file(GLOB_RECURSE TEMP_GLOB_RES "${NBL_ROOT_PATH}/include/*.h")
set(NABLA_HEADERS_PUBLIC ${NABLA_HEADERS_PUBLIC} ${TEMP_GLOB_RES})
file(GLOB_RECURSE TEMP_GLOB_RES "${NBL_ROOT_PATH}/include/*.tcc")
set(NABLA_HEADERS_PUBLIC ${NABLA_HEADERS_PUBLIC} ${TEMP_GLOB_RES})
file(GLOB_RECURSE TEMP_GLOB_RES  "${NBL_ROOT_PATH}/src/*.h")
set(NABLA_HEADERS_PUBLIC2 ${NABLA_HEADERS_PUBLIC2} ${TEMP_GLOB_RES}) # TODO: we should have moved all headers from src directory to /include, but we need it for install target

foreach(NBL_CURRENT_HEADER IN LISTS NABLA_HEADERS_PUBLIC2)
	cmake_path(GET NBL_CURRENT_HEADER PARENT_PATH NBL_CURRENT_HEADER_DIR)
	file(RELATIVE_PATH NBL_TMP_REL_DESTINATION "${NBL_ROOT_PATH}/src" ${NBL_CURRENT_HEADER_DIR})
	nbl_install_file_spec(${NBL_CURRENT_HEADER} ${NBL_TMP_REL_DESTINATION})
endforeach()

unset(NABLA_HEADERS_PUBLIC2 ${NBL_TMP_FULL_PATHS})
# <=

set(NBL_CORE_SOURCES
	core/alloc/refctd_memory_resource.cpp
	core/hash/blake.cpp
)
set(NBL_SYSTEM_SOURCES
	system/DefaultFuncPtrLoader.cpp
	system/IFileBase.cpp
	system/ILogger.cpp
	system/CArchiveLoaderZip.cpp
	system/CArchiveLoaderTar.cpp
	system/CAPKResourcesArchive.cpp
	system/ISystem.cpp
	system/IFileArchive.cpp
	system/CColoredStdoutLoggerWin32.cpp
	system/CStdoutLoggerAndroid.cpp
	system/CFileViewVirtualAllocatorWin32.cpp
	system/CFileViewVirtualAllocatorPOSIX.cpp
	system/CFileViewAPKAllocator.cpp
	system/CFileWin32.cpp
	system/CFilePOSIX.cpp
	system/CSystemWin32.cpp
	system/CSystemAndroid.cpp
	system/ISystemPOSIX.cpp
	system/CSystemLinux.cpp
)
set(NBL_UI_SOURCES
	ui/CWindowWin32.cpp
	ui/CWindowManagerWin32.cpp
	ui/CWindowManagerAndroid.cpp
	ui/CGraphicalApplicationAndroid.cpp
)
set(NBL_ASSET_SOURCES	
# Assets
	asset/ECommonEnums.cpp
	asset/IAsset.cpp
	asset/IRenderpass.cpp
	asset/IAssetManager.cpp
	asset/ICPUDescriptorSet.cpp
	asset/ICPUImage.cpp
	asset/ICPUPolygonGeometry.cpp
	asset/interchange/IAssetWriter.cpp
	asset/interchange/IAssetLoader.cpp
	
# Shaders
	asset/utils/ISPIRVOptimizer.cpp
	asset/utils/ISPIRVEntryPointTrimmer.cpp
	asset/utils/IShaderCompiler.cpp
	asset/utils/CGLSLCompiler.cpp
	asset/utils/CHLSLCompiler.cpp
	asset/utils/CWaveStringResolver.cpp
	asset/utils/CCompilerSet.cpp
	asset/utils/CSPIRVIntrospector.cpp
	asset/interchange/CGLSLLoader.cpp
	asset/interchange/CHLSLLoader.cpp
	asset/interchange/CSPVLoader.cpp

# Pipeline loaders
	asset/interchange/CGraphicsPipelineLoaderMTL.cpp

# Meshes
	asset/utils/CForsythVertexCacheOptimizer.cpp
	asset/utils/CSmoothNormalGenerator.cpp
	asset/utils/CGeometryCreator.cpp
	asset/utils/CPolygonGeometryManipulator.cpp
	asset/utils/COverdrawPolygonGeometryOptimizer.cpp
	asset/utils/CSmoothNormalGenerator.cpp

# Mesh loaders
	asset/interchange/COBJMeshFileLoader.cpp
	asset/interchange/CPLYMeshFileLoader.cpp
	asset/interchange/CSTLMeshFileLoader.cpp
	asset/interchange/CBufferLoaderBIN.cpp
	asset/interchange/CGLTFLoader.cpp

# Mesh writers
	asset/interchange/CPLYMeshWriter.cpp
	asset/interchange/CSTLMeshWriter.cpp
	asset/interchange/CGLTFWriter.cpp

# Images
	asset/interchange/IImageAssetHandlerBase.cpp
	asset/filters/CBasicImageFilterCommon.cpp
	asset/filters/kernels/CConvolutionWeightFunction.cpp
	asset/utils/CDerivativeMapCreator.cpp

# Image loaders
	asset/interchange/IImageLoader.cpp
	asset/interchange/CImageLoaderJPG.cpp
	asset/interchange/CImageLoaderPNG.cpp
	asset/interchange/CImageLoaderTGA.cpp
	asset/interchange/CImageLoaderOpenEXR.cpp # TODO: Nahim
	asset/interchange/CGLILoader.cpp

# Image writers
	asset/interchange/IImageWriter.cpp
	asset/interchange/CImageWriterJPG.cpp
	asset/interchange/CImageWriterPNG.cpp
	asset/interchange/CImageWriterTGA.cpp
	asset/interchange/CImageWriterOpenEXR.cpp # TODO: Nahim
	asset/interchange/CGLIWriter.cpp
)
set(NBL_VIDEO_SOURCES
# Utilities
	video/utilities/ImageRegionIterator.cpp
	video/utilities/ICommandPoolCache.cpp
	video/utilities/IPropertyPool.cpp
	video/utilities/IUtilities.cpp
	video/utilities/CPropertyPoolHandler.cpp
	video/utilities/CScanner.cpp
	video/utilities/CComputeBlit.cpp
	video/utilities/CAssetConverter.cpp

# Interfaces
	video/IAPIConnection.cpp
	video/IPhysicalDevice.cpp
	video/IDescriptorPool.cpp
	video/ILogicalDevice.cpp
	video/IGPUAccelerationStructure.cpp
	video/IGPUCommandBuffer.cpp
	video/IQueue.cpp
	video/IGPUDescriptorSet.cpp
	video/IDeviceMemoryAllocation.cpp
	video/IDeviceMemoryBacked.cpp
	video/ISemaphore.cpp
	video/ISwapchain.cpp
	video/CJITIncludeLoader.cpp
	
# Vulkan
	video/CVulkanDeviceMemoryBacked.cpp
	video/CVulkanSwapchain.cpp
	video/CVulkanImage.cpp
	video/CVulkanQueue.cpp
	video/CVulkanSemaphore.cpp
	video/CVulkanFramebuffer.cpp
	video/CVulkanRenderpass.cpp
	video/CVulkanImageView.cpp
	video/CVulkanCommandBuffer.cpp
	video/CVulkanCommandPool.cpp
	video/CVulkanBuffer.cpp
	video/CVulkanDescriptorSetLayout.cpp
	video/CVulkanPipelineLayout.cpp
	video/CVulkanPipelineCache.cpp
	video/CVulkanComputePipeline.cpp
	video/CVulkanDescriptorPool.cpp
	video/CVulkanDescriptorSet.cpp
	video/CVulkanMemoryAllocation.cpp
	video/CVulkanBufferView.cpp
	video/CVulkanLogicalDevice.cpp
	video/CVulkanDeferredOperation.cpp
	video/CVulkanAccelerationStructure.cpp
	video/CVulkanQueryPool.cpp
	video/CVulkanSampler.cpp
	video/CVulkanConnection.cpp
	video/CVulkanPhysicalDevice.cpp
	video/CVulkanGraphicsPipeline.cpp
	video/CVulkanRayTracingPipeline.cpp
	video/CVulkanEvent.cpp
	video/CSurfaceVulkan.cpp
	
# CUDA
	video/CCUDAHandler.cpp
	video/CCUDADevice.cpp
)

set(NBL_SCENE_SOURCES
	scene/ITransformTree.cpp
)

set(NBL_META_SOURCES
	gtml.cpp
)

set(NABLA_SRCS_COMMON
	${NBL_CORE_SOURCES}
	${NBL_SYSTEM_SOURCES}
	${NBL_UI_SOURCES}
	${NBL_ASSET_SOURCES}
	${NBL_VIDEO_SOURCES}
	${NBL_SCENE_SOURCES}
	${NBL_META_SOURCES}
)

if(MSVC)
	list(APPEND NABLA_SRCS_COMMON "${PROJECT_SOURCE_DIR}/tools/debug/VisualStudio/DynamicArrayVisualizer.natvis")
endif()

get_target_property(PNG_STATIC_BINARY_DIR png_static BINARY_DIR)

set(PUBLIC_BUILD_INCLUDE_DIRS
	${NBL_ROOT_PATH}/include
	${CMAKE_CURRENT_BINARY_DIR}/include
	${THIRD_PARTY_SOURCE_DIR}
	${THIRD_PARTY_SOURCE_DIR}/imath/src
	${THIRD_PARTY_SOURCE_DIR}/gtl/include
	${THIRD_PARTY_SOURCE_DIR}/glm
	${THIRD_PARTY_SOURCE_DIR}/renderdoc 		# for renderdoc api header
	${CMAKE_BINARY_DIR}/3rdparty/zlib 			# for dynamically generated zconf.h
	${PNG_STATIC_BINARY_DIR} 					# for dynamically generated jconfig.h
	${CONFIG_DIRECOTORY}						# Nabla config directory
)

# just gather all the header files and later put them into project so it's easy to search for things using IDE
file(GLOB_RECURSE NABLA_HEADERS_PRIV1 "*.h")
file(GLOB_RECURSE NABLA_HEADERS_PRIV2 "${NBL_ROOT_PATH}/src/nbl/*.h")

set(NABLA_HEADERS "${NBL_ROOT_PATH}/include/nabla.h" ${NABLA_HEADERS_PUBLIC} ${NABLA_HEADERS_PIRV1} ${NABLA_HEADERS_PRIV2})

set(NBL_LIBRARY_CREATION_SOURCES 
	${NABLA_SRCS_COMMON}
	${NABLA_HEADERS}
	$<TARGET_OBJECTS:aesGladman>
	$<TARGET_OBJECTS:lz4>
	$<TARGET_OBJECTS:lzma>
	$<TARGET_OBJECTS:spirv_cross>
)

set(CMAKE_INCLUDE_CURRENT_DIR OFF)
set(CMAKE_INCLUDE_CURRENT_DIR_IN_INTERFACE OFF)

if(NBL_STATIC_BUILD)
	add_library(Nabla STATIC
		${NBL_LIBRARY_CREATION_SOURCES}
	)
else()
	add_library(Nabla SHARED
		${NBL_LIBRARY_CREATION_SOURCES}
	)
endif()
add_library(Nabla::Nabla ALIAS Nabla)

# from old build config template:
# 
# // architecture (TODO: rethink)
# cmakedefine __NBL_COMPILE_WITH_ARM_SIMD_ 					# <- never used, and if even so - should be PRIVATE
# 
# // DXC compile flags
# cmakedefine DXC_COMPILE_FLAGS "@DXC_COMPILE_FLAGS@" 		# <- never used, and if even so - should be PRIVATE

# usage requirements, shared in build & install interface
set(PUBLIC_DEFINITIONS
	# configs
	$<$<CONFIG:Debug>:_NBL_DEBUG>
	$<$<CONFIG:RelWithDebInfo>:_NBL_RELWITHDEBINFO>

	# https://cmake.org/cmake/help/latest/variable/CMAKE_SYSTEM_NAME.html#system-names-known-to-cmake
	$<$<PLATFORM_ID:Windows>:_NBL_PLATFORM_WINDOWS_>
    $<$<PLATFORM_ID:Linux>:_NBL_PLATFORM_LINUX_>
    $<$<PLATFORM_ID:Darwin>:_NBL_PLATFORM_OSX_>
    $<$<PLATFORM_ID:Android>:_NBL_PLATFORM_ANDROID_>

	# extras
	BOOST_ALL_NO_LIB
	NOMINMAX
	_CRT_SECURE_NO_DEPRECATE
	_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR
)

if(NOT NBL_STATIC_BUILD)
	list(APPEND PUBLIC_DEFINITIONS _NBL_SHARED_BUILD_)
endif()

target_compile_definitions(Nabla
	PUBLIC ${PUBLIC_DEFINITIONS}
)

nbl_adjust_flags(TARGET Nabla MAP_RELEASE Release MAP_RELWITHDEBINFO RelWithDebInfo MAP_DEBUG Debug)
nbl_adjust_definitions()

option(NBL_WAVE_STRING_RESOLVER_TU_DEBUG_OPTIMISATION "Enable to optimise CWaveStringResolver.cpp in Debug configuration, uses RWDI compile options for the TU" ON)
if(NBL_WAVE_STRING_RESOLVER_TU_DEBUG_OPTIMISATION)
	set_source_files_properties(asset/utils/CWaveStringResolver.cpp PROPERTIES
		# just enabling inlining and optimisations will help a lot
		COMPILE_OPTIONS "$<$<CONFIG:Debug>:${NBL_CXX_RELWITHDEBINFO_COMPILE_OPTIONS}>"
		# trade is you cannot mix with PCH + set this property per config (it seems), different compile options would lead to corrruptions and undefined behaviours
		SKIP_PRECOMPILE_HEADERS ON
	)
endif()

if(NBL_EXPLICIT_MODULE_LOAD_LOG)
	target_compile_definitions(Nabla PUBLIC NBL_EXPLICIT_MODULE_LOAD_LOG)
endif()

if(NBL_CPACK_NO_BUILD_DIRECTORY_MODULES)
	target_compile_definitions(Nabla PUBLIC NBL_CPACK_NO_BUILD_DIRECTORY_MODULES)
endif()

set(INTERFACE_BUILD_DEFINITIONS
	_DXC_DLL_="${DXC_DLL}"
)

target_compile_definitions(Nabla 
	INTERFACE _DXC_DLL_NAME_="dxcompiler"
	PRIVATE __NBL_BUILDING_NABLA__
)
# OK: im assuming "Nabla" target will be SHARED and we will have separate Nabla-static target in future 
# I'm letting it to trigger linker warnings in static build now as I need this generator expressions to propagete to install consumer
target_link_options(Nabla INTERFACE $<$<CXX_COMPILER_FRONTEND_VARIANT:MSVC>:/DELAYLOAD:$<TARGET_FILE_NAME:Nabla::Nabla>>)
if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT MATCHES MSVC)
	target_link_options(Nabla
		PRIVATE /DELAYLOAD:dxcompiler.dll
	)
endif()
target_link_libraries(Nabla PUBLIC $<$<PLATFORM_ID:Windows>:delayimp>)

if (ANDROID)
	add_library(android_native_app_glue STATIC
    	${ANDROID_NDK_ROOT_PATH}/sources/android/native_app_glue/android_native_app_glue.c
	)
	target_link_libraries(Nabla PUBLIC 
		android_native_app_glue
		android
		log
	)
	list(APPEND PUBLIC_BUILD_INCLUDE_DIRS ${ANDROID_NDK_ROOT_PATH}/sources/android/native_app_glue)
endif()

set(NBL_ASSEMBLY_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/devshgraphicsprogramming.nabla")
if(NOT NBL_STATIC_BUILD)
	set_target_properties(Nabla PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${NBL_ASSEMBLY_DIRECTORY})
	list(APPEND INTERFACE_BUILD_DEFINITIONS
		_NABLA_OUTPUT_DIR_="${NBL_ASSEMBLY_DIRECTORY}"
	)

	target_compile_definitions(Nabla
		INTERFACE _NABLA_DLL_NAME_="$<PATH:REMOVE_EXTENSION,$<TARGET_FILE_NAME:Nabla::Nabla>>"
	)
endif()

## Set up 3rdparty deps
# Parallel Hashmap
list(APPEND PUBLIC_BUILD_INCLUDE_DIRS ${THIRD_PARTY_SOURCE_DIR}/parallel-hashmap/parallel_hashmap)
# TCPP
list(APPEND PUBLIC_BUILD_INCLUDE_DIRS ${THIRD_PARTY_SOURCE_DIR}/tcpp)
# oneDPL for clang C++17 parallelism
if(_NBL_BUILD_DPL_)
	add_dependencies(Nabla oneDPL)
	list(APPEND PUBLIC_BUILD_INCLUDE_DIRS
		${ONETBB_INSTALL_DIR}/include
		${ONEDPL_INSTALL_DIR}/include
	)
	target_link_libraries(Nabla INTERFACE tbb tbbmalloc tbbmalloc_proxy)
endif()

# bzip2
if(NBL_STATIC_BUILD)
	target_link_libraries(Nabla INTERFACE bz2_static)
else()
	target_link_libraries(Nabla PRIVATE bz2_static)
endif()
add_dependencies(Nabla bz2_static)

# boost
list(APPEND PUBLIC_BUILD_INCLUDE_DIRS ${BOOST_PREPROCESSOR_INCLUDE})

if (UNIX)
	target_link_libraries(Nabla INTERFACE 
		${CMAKE_THREAD_LIBS_INIT}
		${CMAKE_DL_LIBS}
	)
	target_link_libraries(Nabla PUBLIC atomic)
endif()

# zlib
if(NBL_STATIC_BUILD)
	target_link_libraries(Nabla INTERFACE zlibstatic)
else()
	target_link_libraries(Nabla PRIVATE zlibstatic)
endif()

# blake3
add_dependencies(Nabla blake3)
list(APPEND PUBLIC_BUILD_INCLUDE_DIRS $<TARGET_PROPERTY:blake3,INCLUDE_DIRECTORIES>)
if(NBL_STATIC_BUILD)
	target_link_libraries(Nabla INTERFACE blake3)
else()
	target_link_libraries(Nabla PRIVATE blake3)
endif()

# shaderc
add_dependencies(Nabla shaderc)
if(NBL_STATIC_BUILD)
	target_link_libraries(Nabla INTERFACE shaderc)
else()
	target_link_libraries(Nabla PRIVATE shaderc)
endif()
list(APPEND PUBLIC_BUILD_INCLUDE_DIRS ${THIRD_PARTY_SOURCE_DIR}/shaderc/libshaderc/include)

# spirv tools
add_dependencies(Nabla SPIRV)
add_dependencies(Nabla SPIRV-Tools)
add_dependencies(Nabla SPIRV-Tools-opt)
if(NBL_STATIC_BUILD)
	target_link_libraries(Nabla INTERFACE SPIRV SPIRV-Tools SPIRV-Tools-opt)
else()
	target_link_libraries(Nabla PRIVATE SPIRV SPIRV-Tools SPIRV-Tools-opt)
endif()
target_include_directories(Nabla BEFORE 
	PRIVATE "${THIRD_PARTY_SOURCE_DIR}/dxc/dxc/include"
	PRIVATE $<TARGET_PROPERTY:SPIRV-Tools-static,INCLUDE_DIRECTORIES>
	PRIVATE $<TARGET_PROPERTY:SPIRV-Headers,INCLUDE_DIRECTORIES>
)

# nlohmann_json
add_dependencies(Nabla nlohmann_json)
if(NBL_STATIC_BUILD)
	target_link_libraries(Nabla INTERFACE nlohmann_json::nlohmann_json)
else()
	target_link_libraries(Nabla PRIVATE nlohmann_json::nlohmann_json)
endif()
list(APPEND PUBLIC_BUILD_INCLUDE_DIRS ${THIRD_PARTY_SOURCE_DIR}/nlohmann_json/include)


# simdjson
add_dependencies(Nabla simdjson)
if(NBL_STATIC_BUILD)
	target_link_libraries(Nabla INTERFACE simdjson)
else()
	target_link_libraries(Nabla PRIVATE simdjson)
endif()
list(APPEND PUBLIC_BUILD_INCLUDE_DIRS ${THIRD_PARTY_SOURCE_DIR}/simdjson)

# libjpeg
add_dependencies(Nabla jpeg-static)
if(NBL_STATIC_BUILD)
	target_link_libraries(Nabla PUBLIC jpeg-static)
else()
	target_link_libraries(Nabla PRIVATE jpeg-static)
endif()
list(APPEND PUBLIC_BUILD_INCLUDE_DIRS ${THIRD_PARTY_SOURCE_DIR}/libjpeg-turbo)

# libpng
add_dependencies(Nabla png_static)
if(NBL_STATIC_BUILD)
	target_link_libraries(Nabla INTERFACE png_static)
else()
	target_link_libraries(Nabla PRIVATE png_static)
endif()
list(APPEND PUBLIC_BUILD_INCLUDE_DIRS ${THIRD_PARTY_SOURCE_DIR}/libpng)


# OpenEXR
if (_NBL_COMPILE_WITH_OPEN_EXR_)
    add_dependencies(Nabla OpenEXR)
    if(NBL_STATIC_BUILD)
        target_link_libraries(Nabla INTERFACE OpenEXR)
    else()
        target_link_libraries(Nabla PRIVATE OpenEXR)
    endif()

	list(APPEND PUBLIC_BUILD_INCLUDE_DIRS $<TARGET_PROPERTY:OpenEXR,INCLUDE_DIRECTORIES>)
endif()

add_dependencies(Nabla OpenEXRCore OpenEXRUtil)
if(NBL_STATIC_BUILD)
    target_link_libraries(Nabla INTERFACE 
      OpenEXRCore
      OpenEXRUtil
)
else()
    target_link_libraries(Nabla PRIVATE 
      OpenEXRCore
      OpenEXRUtil  
  )
endif()

list(APPEND PUBLIC_BUILD_INCLUDE_DIRS
	$<TARGET_PROPERTY:OpenEXRCore,INCLUDE_DIRECTORIES>
	$<TARGET_PROPERTY:OpenEXRUtil,INCLUDE_DIRECTORIES>
)

# GLI
if(_NBL_COMPILE_WITH_GLI_)
	add_dependencies(Nabla gli)
	if(NBL_STATIC_BUILD)
		target_link_libraries(Nabla INTERFACE gli)
	else()
		target_link_libraries(Nabla PRIVATE gli)
	endif()
	list(APPEND PUBLIC_BUILD_INCLUDE_DIRS ${THIRD_PARTY_SOURCE_DIR}/gli)
endif()

# DXC (TODO: nah, lets remove/disable it)
if(_NBL_ENABLE_DXC_COMPILE_TESTS_)
	add_dependencies(Nabla HLSL_NABLA_COMPILE_TEST)
endif()

# TODO: remove and replace with target in tree
add_custom_target(dxcompiler 
	COMMAND "${CMAKE_COMMAND}" --build "${DXC_BUILD_DIR}" --target dxcompiler --config $<IF:$<CONFIG:Debug>,Debug,Release> -j${_NBL_JOBS_AMOUNT_}
	BYPRODUCTS "${DXC_LIB}" "${DXC_DLL}"
	VERBATIM
	COMMAND_EXPAND_LISTS
)

add_dependencies(Nabla dxcompiler)
add_dependencies(3rdparty dxcompiler)

if(NBL_STATIC_BUILD)
	target_link_libraries(Nabla INTERFACE "${DXC_LIB}")
else()
	target_link_libraries(Nabla PRIVATE "${DXC_LIB}")
endif()
	
# Linux Display Systems
if (UNIX AND NOT ANDROID AND NOT APPLE)
	list(APPEND PUBLIC_BUILD_INCLUDE_DIRS 
		${X11_INCLUDE_DIR} 
		${X11_Xrandr_INCLUDE_PATH}
		${X11_xf86vmode_INCLUDE_PATH}
	)
endif()

# Vulkan
list(APPEND PUBLIC_BUILD_INCLUDE_DIRS 
	${THIRD_PARTY_SOURCE_DIR}/Vulkan-Headers/include
	${THIRD_PARTY_SOURCE_DIR}/volk
)
target_link_libraries(Nabla PRIVATE volk)
# volk is part of public interface headers in Nabla
target_compile_definitions(Nabla PUBLIC $<$<PLATFORM_ID:Windows>:VK_USE_PLATFORM_WIN32_KHR>)

# CUDA
if (NBL_COMPILE_WITH_CUDA)
	list(APPEND PUBLIC_BUILD_INCLUDE_DIRS "${CUDAToolkit_INCLUDE_DIRS}")
endif()

list(APPEND PUBLIC_BUILD_INCLUDE_DIRS
	# this should be PRIVATE, but things from /src (or /source) are sometimes included in things in /include and so examples have to put source dirs into theirs Include Path
	# -> TODO
	${NBL_ROOT_PATH}/src 
	${NBL_ROOT_PATH}/source/Nabla 
	${NBL_ROOT_PATH}/src/3rdparty
	# <-
)

target_include_directories(Nabla PRIVATE ${CMAKE_BINARY_DIR}/include)

target_compile_definitions(Nabla 
	PRIVATE SPIRV_CROSS_EXCEPTIONS_TO_ASSERTIONS
	PUBLIC _NBL_PCH_IGNORE_PRIVATE_HEADERS # TODO: remove it
)

# Boost
add_dependencies(Nabla boost_wave)
list(APPEND PUBLIC_BUILD_INCLUDE_DIRS $<TARGET_PROPERTY:boost_wave,INCLUDE_DIRECTORIES>)

if(NBL_STATIC_BUILD)
	target_link_libraries(Nabla PUBLIC boost_wave)
else()
	target_link_libraries(Nabla PRIVATE boost_wave)
endif()

add_dependencies(Nabla DeviceHeaders)
list(APPEND PUBLIC_BUILD_INCLUDE_DIRS ${NBL_DEVICE_GEN_INCLUDE_DIR})

if(NBL_EMBED_BUILTIN_RESOURCES)
	add_subdirectory(builtin EXCLUDE_FROM_ALL)
	
	function(NBL_HOOK_INTERNAL_BUILTIN_RESOURCES NBL_TARGETS)	
		foreach(NBL_TARGET IN LISTS NBL_TARGETS)
			get_target_property(_INTERNAL_BR_OUTPUT_INCLUDE_SEARCH_DIRECTORY_ ${NBL_TARGET} BUILTIN_RESOURCES_INCLUDE_SEARCH_DIRECTORY)
			target_compile_definitions(${NBL_TARGET} PUBLIC "$<TARGET_PROPERTY:Nabla,INTERFACE_COMPILE_DEFINITIONS>") # workaround because must use Nabla headers without linking Nabla to itself..
			target_include_directories(${NBL_TARGET} PUBLIC "$<TARGET_PROPERTY:Nabla,INCLUDE_DIRECTORIES>")
			target_include_directories(Nabla PRIVATE "${_INTERNAL_BR_OUTPUT_INCLUDE_SEARCH_DIRECTORY_}")
		endforeach()
	
		add_dependencies(Nabla ${NBL_TARGETS})
		
		if(NBL_STATIC_BUILD)
			set(_NBL_LINK_QUALIFIER_ INTERFACE)
		else()
			set(_NBL_LINK_QUALIFIER_ PRIVATE)
		endif()
	
		target_link_libraries(Nabla ${_NBL_LINK_QUALIFIER_} ${NBL_TARGETS})
	endfunction()
	
	set(NBL_INTERNAL_BUILTIN_TARGETS
		nblBuiltinResourceData 
		spirvBuiltinResourceData 
		boostBuiltinResourceData 
		deviceGenBuiltinResourceData
	)
	
	NBL_HOOK_INTERNAL_BUILTIN_RESOURCES("${NBL_INTERNAL_BUILTIN_TARGETS}")
	
	# always install Nabla builtins (some may be included in cpp regardless if embedded or not)
	get_target_property(_BUILTIN_RESOURCES_BUNDLE_ARCHIVE_ABSOLUTE_PATH_ nblBuiltinResourceData BUILTIN_RESOURCES_BUNDLE_ARCHIVE_ABSOLUTE_PATH)
	get_target_property(NABLA_RESOURCES_TO_EMBED_PUBLIC nblBuiltinResourceData BUILTIN_RESOURCES)
	
	target_sources(Nabla PRIVATE 
		${NABLA_RESOURCES_TO_EMBED_PUBLIC}
	)
	
	list(APPEND NABLA_HEADERS_PUBLIC ${NABLA_RESOURCES_TO_EMBED_PUBLIC})
	source_group("Resources to embed" FILES ${NABLA_RESOURCES_TO_EMBED_PUBLIC})
	set_source_files_properties(${NABLA_RESOURCES_TO_EMBED_PUBLIC} PROPERTIES HEADER_FILE_ONLY TRUE)
else()
	# this is only visible in cpps and should never leak to public interface
	target_compile_definitions(Nabla
		PRIVATE NBL_BUILTIN_RESOURCES_DIRECTORY_PATH="${NBL_BUILTIN_RESOURCES_DIRECTORY_PATH}"
		PRIVATE SPIRV_BUILTIN_RESOURCES_DIRECTORY_PATH="${SPIRV_BUILTIN_RESOURCES_DIRECTORY_PATH}"
		PRIVATE BOOST_BUILTIN_RESOURCES_DIRECTORY_PATH="${BOOST_BUILTIN_RESOURCES_DIRECTORY_PATH}"
		PRIVATE DEVICEGEN_BUILTIN_RESOURCES_DIRECTORY_PATH="${DEVICEGEN_BUILTIN_RESOURCES_DIRECTORY_PATH}"
	)
endif()

list(APPEND INTERFACE_BUILD_DEFINITIONS
	_NABLA_INSTALL_DIR_="${CMAKE_INSTALL_PREFIX}"
)

# git version tracking
# TODO: private
target_link_libraries(Nabla PUBLIC $<BUILD_INTERFACE:gtml>)

# vulkaninfo
if (NBL_STATIC_BUILD)
	target_link_libraries(Nabla PUBLIC $<BUILD_INTERFACE:vulkaninfo>)
else()
	target_link_libraries(Nabla PRIVATE $<BUILD_INTERFACE:vulkaninfo>)
endif()

# NGFX
if(TARGET ngfx)
	if(NBL_STATIC_BUILD)
		target_link_libraries(Nabla INTERFACE ngfx)
	else()
		target_link_libraries(Nabla PRIVATE ngfx)
	endif()

	target_include_directories(Nabla PRIVATE $<TARGET_PROPERTY:ngfx,INTERFACE_INCLUDE_DIRECTORIES>)
	target_compile_definitions(Nabla 
		PRIVATE NBL_BUILD_WITH_NGFX
		PRIVATE $<TARGET_PROPERTY:ngfx,INTERFACE_COMPILE_DEFINITIONS>
	)
endif()

# on MSVC it won't compile without this option!
target_compile_options(Nabla PUBLIC $<$<CXX_COMPILER_FRONTEND_VARIANT:MSVC>:/bigobj>)

if(NBL_PCH)
	target_precompile_headers(Nabla
		# private as nothing from source directory should ever leak to downstream targets!
		# NOTE: currently our whole public and private interface is broken
		# and private headers leak to public includes
		PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/pch.h"
	)
endif()

# extensions
start_tracking_variables_for_propagation_to_parent()
add_subdirectory(ext EXCLUDE_FROM_ALL)
propagate_changed_variables_to_parent_scope()

nbl_install_headers("${NABLA_HEADERS_PUBLIC}")
nbl_install_file_spec("${NBL_ROOT_PATH_BINARY}/include/nbl/builtin/builtinResources.h" "nbl/builtin")

if(NBL_EMBED_BUILTIN_RESOURCES)
	nbl_install_builtin_resources(nblBuiltinResourceData)
	nbl_install_builtin_resources(spirvBuiltinResourceData)
	nbl_install_builtin_resources(boostBuiltinResourceData)
endif()

set_target_properties(Nabla PROPERTIES DEBUG_POSTFIX _debug)
set_target_properties(Nabla PROPERTIES RELWITHDEBINFO_POSTFIX _relwithdebinfo)

set(INSTALL_INCLUDE_DIRS
	include
	include/nbl/config # TMP, to let #include "BuildConfigOptions.h" work
)

# target_include_directories for Nabla with PUBLIC *must not* be called after following
target_include_directories(Nabla PUBLIC
	"$<BUILD_INTERFACE:${PUBLIC_BUILD_INCLUDE_DIRS}>"
	"$<INSTALL_INTERFACE:${INSTALL_INCLUDE_DIRS}>"
)

target_compile_definitions(Nabla INTERFACE 
	"$<BUILD_INTERFACE:${INTERFACE_BUILD_DEFINITIONS}>"
)

if(NBL_STATIC_BUILD)
	# install dependencies, they are required only for static Nabla builds
	nbl_install_lib(glslang)
	nbl_install_lib(GenericCodeGen)
	nbl_install_lib(MachineIndependent)
	nbl_install_lib(jpeg-static)
	if (_NBL_COMPILE_WITH_OPEN_EXR_)
		nbl_install_lib(OpenEXR)
	else()
		nbl_install_lib(OpenEXRCore)
		nbl_install_lib(OpenEXRUtil)
	endif()
	nbl_install_lib(png_static)
	nbl_install_lib(shaderc)
	nbl_install_lib(shaderc_util)
	nbl_install_lib(SPIRV)
	nbl_install_lib(SPIRV-Tools-static) # TODO: make this function/macro work with alias target
	nbl_install_lib(SPIRV-Tools-opt)
	nbl_install_lib(OSDependent)
	nbl_install_lib(zlibstatic)
	nbl_install_lib(simdjson)
	nbl_install_lib(volk)
	
	if(_NBL_BUILD_DPL_)
		nbl_install_program(tbb)
		nbl_install_program(tbbmalloc)
		nbl_install_program(tbbmalloc_proxy)
	endif()
else()
	nbl_install_program_spec(Nabla "nbl")
endif()

nbl_install_program_spec("${DXC_DLL}" "nbl/3rdparty/dxc")
nbl_install_dir_spec(../../include/nbl/application_templates nbl)

# note: order important, keep after install rules due to NBL_3RDPARTY_DXC_NS_PACKAGE_RUNTIME_DLL_DIR_PATH property
get_property(_NBL_DXC_PACKAGE_RUNTIME_DLL_DIR_PATH_ GLOBAL PROPERTY NBL_3RDPARTY_DXC_NS_PACKAGE_RUNTIME_DLL_DIR_PATH)
get_target_property(_NBL_NABLA_PACKAGE_RUNTIME_DLL_DIR_PATH_ Nabla NBL_PACKAGE_RUNTIME_DLL_DIR_PATH)

target_compile_definitions(Nabla 
	INTERFACE NBL_CPACK_PACKAGE_NABLA_DLL_DIR_ABS_KEY="${_NBL_NABLA_PACKAGE_RUNTIME_DLL_DIR_PATH_}"
	INTERFACE NBL_CPACK_PACKAGE_DXC_DLL_DIR_ABS_KEY="${_NBL_DXC_PACKAGE_RUNTIME_DLL_DIR_PATH_}"
)

NBL_ADJUST_FOLDERS(src)