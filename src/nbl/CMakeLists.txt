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

add_subdirectory(device EXCLUDE_FROM_ALL)

if(NBL_BUILD_ANDROID)
	set(NBL_BUILD_MITSUBA_LOADER OFF CACHE BOOL "Android doesn't need this loader, if you want it, pay us.'" FORCE)
endif()

nbl_generate_conf_files()

nbl_get_conf_dir(NABLA_CONF_DIR_DEBUG Debug)
nbl_get_conf_dir(NABLA_CONF_DIR_RELEASE Release)
nbl_get_conf_dir(NABLA_CONF_DIR_RELWITHDEBINFO RelWithDebInfo)

if (NBL_COMPILE_WITH_CUDA)
	message(STATUS "Building with CUDA interop")
	set(_NBL_COMPILE_WITH_CUDA_ ${NBL_COMPILE_WITH_CUDA})
	if (NBL_BUILD_OPTIX)
		set(_NBL_BUILD_OPTIX_ ${NBL_BUILD_OPTIX})
	endif()
endif()

#set(_NBL_TARGET_ARCH_ARM_ ${NBL_TARGET_ARCH_ARM}) #uncomment in the future

set(__NBL_FAST_MATH ${NBL_FAST_MATH})

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
	
	install(FILES ${NBL_CURRENT_HEADER} DESTINATION include/${NBL_TMP_REL_DESTINATION} CONFIGURATIONS Release)
	install(FILES ${NBL_CURRENT_HEADER} DESTINATION debug/include/${NBL_TMP_REL_DESTINATION} CONFIGURATIONS Debug)
	install(FILES ${NBL_CURRENT_HEADER} DESTINATION relwithdebinfo/include/${NBL_TMP_REL_DESTINATION} CONFIGURATIONS RelWithDebInfo)
endforeach()

unset(NABLA_HEADERS_PUBLIC2 ${NBL_TMP_FULL_PATHS})
#

set(NBL_CORE_SOURCES
	core/IReferenceCounted.cpp
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
	
# Material compiler
	asset/material_compiler/CMaterialCompilerGLSLBackendCommon.cpp
	asset/material_compiler/CMaterialCompilerGLSLRasterBackend.cpp
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

set(COMMON_INCLUDE_DIRS
	${THIRD_PARTY_SOURCE_DIR}/glm
	${THIRD_PARTY_SOURCE_DIR}/renderdoc # for renderdoc api header
	${CMAKE_BINARY_DIR}/3rdparty/zlib #for dynamically generated zconf.h
	$<TARGET_PROPERTY:png_static,BINARY_DIR> #for dynamically generated pnglibconf.h
	$<TARGET_PROPERTY:jpeg-static,BINARY_DIR> #for dynamically generated jconfig.h
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

if(NBL_STATIC_BUILD)
	add_library(Nabla STATIC
		${NBL_LIBRARY_CREATION_SOURCES}
	)
else()
	add_library(Nabla SHARED
		${NBL_LIBRARY_CREATION_SOURCES}
	)
endif()

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

target_compile_definitions(Nabla 
	PUBLIC _DXC_DLL_="${DXC_DLL}"
	PRIVATE __NBL_BUILDING_NABLA__
)

if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT MATCHES MSVC)
	target_link_options(Nabla 
		INTERFACE /DELAYLOAD:$<TARGET_FILE_NAME:Nabla>
		PRIVATE /DELAYLOAD:dxcompiler.dll
	)
endif()

if (ANDROID)
	add_library(android_native_app_glue STATIC
    	${ANDROID_NDK_ROOT_PATH}/sources/android/native_app_glue/android_native_app_glue.c
	)
	target_link_libraries(Nabla PUBLIC 
		android_native_app_glue
		android
		log
	)
	target_include_directories(Nabla PUBLIC
    	${ANDROID_NDK_ROOT_PATH}/sources/android/native_app_glue
	)
endif()

set(NBL_ASSEMBLY_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/devshgraphicsprogramming.nabla")
if(NOT NBL_STATIC_BUILD)
	set_target_properties(Nabla PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${NBL_ASSEMBLY_DIRECTORY})
	target_compile_definitions(Nabla PUBLIC
		_NABLA_DLL_NAME_="$<PATH:REMOVE_EXTENSION,$<TARGET_FILE_NAME:Nabla>>";_NABLA_OUTPUT_DIR_="${NBL_ASSEMBLY_DIRECTORY}"
	)
endif()

## Set up 3rdparty deps
# Parallel Hashmap
target_include_directories(Nabla PUBLIC ${THIRD_PARTY_SOURCE_DIR}/parallel-hashmap/parallel_hashmap)
# TCPP
target_include_directories(Nabla PUBLIC ${THIRD_PARTY_SOURCE_DIR}/tcpp)
# oneDPL for clang C++17 parallelism
if(_NBL_BUILD_DPL_)
	add_dependencies(Nabla oneDPL)
	target_include_directories(Nabla PUBLIC ${ONETBB_INSTALL_DIR}/include)
	target_include_directories(Nabla PUBLIC ${ONEDPL_INSTALL_DIR}/include)
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
target_include_directories(Nabla PUBLIC "${BOOST_PREPROCESSOR_INCLUDE}")

if (UNIX)
	target_link_libraries(Nabla INTERFACE 
		${CMAKE_THREAD_LIBS_INIT}
		${CMAKE_DL_LIBS}
	)
	target_link_libraries(Nabla PUBLIC atomic)
endif()

# zlib
add_dependencies(zlibstatic zlib_copy)
add_dependencies(Nabla zlibstatic)

if(NBL_STATIC_BUILD)
	target_link_libraries(Nabla INTERFACE zlibstatic)
	target_link_libraries(Nabla INTERFACE
		$<$<CONFIG:Debug>:${ZLIB_LIBRARY_DEBUG}>
		$<$<CONFIG:Release>:${ZLIB_LIBRARY_RELEASE}>
		$<$<CONFIG:RelWithDebInfo>:${ZLIB_LIBRARY_RELWITHDEBINFO}>
	)
else()
	target_link_libraries(Nabla PRIVATE zlibstatic)
	target_link_libraries(Nabla PRIVATE
		$<$<CONFIG:Debug>:${ZLIB_LIBRARY_DEBUG}>
		$<$<CONFIG:Release>:${ZLIB_LIBRARY_RELEASE}>
		$<$<CONFIG:RelWithDebInfo>:${ZLIB_LIBRARY_RELWITHDEBINFO}>
	)
endif()
target_include_directories(Nabla PUBLIC $<TARGET_PROPERTY:zlibstatic,BINARY_DIR>/copy_source)

# blake3
add_dependencies(Nabla blake3)
target_include_directories(Nabla PUBLIC $<TARGET_PROPERTY:blake3,INCLUDE_DIRECTORIES>)
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
target_include_directories(Nabla PUBLIC ${THIRD_PARTY_SOURCE_DIR}/shaderc/libshaderc/include)
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
target_include_directories(Nabla PUBLIC ${THIRD_PARTY_SOURCE_DIR}/nlohmann_json/include)


# simdjson
add_dependencies(Nabla simdjson)
if(NBL_STATIC_BUILD)
	target_link_libraries(Nabla INTERFACE simdjson)
else()
	target_link_libraries(Nabla PRIVATE simdjson)
endif()
target_include_directories(Nabla PUBLIC ${THIRD_PARTY_SOURCE_DIR}/simdjson)

# libjpeg
add_dependencies(Nabla jpeg-static)
if(NBL_STATIC_BUILD)
	target_link_libraries(Nabla INTERFACE jpeg-static)
else()
	target_link_libraries(Nabla PRIVATE jpeg-static)
endif()
target_include_directories(Nabla PUBLIC ${THIRD_PARTY_SOURCE_DIR}/libjpeg-turbo)
# libpng
add_dependencies(Nabla png_static)
if(NBL_STATIC_BUILD)
	target_link_libraries(Nabla INTERFACE png_static)
else()
	target_link_libraries(Nabla PRIVATE png_static)
endif()
target_include_directories(Nabla PUBLIC ${THIRD_PARTY_SOURCE_DIR}/libpng)
# OpenEXR
if (_NBL_COMPILE_WITH_OPEN_EXR_)
    add_dependencies(Nabla OpenEXR)
    if(NBL_STATIC_BUILD)
        target_link_libraries(Nabla INTERFACE OpenEXR)
    else()
        target_link_libraries(Nabla PRIVATE OpenEXR)
    endif()
    
    target_include_directories(
        Nabla PUBLIC $<TARGET_PROPERTY:OpenEXR,INCLUDE_DIRECTORIES>                   
    )
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

target_include_directories(
    Nabla PUBLIC $<TARGET_PROPERTY:OpenEXRCore,INCLUDE_DIRECTORIES> # for OpenEXR's core headers
    Nabla PUBLIC $<TARGET_PROPERTY:OpenEXRUtil,INCLUDE_DIRECTORIES> # for OpenEXR's util headers                
)

# GLI
if(_NBL_COMPILE_WITH_GLI_)
	add_dependencies(Nabla gli)
	if(NBL_STATIC_BUILD)
		target_link_libraries(Nabla INTERFACE gli)
	else()
		target_link_libraries(Nabla PRIVATE gli)
	endif()
	target_include_directories(Nabla PUBLIC ${THIRD_PARTY_SOURCE_DIR}/gli)
endif()

# DXC (TODO: nah, lets remove/disable it)
if(_NBL_ENABLE_DXC_COMPILE_TESTS_)
	add_dependencies(Nabla HLSL_NABLA_COMPILE_TEST)
endif()

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
	target_include_directories(Nabla PUBLIC 
		${X11_INCLUDE_DIR} 
		X11_Xrandr_INCLUDE_PATH
		${X11_xf86vmode_INCLUDE_PATH}
	)
endif()

# Vulkan
target_include_directories(Nabla PUBLIC "${THIRD_PARTY_SOURCE_DIR}/Vulkan-Headers/include")
target_include_directories(Nabla PUBLIC "${THIRD_PARTY_SOURCE_DIR}/volk")
target_link_libraries(Nabla PRIVATE volk)
if(WIN32)
	target_compile_definitions(Nabla PRIVATE VK_USE_PLATFORM_WIN32_KHR)
endif()
# CUDA
if (NBL_COMPILE_WITH_CUDA)
	target_include_directories(Nabla PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
endif()

# Include dirs for self
target_include_directories(Nabla PUBLIC 
	"${CMAKE_CURRENT_BINARY_DIR}/include"
	"${NBL_ROOT_PATH}/include"
	${COMMON_INCLUDE_DIRS}
	"${THIRD_PARTY_SOURCE_DIR}"
	#those also goes as PUBLIC because of examples
	"$<$<CONFIG:DEBUG>:${NABLA_CONF_DIR_DEBUG}>"
	"$<$<CONFIG:RELEASE>:${NABLA_CONF_DIR_RELEASE}>"
	"$<$<CONFIG:RELWITHDEBINFO>:${NABLA_CONF_DIR_RELWITHDEBINFO}>"
)

target_include_directories(Nabla PUBLIC ${NBL_ROOT_PATH}/src ${NBL_ROOT_PATH}/source/Nabla ${NBL_ROOT_PATH}/src/3rdparty) # this should be PRIVATE, but things from /src (or /source) are sometimes included in things in /include and so examples have to put source dirs into theirs Include Path
target_include_directories(Nabla PRIVATE ${CMAKE_BINARY_DIR}/include)

target_compile_definitions(Nabla 
	PRIVATE SPIRV_CROSS_EXCEPTIONS_TO_ASSERTIONS
	INTERFACE _NBL_PCH_IGNORE_PRIVATE_HEADERS
)

# Boost
add_dependencies(Nabla boost_wave)
target_include_directories(Nabla PUBLIC $<TARGET_PROPERTY:boost_wave,INCLUDE_DIRECTORIES>)
if(NBL_STATIC_BUILD)
	target_link_libraries(Nabla INTERFACE boost_wave)
else()
	target_link_libraries(Nabla PRIVATE boost_wave)
endif()

add_dependencies(Nabla DeviceHeaders)
target_include_directories(Nabla PUBLIC "${NBL_DEVICE_GEN_INCLUDE_DIR}")

if(NBL_EMBED_BUILTIN_RESOURCES)
	add_subdirectory(builtin EXCLUDE_FROM_ALL)
	
	function(NBL_HOOK_INTERNAL_BUILTIN_RESOURCES NBL_TARGETS)
		get_target_property(_NABLA_INCLUDE_DIRECTORIES_ Nabla INCLUDE_DIRECTORIES)
	
		foreach(NBL_TARGET IN LISTS NBL_TARGETS)
			get_target_property(_INTERNAL_BR_OUTPUT_INCLUDE_SEARCH_DIRECTORY_ ${NBL_TARGET} BUILTIN_RESOURCES_INCLUDE_SEARCH_DIRECTORY)
			target_include_directories(${NBL_TARGET} PUBLIC "${_NABLA_INCLUDE_DIRECTORIES_}")
			target_include_directories(Nabla PUBLIC "${_INTERNAL_BR_OUTPUT_INCLUDE_SEARCH_DIRECTORY_}")
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
endif()

target_compile_definitions(Nabla 
	PUBLIC _NABLA_INSTALL_DIR_="${CMAKE_INSTALL_PREFIX}"
)

# generate Nabla definitions to a header
glue_source_definitions(Nabla _NBL_SOURCE_DEFINITIONS_)
set(_NBL_DEFINE_FILE_WRAPPER_ ${CMAKE_CURRENT_BINARY_DIR}/include/define.h)
write_source_definitions("${_NBL_DEFINE_FILE_WRAPPER_}" "${_NBL_SOURCE_DEFINITIONS_}")

# git version tracking
target_link_libraries(Nabla PUBLIC gtml)

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

#on MSVC it won't compile without this option!
if (MSVC)
	target_compile_options(Nabla PUBLIC /bigobj)
endif()

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

if(NOT NBL_STATIC_BUILD)
	add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/generated/define.h"
		COMMAND "${CMAKE_COMMAND}" -DNBL_ROOT_PATH:PATH=${NBL_ROOT_PATH} -DNBL_WRAPPER_FILE:FILEPATH=${_NBL_DEFINE_FILE_WRAPPER_} -DNBL_GEN_DIRECTORY:PATH=${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/generated -D_NABLA_DLL_NAME_:STRING=$<PATH:REMOVE_EXTENSION,$<TARGET_FILE_NAME:Nabla>> -D_DXC_DLL_NAME_:STRING=${DXC_DLL_NAME} -D_NABLA_INSTALL_DIR_:PATH="${CMAKE_INSTALL_PREFIX}" -P ${NBL_ROOT_PATH}/cmake/scripts/nbl/nablaDefines.cmake
		COMMENT "Launching defines.h generation script!"
		VERBATIM
	)
	
	add_custom_target(generate_nabla_defines ALL
        DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/generated/define.h"
        VERBATIM
    )
	
	add_dependencies(Nabla generate_nabla_defines)
endif()

nbl_install_headers("${NABLA_HEADERS_PUBLIC}")
if(NOT NBL_STATIC_BUILD)
	nbl_install_file("${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/generated/define.h")
endif()
nbl_install_file_spec("${NBL_ROOT_PATH_BINARY}/include/nbl/builtin/builtinResources.h" "nbl/builtin")
nbl_install_config_header(BuildConfigOptions.h)

if(NBL_EMBED_BUILTIN_RESOURCES)
	nbl_install_builtin_resources(nblBuiltinResourceData)
	nbl_install_builtin_resources(spirvBuiltinResourceData)
	nbl_install_builtin_resources(boostBuiltinResourceData)
endif()

set_target_properties(Nabla PROPERTIES DEBUG_POSTFIX _debug)
set_target_properties(Nabla PROPERTIES RELWITHDEBINFO_POSTFIX _relwithdebinfo)

nbl_install_lib_spec(Nabla "nbl")

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

NBL_ADJUST_FOLDERS(src)