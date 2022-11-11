# Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
# Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
# This file is part of the "Nabla Engine".
# For conditions of distribution and use, see copyright notice in nabla.h

#[[
macro(FindLibraryAndSONAME _LIB)
  string(TOUPPER ${_LIB} _UPPERLNAME)
  string(REGEX REPLACE "\\-" "_" _LNAME "${_UPPERLNAME}")

  find_library(${_LNAME}_LIB ${_LIB})
  #message(FATAL_ERROR ${${_LNAME}_LIB})
  if(${_LNAME}_LIB)
    # reduce the library name for shared linking

    get_filename_component(_LIB_REALPATH ${${_LNAME}_LIB} REALPATH)  # resolves symlinks
    get_filename_component(_LIB_JUSTNAME ${_LIB_REALPATH} NAME)

    if(APPLE)
      string(REGEX REPLACE "(\\.[0-9]*)\\.[0-9\\.]*dylib$" "\\1.dylib" _LIB_REGEXD "${_LIB_JUSTNAME}")
    else()
      string(REGEX REPLACE "(\\.[0-9]*)\\.[0-9\\.]*$" "\\1" _LIB_REGEXD "${_LIB_JUSTNAME}")
    endif()

    SET(_DEBUG_FindSONAME TRUE)
    #if(_DEBUG_FindSONAME)
      #message(WARNING "DYNLIB OUTPUTVAR: ${_LIB} ... ${_LNAME}_LIB")
      message(WARNING "DYNLIB ORIGINAL LIB: ${_LIB} ... ${${_LNAME}_LIB}")
      #message(WARNING "DYNLIB REALPATH LIB: ${_LIB} ... ${_LIB_REALPATH}")
      #message(WARNING "DYNLIB JUSTNAME LIB: ${_LIB} ... ${_LIB_JUSTNAME}")
      #message(WARNING "DYNLIB REGEX'd LIB: ${_LIB} ... ${_LIB_REGEXD}")
    #endif()

    message(STATUS "dynamic lib${_LIB} -> ${_LIB_REGEXD}") 
    set(${_LNAME}_LIB_SONAME ${_LIB_REGEXD})
  endif()
endmacro()

FindLibraryAndSONAME(wayland-client)
FindLibraryAndSONAME(wayland-egl)
FindLibraryAndSONAME(wayland-cursor)
message(FATAL_ERROR "XDDDDDDD")
]]

if (NOT ANDROID)
	find_package(OpenGL REQUIRED OPTIONAL_COMPONENTS EGL)
else()
	# TODO: @AnastaZluk we have these in our own 3rdparty anyway!
	find_path(GLES3_INCLUDE_DIR GLES3/gl31.h HINTS ${ANDROID_NDK_ROOT_PATH} REQUIRED)
	if (${GLES3_INCLUDE_DIR})
		set(OpenGL_OpenGL_FOUND True)
	endif()
	find_path(EGL_INCLUDE_DIR EGL/egl.h HINTS ${ANDROID_NDK_ROOT_PATH} REQUIRED)
	if (${OPENGL_EGL_INCLUDE_DIRS})
		set(OpenGL_EGL_FOUND True)
		set(OPENGL_EGL_INCLUDE_DIRS True)
	endif()
endif()
if(UNIX AND NOT ANDROID)
	# TODO: maybe require the `-dev` packages for both? (or ship headers ourselves?)
	find_package(Wayland)
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
option(_NBL_COMPILE_WITH_MTL_LOADER_ "Compile with MTL Loader" ON)
option(_NBL_COMPILE_WITH_OBJ_LOADER_ "Compile with OBJ Loader" ON)
#option(_NBL_COMPILE_WITH_OBJ_WRITER_ "Compile with OBJ Writer" ON) uncomment when writer exists
option(_NBL_COMPILE_WITH_STL_LOADER_ "Compile with STL Loader" ON)
option(_NBL_COMPILE_WITH_STL_WRITER_ "Compile with STL Writer" ON)
option(_NBL_COMPILE_WITH_PLY_LOADER_ "Compile with PLY Loader" ON)
option(_NBL_COMPILE_WITH_PLY_WRITER_ "Compile with PLY Writer" ON)
option(_NBL_COMPILE_WITH_BAW_LOADER_ "Compile with BAW Loader" OFF)
option(_NBL_COMPILE_WITH_BAW_WRITER_ "Compile with BAW Writer" OFF)
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
option(_NBL_COMPILE_WITH_GLTF_LOADER_ "Compile with GLTF Loader" ON)
option(_NBL_COMPILE_WITH_GLTF_WRITER_ "Compile with GLTF Writer" ON)
set(_NBL_EG_PRFNT_LEVEL 0 CACHE STRING "EasterEgg Profanity Level")

if(NBL_BUILD_ANDROID)
	set(_NBL_COMPILE_WITH_BAW_LOADER_ OFF CACHE BOOL "Off for Android purposes at the moment" FORCE)	# TODO - remove when BAW-v4 is done
	set(_NBL_COMPILE_WITH_BAW_WRITER_ OFF CACHE BOOL "Off for Android purposes at the moment" FORCE)	# TODO - remove when BAW-v4 is done
	set(NBL_BUILD_MITSUBA_LOADER OFF CACHE BOOL "Android doesn't need this loader, if you want it, pay us.'" FORCE)
endif()

nbl_get_conf_dir(NABLA_CONF_DIR_DEBUG Debug)
nbl_get_conf_dir(NABLA_CONF_DIR_RELEASE Release)
if(NBL_EMBED_BUILTIN_RESOURCES)	
	add_subdirectory(builtin)
	nbl_add_builtin_resources("${nbl_resources_to_embed}")
endif()
nbl_get_conf_dir(NABLA_CONF_DIR_RELWITHDEBINFO RelWithDebInfo)

set(_NBL_COMPILE_WITH_OPENGL_ ${NBL_COMPILE_WITH_OPENGL})
if (NBL_COMPILE_WITH_CUDA)
	message(STATUS "Building with CUDA interop")
	set(_NBL_COMPILE_WITH_CUDA_ ${NBL_COMPILE_WITH_CUDA})
	if (NBL_BUILD_OPTIX)
		set(_NBL_BUILD_OPTIX_ ${NBL_BUILD_OPTIX})
	endif()
endif()

set(_NBL_EMBED_BUILTIN_RESOURCES_ ${NBL_EMBED_BUILTIN_RESOURCES})

#set(_NBL_TARGET_ARCH_ARM_ ${NBL_TARGET_ARCH_ARM}) #uncomment in the future

# TODO: this is a mess and not the right way to about it
if (WAYLAND_CLIENT_FOUND AND OpenGL_EGL_FOUND)
	set(_NBL_BUILD_WITH_WAYLAND 1)
	#message(FATAL_ERROR "found wayland! ${WAYLAND_CLIENT_LIBRARY}")
else()
	set(_NBL_BUILD_WITH_WAYLAND 0)
	#message(FATAL_ERROR "NOT found wayland!")
endif()

set(__NBL_FAST_MATH ${NBL_FAST_MATH})
set(_NBL_DEBUG 0)
set(_NBL_RELWITHDEBINFO 0)
configure_file("${NBL_ROOT_PATH}/include/nbl/config/BuildConfigOptions.h.in" "${NABLA_CONF_DIR_RELEASE}/BuildConfigOptions.h")
set(_NBL_DEBUG 0)
set(_NBL_RELWITHDEBINFO 1)
configure_file("${NBL_ROOT_PATH}/include/nbl/config/BuildConfigOptions.h.in" "${NABLA_CONF_DIR_RELWITHDEBINFO}/BuildConfigOptions.h")
set(_NBL_DEBUG 1)
set(_NBL_RELWITHDEBINFO 0)
configure_file("${NBL_ROOT_PATH}/include/nbl/config/BuildConfigOptions.h.in" "${NABLA_CONF_DIR_DEBUG}/BuildConfigOptions.h")

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
	${NBL_ROOT_PATH}/src/nbl/core/IReferenceCounted.cpp
)
set(NBL_SYSTEM_SOURCES
	${NBL_ROOT_PATH}/src/nbl/system/IFileBase.cpp
	${NBL_ROOT_PATH}/src/nbl/system/ILogger.cpp
	${NBL_ROOT_PATH}/src/nbl/system/CArchiveLoaderZip.cpp
	${NBL_ROOT_PATH}/src/nbl/system/CArchiveLoaderTar.cpp
	${NBL_ROOT_PATH}/src/nbl/system/CAPKResourcesArchive.cpp
	${NBL_ROOT_PATH}/src/nbl/system/ISystem.cpp
	${NBL_ROOT_PATH}/src/nbl/system/IFileArchive.cpp
	${NBL_ROOT_PATH}/src/nbl/system/CStdoutLoggerAndroid.cpp
	${NBL_ROOT_PATH}/src/nbl/system/CFileViewVirtualAllocatorWin32.cpp
	${NBL_ROOT_PATH}/src/nbl/system/CFileViewVirtualAllocatorPOSIX.cpp
	${NBL_ROOT_PATH}/src/nbl/system/CFileViewAPKAllocator.cpp
	${NBL_ROOT_PATH}/src/nbl/system/CFileWin32.cpp
	${NBL_ROOT_PATH}/src/nbl/system/CFilePOSIX.cpp
	${NBL_ROOT_PATH}/src/nbl/system/CSystemWin32.cpp
	${NBL_ROOT_PATH}/src/nbl/system/CSystemAndroid.cpp
	${NBL_ROOT_PATH}/src/nbl/system/CSystemLinux.cpp
)
set(NBL_UI_SOURCES
	${NBL_ROOT_PATH}/src/nbl/ui/CWindowWin32.cpp
	${NBL_ROOT_PATH}/src/nbl/ui/CWindowX11.cpp
	${NBL_ROOT_PATH}/src/nbl/ui/CWindowWayland.cpp
	${NBL_ROOT_PATH}/src/nbl/ui/CWaylandCaller.cpp
	${NBL_ROOT_PATH}/src/nbl/ui/CCursorControlWin32.cpp
	${NBL_ROOT_PATH}/src/nbl/ui/CClipboardManagerWin32.cpp
	${NBL_ROOT_PATH}/src/nbl/ui/CWindowManagerAndroid.cpp
	${NBL_ROOT_PATH}/src/nbl/ui/CGraphicalApplicationAndroid.cpp
)
set(NBL_ASSET_SOURCES	
# Assets
	${NBL_ROOT_PATH}/src/nbl/asset/IAsset.cpp
	${NBL_ROOT_PATH}/src/nbl/asset/IAssetManager.cpp
	${NBL_ROOT_PATH}/src/nbl/asset/interchange/IAssetWriter.cpp
	${NBL_ROOT_PATH}/src/nbl/asset/interchange/IAssetLoader.cpp
	${NBL_ROOT_PATH}/src/nbl/asset/interchange/IRenderpassIndependentPipelineLoader.cpp
	
# Shaders
	${NBL_ROOT_PATH}/src/nbl/asset/utils/ISPIRVOptimizer.cpp
	${NBL_ROOT_PATH}/src/nbl/asset/utils/IShaderCompiler.cpp
	${NBL_ROOT_PATH}/src/nbl/asset/utils/CGLSLCompiler.cpp
	${NBL_ROOT_PATH}/src/nbl/asset/utils/CHLSLCompiler.cpp
	${NBL_ROOT_PATH}/src/nbl/asset/utils/CCompilerSet.cpp
	${NBL_ROOT_PATH}/src/nbl/asset/utils/CSPIRVIntrospector.cpp
	${NBL_ROOT_PATH}/src/nbl/asset/interchange/CGLSLLoader.cpp
	${NBL_ROOT_PATH}/src/nbl/asset/interchange/CSPVLoader.cpp
	
# Pipeline loaders
	${NBL_ROOT_PATH}/src/nbl/asset/interchange/CGraphicsPipelineLoaderMTL.cpp

# Meshes
	${NBL_ROOT_PATH}/src/nbl/asset/utils/CForsythVertexCacheOptimizer.cpp
	${NBL_ROOT_PATH}/src/nbl/asset/utils/CSmoothNormalGenerator.cpp
	${NBL_ROOT_PATH}/src/nbl/asset/utils/CGeometryCreator.cpp
	${NBL_ROOT_PATH}/src/nbl/asset/utils/CMeshManipulator.cpp
	${NBL_ROOT_PATH}/src/nbl/asset/utils/COverdrawMeshOptimizer.cpp
	${NBL_ROOT_PATH}/src/nbl/asset/utils/CSmoothNormalGenerator.cpp

# Mesh loaders
#	${NBL_ROOT_PATH}/src/nbl/asset/bawformat/CBAWMeshFileLoader.cpp
	${NBL_ROOT_PATH}/src/nbl/asset/interchange/COBJMeshFileLoader.cpp
	${NBL_ROOT_PATH}/src/nbl/asset/interchange/CPLYMeshFileLoader.cpp
	${NBL_ROOT_PATH}/src/nbl/asset/interchange/CSTLMeshFileLoader.cpp
	${NBL_ROOT_PATH}/src/nbl/asset/interchange/CBufferLoaderBIN.cpp
	${NBL_ROOT_PATH}/src/nbl/asset/interchange/CGLTFLoader.cpp

# Mesh writers
#	${NBL_ROOT_PATH}/src/nbl/asset/bawformat/CBAWMeshWriter.cpp
	${NBL_ROOT_PATH}/src/nbl/asset/interchange/CPLYMeshWriter.cpp
	${NBL_ROOT_PATH}/src/nbl/asset/interchange/CSTLMeshWriter.cpp
	${NBL_ROOT_PATH}/src/nbl/asset/interchange/CGLTFWriter.cpp

# BaW Format
#	${NBL_ROOT_PATH}/src/nbl/asset/bawformat/TypedBlob.cpp
#	${NBL_ROOT_PATH}/src/nbl/asset/bawformat/CBAWFile.cpp
#	${NBL_ROOT_PATH}/src/nbl/asset/bawformat/legacy/CBAWLegacy.cpp
#	${NBL_ROOT_PATH}/src/nbl/asset/bawformat/legacy/CBAWVersionUpFunctions.cpp
#	${NBL_ROOT_PATH}/src/nbl/asset/bawformat/CBlobsLoadingManager.cpp

# Images
	${NBL_ROOT_PATH}/src/nbl/asset/interchange/IImageAssetHandlerBase.cpp
	${NBL_ROOT_PATH}/src/nbl/asset/filters/CBasicImageFilterCommon.cpp
	${NBL_ROOT_PATH}/src/nbl/asset/utils/CDerivativeMapCreator.cpp

# Image loaders
	${NBL_ROOT_PATH}/src/nbl/asset/interchange/IImageLoader.cpp
	${NBL_ROOT_PATH}/src/nbl/asset/interchange/CImageLoaderJPG.cpp
	${NBL_ROOT_PATH}/src/nbl/asset/interchange/CImageLoaderPNG.cpp
	${NBL_ROOT_PATH}/src/nbl/asset/interchange/CImageLoaderTGA.cpp
	${NBL_ROOT_PATH}/src/nbl/asset/interchange/CImageLoaderOpenEXR.cpp
	${NBL_ROOT_PATH}/src/nbl/asset/interchange/CGLILoader.cpp

# Image writers
	${NBL_ROOT_PATH}/src/nbl/asset/interchange/IImageWriter.cpp
	${NBL_ROOT_PATH}/src/nbl/asset/interchange/CImageWriterJPG.cpp
	${NBL_ROOT_PATH}/src/nbl/asset/interchange/CImageWriterPNG.cpp
	${NBL_ROOT_PATH}/src/nbl/asset/interchange/CImageWriterTGA.cpp
	${NBL_ROOT_PATH}/src/nbl/asset/interchange/CImageWriterOpenEXR.cpp
	${NBL_ROOT_PATH}/src/nbl/asset/interchange/CGLIWriter.cpp
	
# Material compiler
	${NBL_ROOT_PATH}/src/nbl/asset/material_compiler/CMaterialCompilerGLSLBackendCommon.cpp
	${NBL_ROOT_PATH}/src/nbl/asset/material_compiler/CMaterialCompilerGLSLRasterBackend.cpp
)
set(NBL_VIDEO_SOURCES
# Allocators
	${NBL_ROOT_PATH}/src/nbl/video/alloc/CSimpleBufferAllocator.cpp

# Utilities
	${NBL_ROOT_PATH}/src/nbl/video/utilities/IDescriptorSetCache.cpp
	${NBL_ROOT_PATH}/src/nbl/video/utilities/ICommandPoolCache.cpp
	${NBL_ROOT_PATH}/src/nbl/video/utilities/IPropertyPool.cpp
	${NBL_ROOT_PATH}/src/nbl/video/utilities/IUtilities.cpp
	${NBL_ROOT_PATH}/src/nbl/video/utilities/CPropertyPoolHandler.cpp
	${NBL_ROOT_PATH}/src/nbl/video/utilities/CScanner.cpp
	${NBL_ROOT_PATH}/src/nbl/video/utilities/CComputeBlit.cpp

	${NBL_ROOT_PATH}/src/nbl/video/IAPIConnection.cpp
	${NBL_ROOT_PATH}/src/nbl/video/IPhysicalDevice.cpp
	${NBL_ROOT_PATH}/src/nbl/video/ILogicalDevice.cpp
	${NBL_ROOT_PATH}/src/nbl/video/IGPUFence.cpp
	${NBL_ROOT_PATH}/src/nbl/video/IDeviceMemoryAllocation.cpp
	${NBL_ROOT_PATH}/src/nbl/video/IDeviceMemoryBacked.cpp

# OpenGL
	${NBL_ROOT_PATH}/src/nbl/video/IOpenGL_FunctionTable.cpp
	${NBL_ROOT_PATH}/src/nbl/video/COpenGLPipelineCache.cpp
	${NBL_ROOT_PATH}/src/nbl/video/COpenGLImage.cpp
	${NBL_ROOT_PATH}/src/nbl/video/COpenGLImageView.cpp
	${NBL_ROOT_PATH}/src/nbl/video/COpenGLBuffer.cpp
	${NBL_ROOT_PATH}/src/nbl/video/COpenGLBufferView.cpp
	${NBL_ROOT_PATH}/src/nbl/video/COpenGLSpecializedShader.cpp
	${NBL_ROOT_PATH}/src/nbl/video/COpenGLSampler.cpp
	${NBL_ROOT_PATH}/src/nbl/video/COpenGLCommandBuffer.cpp
	${NBL_ROOT_PATH}/src/nbl/video/SOpenGLContextLocalCache.cpp
	${NBL_ROOT_PATH}/src/nbl/video/COpenGLSync.cpp
	${NBL_ROOT_PATH}/src/nbl/video/COpenGLShader.cpp
	${NBL_ROOT_PATH}/src/nbl/video/COpenGLRenderpassIndependentPipeline.cpp
	${NBL_ROOT_PATH}/src/nbl/video/COpenGLFramebuffer.cpp
	${NBL_ROOT_PATH}/src/nbl/video/COpenGL_Connection.cpp
	${NBL_ROOT_PATH}/src/nbl/video/COpenGLComputePipeline.cpp
	${NBL_ROOT_PATH}/src/nbl/video/COpenGLQueryPool.cpp
	${NBL_ROOT_PATH}/src/nbl/video/COpenGL_Swapchain.cpp
	${NBL_ROOT_PATH}/src/nbl/video/debug/COpenGLDebugCallback.cpp
	
# Vulkan
	${NBL_ROOT_PATH}/src/nbl/video/CVulkanSwapchain.cpp
	${NBL_ROOT_PATH}/src/nbl/video/CVulkanFence.cpp
	${NBL_ROOT_PATH}/src/nbl/video/CVulkanImage.cpp
	${NBL_ROOT_PATH}/src/nbl/video/CVulkanQueue.cpp
	${NBL_ROOT_PATH}/src/nbl/video/CVulkanSemaphore.cpp
	${NBL_ROOT_PATH}/src/nbl/video/CVulkanFramebuffer.cpp
	${NBL_ROOT_PATH}/src/nbl/video/CVulkanRenderpass.cpp
	${NBL_ROOT_PATH}/src/nbl/video/CVulkanImageView.cpp
	${NBL_ROOT_PATH}/src/nbl/video/CVulkanCommandBuffer.cpp
	${NBL_ROOT_PATH}/src/nbl/video/CVulkanShader.cpp
	${NBL_ROOT_PATH}/src/nbl/video/CVulkanCommandPool.cpp
	${NBL_ROOT_PATH}/src/nbl/video/CVulkanBuffer.cpp
	${NBL_ROOT_PATH}/src/nbl/video/CVulkanDescriptorSetLayout.cpp
	${NBL_ROOT_PATH}/src/nbl/video/CVulkanPipelineLayout.cpp
	${NBL_ROOT_PATH}/src/nbl/video/CVulkanPipelineCache.cpp
	${NBL_ROOT_PATH}/src/nbl/video/CVulkanComputePipeline.cpp
	${NBL_ROOT_PATH}/src/nbl/video/CVulkanDescriptorPool.cpp
	${NBL_ROOT_PATH}/src/nbl/video/CVulkanMemoryAllocation.cpp
	${NBL_ROOT_PATH}/src/nbl/video/CVulkanBufferView.cpp
	${NBL_ROOT_PATH}/src/nbl/video/CVulkanLogicalDevice.cpp
	${NBL_ROOT_PATH}/src/nbl/video/CVulkanDeferredOperation.cpp
	${NBL_ROOT_PATH}/src/nbl/video/CVulkanAccelerationStructure.cpp
	${NBL_ROOT_PATH}/src/nbl/video/CVulkanQueryPool.cpp
	${NBL_ROOT_PATH}/src/nbl/video/CVulkanSampler.cpp
	${NBL_ROOT_PATH}/src/nbl/video/CVulkanConnection.cpp
	${NBL_ROOT_PATH}/src/nbl/video/CVulkanGraphicsPipeline.cpp
	${NBL_ROOT_PATH}/src/nbl/video/CVulkanEvent.cpp
	${NBL_ROOT_PATH}/src/nbl/video/CSurfaceVulkan.cpp
	
# CUDA
	${NBL_ROOT_PATH}/src/nbl/video/CCUDAHandler.cpp
	${NBL_ROOT_PATH}/src/nbl/video/CCUDADevice.cpp
)

set(NBL_SCENE_SOURCES
	${NBL_ROOT_PATH}/src/nbl/scene/ITransformTree.cpp
)

set(NABLA_SRCS_COMMON
	${NBL_CORE_SOURCES}
	${NBL_SYSTEM_SOURCES}
	${NBL_UI_SOURCES}
	${NBL_ASSET_SOURCES}
	${NBL_VIDEO_SOURCES}
	${NBL_SCENE_SOURCES}
)

if(MSVC)
	list(APPEND NABLA_SRCS_COMMON "${PROJECT_SOURCE_DIR}/tools/debug/VisualStudio/DynamicArrayVisualizer.natvis")
endif()

if (NBL_EMBED_BUILTIN_RESOURCES)
	list(APPEND NABLA_SRCS_COMMON "${CMAKE_BINARY_DIR}/src/nbl/builtin/builtinResourceData.cpp")
endif()

set(COMMON_INCLUDE_DIRS
	${THIRD_PARTY_SOURCE_DIR}/glm
        ${THIRD_PARTY_SOURCE_DIR}/renderdoc # for renderdoc api header
	${CMAKE_BINARY_DIR}/3rdparty/zlib #for dynamically generated zconf.h
	${CMAKE_BINARY_DIR}/3rdparty/openexr/IlmBase/config #for dynamically generated IlmBaseConfig.h
	${CMAKE_BINARY_DIR}/3rdparty/openexr/OpenEXR/config #for dynamically generated OpenEXRConfig.h
	$<TARGET_PROPERTY:png_static,BINARY_DIR> #for dynamically generated pnglibconf.h
	$<TARGET_PROPERTY:jpeg-static,BINARY_DIR> #for dynamically generated jconfig.h
)

# just gather all the header files and later put them into project so it's easy to search for things using IDE
file(GLOB_RECURSE NABLA_HEADERS_PRIV1 "*.h")
file(GLOB_RECURSE NABLA_HEADERS_PRIV2 "${NBL_ROOT_PATH}/src/nbl/*.h")

#always install builtins (some may be included in cpp regardless if embedded or not)
foreach(X IN LISTS nbl_resources_to_embed)
	list(APPEND NABLA_RESOURCES_TO_EMBED_PUBLIC ${NBL_ROOT_PATH}/include/${X})
endforeach()
list(APPEND NABLA_HEADERS_PUBLIC ${NABLA_RESOURCES_TO_EMBED_PUBLIC})
source_group("Resources to embed" FILES ${NABLA_RESOURCES_TO_EMBED_PUBLIC})

set(NABLA_HEADERS "${NBL_ROOT_PATH}/include/nabla.h" ${NABLA_HEADERS_PUBLIC} ${NABLA_HEADERS_PIRV1} ${NABLA_HEADERS_PRIV2})

set(NBL_LIBRARY_CREATION_SOURCES 
	${NABLA_SRCS_COMMON}
	${NABLA_HEADERS}
	$<TARGET_OBJECTS:aesGladman>
	$<TARGET_OBJECTS:bzip2>
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

nbl_adjust_flags() # macro defined in root CMakeLists
nbl_adjust_definitions() # macro defined in root CMakeLists

if(NBL_DYNAMIC_MSVC_RUNTIME)
	set_property(TARGET Nabla PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
else()
	set_property(TARGET Nabla PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

target_compile_definitions(Nabla PRIVATE __NBL_BUILDING_NABLA__)

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

if(NOT NBL_STATIC_BUILD)
	set(NBL_ASSEMBLY_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/devshgraphicsprogramming.nabla" CACHE INTERNAL "" FORCE)	
	set_target_properties(Nabla PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${NBL_ASSEMBLY_DIRECTORY})
endif()

## Set up 3rdparty deps
# Parallel Hashmap
target_include_directories(Nabla PUBLIC ${THIRD_PARTY_SOURCE_DIR}/parallel-hashmap/parallel_hashmap)
# oneDPL for clang C++17 parallelism
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
	add_dependencies(Nabla oneDPL)
	target_include_directories(Nabla PUBLIC ${ONETBB_INSTALL_DIR}/include)
	target_include_directories(Nabla PUBLIC ${ONEDPL_INSTALL_DIR}/include)
	target_link_libraries(Nabla INTERFACE tbb tbbmalloc tbbmalloc_proxy)
endif()

if (UNIX)
	target_link_libraries(Nabla INTERFACE 
		${CMAKE_THREAD_LIBS_INIT}
		${CMAKE_DL_LIBS}
	)
endif()
# OpenSSL
if(ANDROID AND NOT "${CMAKE_HOST_SYSTEM_NAME}" STREQUAL "Windows") # OpenSSL can't be built for Android platform using windows OS host
	add_dependencies(Nabla openssl_build)
	target_link_libraries(Nabla INTERFACE "$<TARGET_PROPERTY:OpenSSL,INTERFACE_LINK_LIBRARIES>")
	target_include_directories(Nabla PUBLIC "$<TARGET_PROPERTY:OpenSSL,BINARY_DIR>/build/include")
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
target_include_directories(Nabla PUBLIC BEFORE ${THIRD_PARTY_SOURCE_DIR}/SPIRV-Tools/include)
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
	add_dependencies(Nabla OpenEXR::IlmImf)
	if(NBL_STATIC_BUILD)
		target_link_libraries(Nabla INTERFACE OpenEXR::IlmImf)
	else()
		target_link_libraries(Nabla PRIVATE OpenEXR::IlmImf)
	endif()
	target_include_directories(
		Nabla PUBLIC ${THIRD_PARTY_SOURCE_DIR}/openexr/OpenEXR/IlmImf
		Nabla PUBLIC ${THIRD_PARTY_SOURCE_DIR}/openexr/IlmBase/Half
		Nabla PUBLIC ${THIRD_PARTY_SOURCE_DIR}/openexr/IlmBase/Iex
		Nabla PUBLIC ${THIRD_PARTY_SOURCE_DIR}/openexr/IlmBase/IexMath
		Nabla PUBLIC ${THIRD_PARTY_SOURCE_DIR}/openexr/IlmBase/Imath
		Nabla PUBLIC ${THIRD_PARTY_SOURCE_DIR}/openexr/IlmBase/IlmThread						   
	)
endif()
if(_NBL_COMPILE_WITH_GLI_)
	add_dependencies(Nabla gli)
	if(NBL_STATIC_BUILD)
		target_link_libraries(Nabla INTERFACE gli)
	else()
		target_link_libraries(Nabla PRIVATE gli)
	endif()
	target_include_directories(Nabla PUBLIC ${THIRD_PARTY_SOURCE_DIR}/gli)
endif()

# DXC
if(_NBL_COMPILE_WITH_DXC_)
	add_dependencies(Nabla dxc)
	# TODO
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
# OpenGL (EGL)
target_link_libraries(Nabla PRIVATE egl)
if (ANDROID)
	target_compile_definitions(Nabla PUBLIC $<$<BOOL:${OpenGL_EGL_FOUND}>:WL_EGL_PLATFORM>)
endif()

# Include dirs for self
target_include_directories(Nabla PUBLIC 
	${CMAKE_CURRENT_BINARY_DIR}/include
	${NBL_ROOT_PATH}/include
	${COMMON_INCLUDE_DIRS}
	${THIRD_PARTY_SOURCE_DIR}
	#those also goes as PUBLIC because of examples
	"$<$<CONFIG:DEBUG>:${NABLA_CONF_DIR_DEBUG}>"
	"$<$<CONFIG:RELEASE>:${NABLA_CONF_DIR_RELEASE}>"
	"$<$<CONFIG:RELWITHDEBINFO>:${NABLA_CONF_DIR_RELWITHDEBINFO}>"
	"$<$<BOOL:${OpenGL_EGL_FOUND}>:${OPENGL_EGL_INCLUDE_DIRS}>"
	"$<$<BOOL:${WAYLAND_CLIENT_FOUND}>:${WAYLAND_CLIENT_INCLUDE_DIR}>"
)

target_include_directories(Nabla PUBLIC ${NBL_ROOT_PATH}/src ${NBL_ROOT_PATH}/source/Nabla ${NBL_ROOT_PATH}/src/3rdparty) # this should be PRIVATE, but things from /src (or /source) are sometimes included in things in /include and so examples have to put source dirs into theirs Include Path
target_include_directories(Nabla PRIVATE ${CMAKE_BINARY_DIR}/include)

target_compile_definitions(Nabla 
	PRIVATE EGL_STATIC SPIRV_CROSS_EXCEPTIONS_TO_ASSERTIONS
	INTERFACE _NBL_PCH_IGNORE_PRIVATE_HEADERS
)

# generate Nabla definitions to a header
glue_source_definitions(Nabla _NBL_SOURCE_DEFINITIONS_)
set(_NBL_DEFINE_FILE_WRAPPER_ ${CMAKE_CURRENT_BINARY_DIR}/include/define.h)
write_source_definitions("${_NBL_DEFINE_FILE_WRAPPER_}" "${_NBL_SOURCE_DEFINITIONS_}")

#on MSVC it won't compile without this option!
if (MSVC)
	target_compile_options(Nabla PUBLIC /bigobj)
endif()

#precompiled headers
if(NBL_PCH)
	target_precompile_headers(Nabla PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/pch.h")
endif()

if(NBL_EMBED_BUILTIN_RESOURCES)
	add_custom_target(builtin_resources DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/builtin/builtinResourceData.cpp)
	add_dependencies(Nabla builtin_resources)
endif()

# extensions
start_tracking_variables_for_propagation_to_parent()
add_subdirectory(ext)
propagate_changed_variables_to_parent_scope()

if(NOT NBL_STATIC_BUILD)
	add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/generated/define.h"
		COMMAND "${CMAKE_COMMAND}" -DNBL_ROOT_PATH:PATH=${NBL_ROOT_PATH} -DNBL_WRAPPER_FILE:FILEPATH=${_NBL_DEFINE_FILE_WRAPPER_} -DNBL_GEN_DIRECTORY:PATH=${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/generated -D_NABLA_DLL_NAME_:STRING=$<TARGET_FILE_NAME:Nabla> -D_NABLA_INSTALL_DIR_:PATH="${CMAKE_INSTALL_PREFIX}" -P ${NBL_ROOT_PATH}/cmake/scripts/nbl/nablaDefines.cmake
		COMMENT "Launching defines.h generation script!"
		VERBATIM
	)
	
	add_custom_target(generate_define_header ALL DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/generated/define.h")
	add_dependencies(Nabla generate_define_header)
endif()

nbl_install_headers("${NABLA_HEADERS_PUBLIC}" "${NBL_ROOT_PATH}/include")
nbl_install_file("${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/generated/define.h" "")
nbl_install_config_header(BuildConfigOptions.h)

# TODO: @AnastazIuk change macros to functions! https://github.com/buildaworldnet/IrrlichtBAW/issues/311
macro(nbl_install_program _TRGT)
	install(PROGRAMS $<TARGET_FILE:${_TRGT}> DESTINATION lib CONFIGURATIONS Release)
	install(PROGRAMS $<TARGET_FILE:${_TRGT}> DESTINATION debug/lib CONFIGURATIONS Debug)
	install(PROGRAMS $<TARGET_FILE:${_TRGT}> DESTINATION relwithdebinfo/lib CONFIGURATIONS RelWithDebInfo)
endmacro()
macro(nbl_install_lib _TRGT)
	install(TARGETS ${_TRGT} ARCHIVE DESTINATION lib CONFIGURATIONS Release)
	install(TARGETS ${_TRGT} ARCHIVE DESTINATION debug/lib CONFIGURATIONS Debug)
	install(TARGETS ${_TRGT} ARCHIVE DESTINATION relwithdebinfo/lib CONFIGURATIONS RelWithDebInfo)
endmacro()

set_target_properties(Nabla PROPERTIES DEBUG_POSTFIX _debug)
set_target_properties(Nabla PROPERTIES RELWITHDEBINFO_POSTFIX _relwithdebinfo)

nbl_install_lib(Nabla)

if(NBL_STATIC_BUILD)
	# install dependencies, they are required only for static Nabla builds
	nbl_install_lib(egl)
	nbl_install_lib(glslang)
	nbl_install_lib(GenericCodeGen)
	nbl_install_lib(MachineIndependent)
	nbl_install_lib(HLSL)
	nbl_install_lib(jpeg-static)
	if (_NBL_COMPILE_WITH_OPEN_EXR_)
		nbl_install_lib(Half)
		nbl_install_lib(Iex)
		nbl_install_lib(IexMath)
		nbl_install_lib(IlmImf)
		nbl_install_lib(IlmThread)
		nbl_install_lib(Imath)
	endif()
	nbl_install_lib(png_static)

	if(ANDROID AND NOT "${CMAKE_HOST_SYSTEM_NAME}" STREQUAL "Windows") # OpenSSL can't be built for Android platform using windows OS host
		install(FILES
				"$<TARGET_PROPERTY:OpenSSL,INTERFACE_LINK_LIBRARIES>"
			DESTINATION lib
		)
	endif()
	nbl_install_lib(shaderc)
	nbl_install_lib(shaderc_util)
	nbl_install_lib(SPIRV)
	nbl_install_lib(SPIRV-Tools-static) # TODO: make this function/macro work with alias target
	nbl_install_lib(SPIRV-Tools-opt)
	nbl_install_lib(OGLCompiler)
	nbl_install_lib(OSDependent)
	nbl_install_lib(zlibstatic)
	nbl_install_lib(simdjson)
	nbl_install_lib(volk)
	if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
		nbl_install_program(tbb)
		nbl_install_program(tbbmalloc)
		nbl_install_program(tbbmalloc_proxy)
	endif()
else()
	nbl_install_program(Nabla)
endif()
