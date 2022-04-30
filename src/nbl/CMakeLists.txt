# Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
# This file is part of the "Nabla Engine".
# For conditions of distribution and use, see copyright notice in nabla.h

find_package(OpenGL REQUIRED)
if(UNIX)
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
option(_NBL_COMPILE_WITH_BAW_LOADER_ "Compile with BAW Loader" ON)
option(_NBL_COMPILE_WITH_BAW_WRITER_ "Compile with BAW Writer" ON)
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

nbl_get_conf_dir(NABLA_CONF_DIR_DEBUG Debug)
nbl_get_conf_dir(NABLA_CONF_DIR_RELEASE Release)
if(NBL_EMBED_BUILTIN_RESOURCES)	
	add_subdirectory(builtin)
	nbl_add_builtin_resources("${nbl_resources_to_embed}")
endif()
nbl_get_conf_dir(NABLA_CONF_DIR_RELWITHDEBINFO RelWithDebInfo)

set(_NBL_COMPILE_WITH_OPENGL_ ${NBL_COMPILE_WITH_OPENGL})
if (NBL_COMPILE_WITH_OPENCL)
	message(STATUS "Building with OpenCL interop")
		
	set(_NBL_COMPILE_WITH_OPENCL_ ${NBL_COMPILE_WITH_OPENCL})
endif()
if (NBL_COMPILE_WITH_CUDA)
	message(STATUS "Building with CUDA interop")
	set(_NBL_COMPILE_WITH_CUDA_ ${NBL_COMPILE_WITH_CUDA})
	if (NBL_BUILD_OPTIX)
		set(_NBL_BUILD_OPTIX_ ${NBL_BUILD_OPTIX})
	endif()
endif()

set(_NBL_EMBED_BUILTIN_RESOURCES_ ${NBL_EMBED_BUILTIN_RESOURCES})

#set(_NBL_TARGET_ARCH_ARM_ ${NBL_TARGET_ARCH_ARM}) #uncomment in the future
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

set(NABLA_HEADERS_PUBLIC
	"${NBL_ROOT_PATH}/source/Nabla/FW_Mutex.h"
	"${NBL_ROOT_PATH}/source/Nabla/os.h"
	"${NBL_ROOT_PATH}/source/Nabla/CLogger.h" 
)
file(GLOB_RECURSE TEMP_GLOB_RES "${NBL_ROOT_PATH}/include/*.h")
set(NABLA_HEADERS_PUBLIC ${NABLA_HEADERS_PUBLIC} ${TEMP_GLOB_RES})
file(GLOB_RECURSE TEMP_GLOB_RES "${NBL_ROOT_PATH}/include/*.tcc")
set(NABLA_HEADERS_PUBLIC ${NABLA_HEADERS_PUBLIC} ${TEMP_GLOB_RES})

file(GLOB_RECURSE NABLA_HEADERS_PRIV1 "*.h")
file(GLOB_RECURSE NABLA_HEADERS_PRIV2 "${NBL_ROOT_PATH}/src/nbl/*.h")
# just gather all the header files and later put them into project so it's easy to search for things using IDE
set(NABLA_HEADERS ${NABLA_HEADERS_PUBLIC} ${NABLA_HEADERS_PIRV1} ${NABLA_HEADERS_PRIV2})


set(NBL_CORE_SOURCES
	${NBL_ROOT_PATH}/src/nbl/core/IReferenceCounted.cpp
# Core Memory
	${NBL_ROOT_PATH}/src/nbl/core/memory/CLeakDebugger.cpp
)
set(NBL_SYSTEM_SOURCES
# Junk to refactor
	${NBL_ROOT_PATH}/source/Nabla/CFileList.cpp
	${NBL_ROOT_PATH}/source/Nabla/CFileSystem.cpp
	${NBL_ROOT_PATH}/source/Nabla/CLimitReadFile.cpp
	${NBL_ROOT_PATH}/source/Nabla/CMemoryFile.cpp
	${NBL_ROOT_PATH}/source/Nabla/CReadFile.cpp
	${NBL_ROOT_PATH}/source/Nabla/CWriteFile.cpp
	${NBL_ROOT_PATH}/source/Nabla/CMountPointReader.cpp
	${NBL_ROOT_PATH}/source/Nabla/CPakReader.cpp
	${NBL_ROOT_PATH}/source/Nabla/CTarReader.cpp
	${NBL_ROOT_PATH}/source/Nabla/CZipReader.cpp
	${NBL_ROOT_PATH}/source/Nabla/CLogger.cpp
	${NBL_ROOT_PATH}/source/Nabla/COSOperator.cpp
	${NBL_ROOT_PATH}/source/Nabla/os.cpp
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
	${NBL_ROOT_PATH}/src/nbl/asset/utils/IGLSLCompiler.cpp
	${NBL_ROOT_PATH}/src/nbl/asset/utils/CShaderIntrospector.cpp
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
	${NBL_ROOT_PATH}/src/nbl/asset/bawformat/CBAWMeshFileLoader.cpp
	${NBL_ROOT_PATH}/src/nbl/asset/interchange/COBJMeshFileLoader.cpp
	${NBL_ROOT_PATH}/src/nbl/asset/interchange/CPLYMeshFileLoader.cpp
	${NBL_ROOT_PATH}/src/nbl/asset/interchange/CSTLMeshFileLoader.cpp
	${NBL_ROOT_PATH}/src/nbl/asset/interchange/CBufferLoaderBIN.cpp

# Mesh writers
	${NBL_ROOT_PATH}/src/nbl/asset/bawformat/CBAWMeshWriter.cpp
	${NBL_ROOT_PATH}/src/nbl/asset/interchange/CPLYMeshWriter.cpp
	${NBL_ROOT_PATH}/src/nbl/asset/interchange/CSTLMeshWriter.cpp

# BaW Format
	${NBL_ROOT_PATH}/src/nbl/asset/bawformat/TypedBlob.cpp
	${NBL_ROOT_PATH}/src/nbl/asset/bawformat/CBAWFile.cpp
	${NBL_ROOT_PATH}/src/nbl/asset/bawformat/legacy/CBAWLegacy.cpp
	${NBL_ROOT_PATH}/src/nbl/asset/bawformat/legacy/CBAWVersionUpFunctions.cpp
	${NBL_ROOT_PATH}/src/nbl/asset/bawformat/CBlobsLoadingManager.cpp

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
	${NBL_ROOT_PATH}/src/nbl/video/alloc/GPUMemoryAllocatorBase.cpp
	${NBL_ROOT_PATH}/src/nbl/video/alloc/SimpleGPUBufferAllocator.cpp
	
# Shaders
	${NBL_ROOT_PATH}/src/nbl/video/COpenGLSpecializedShader.cpp

# Property Pools
	${NBL_ROOT_PATH}/src/nbl/video/CPropertyPoolHandler.cpp

# Junk to refactor
${NBL_ROOT_PATH}/source/Nabla/IDriver.cpp
${NBL_ROOT_PATH}/source/Nabla/CNullDriver.cpp
	${NBL_ROOT_PATH}/source/Nabla/FW_Mutex.cpp
	${NBL_ROOT_PATH}/source/Nabla/CFPSCounter.cpp
)
set(NBL_SCENE_SOURCES
# Junk to refactor
${NBL_ROOT_PATH}/source/Nabla/CCameraSceneNode.cpp
${NBL_ROOT_PATH}/source/Nabla/CSceneManager.cpp

# Animators
${NBL_ROOT_PATH}/source/Nabla/CSceneNodeAnimatorCameraFPS.cpp
	${NBL_ROOT_PATH}/source/Nabla/CSceneNodeAnimatorCameraMaya.cpp
	${NBL_ROOT_PATH}/source/Nabla/CSceneNodeAnimatorCameraModifiedMaya.cpp
)
set(NABLA_SRCS_COMMON
	${NBL_CORE_SOURCES}
	${NBL_SYSTEM_SOURCES}
	${NBL_ASSET_SOURCES}
	${NBL_VIDEO_SOURCES}
	${NBL_SCENE_SOURCES}

# Junk to refactor
	${NBL_ROOT_PATH}/source/Nabla/IrrlichtDevice.cpp
	${NBL_ROOT_PATH}/source/Nabla/CIrrDeviceLinux.cpp
	${NBL_ROOT_PATH}/source/Nabla/CIrrDeviceConsole.cpp
	${NBL_ROOT_PATH}/source/Nabla/CIrrDeviceStub.cpp
	${NBL_ROOT_PATH}/source/Nabla/CIrrDeviceWin32.cpp
	${NBL_ROOT_PATH}/source/Nabla/Irrlicht.cpp
)

if (NBL_EMBED_BUILTIN_RESOURCES)
	list(APPEND NABLA_SRCS_COMMON "${CMAKE_BINARY_DIR}/src/nbl/builtin/builtinResourceData.cpp")
endif()


set(NABLA_SRCS_NORMAL
	# Vulkan

	#opengl
	${NBL_ROOT_PATH}/source/Nabla/COpenCLHandler.cpp
	${NBL_ROOT_PATH}/source/Nabla/COpenGLDriver.cpp
	${NBL_ROOT_PATH}/source/Nabla/COpenGLExtensionHandler.cpp
	${NBL_ROOT_PATH}/source/Nabla/COpenGLFrameBuffer.cpp
	${NBL_ROOT_PATH}/source/Nabla/COpenGLQuery.cpp
	${NBL_ROOT_PATH}/src/nbl/video/COpenGLPipelineCache.cpp
	
	# CUDA
	${NBL_ROOT_PATH}/src/nbl/video/CCUDAHandler.cpp

	# OpenCL
	${NBL_ROOT_PATH}/source/Nabla/COpenCLHandler.cpp
)

set(COMMON_INCLUDE_DIRS
	${THIRD_PARTY_SOURCE_DIR}/convertutf
	${THIRD_PARTY_SOURCE_DIR}/utfcpp/source
	${THIRD_PARTY_SOURCE_DIR}/glm
	${CMAKE_BINARY_DIR}/3rdparty/zlib #for dynamically generated zconf.h
	${CMAKE_BINARY_DIR}/3rdparty/openexr/IlmBase/config #for dynamically generated IlmBaseConfig.h
	${CMAKE_BINARY_DIR}/3rdparty/openexr/OpenEXR/config #for dynamically generated OpenEXRConfig.h
	$<TARGET_PROPERTY:png_static,BINARY_DIR> #for dynamically generated pnglibconf.h
	$<TARGET_PROPERTY:jpeg,BINARY_DIR> #for dynamically generated jconfig.h
)

nbl_adjust_flags() # macro defined in root CMakeLists
nbl_adjust_definitions() # macro defined in root CMakeLists

# TODO: Change all MACROS into FUNCTIONS (if possible, need to clarify variable scoping rules)
macro(nbl_target_link_zlib _trgt)
	add_dependencies(${_trgt} zlibstatic)

	target_link_libraries(${_trgt} INTERFACE
		$<$<CONFIG:Debug>:${ZLIB_LIBRARY_DEBUG}>
		$<$<CONFIG:Release>:${ZLIB_LIBRARY_RELEASE}>
		$<$<CONFIG:RelWithDebInfo>:${ZLIB_LIBRARY_RELWITHDEBINFO}>
	)
	target_include_directories(${_trgt} PUBLIC ${THIRD_PARTY_SOURCE_DIR}/zlib)
endmacro()

#Parallel hashmap
macro(nbl_target_include_parallel_hashmap _trgt)
	target_include_directories(${_trgt} PUBLIC ${THIRD_PARTY_SOURCE_DIR}/parallel-hashmap/parallel_hashmap)
endmacro()

macro(nbl_target_link_openssl _trgt)
	if(WIN32)
		target_link_libraries(${_trgt} INTERFACE
			"$<TARGET_PROPERTY:openssl_build,BINARY_DIR>/build/lib/libeay32.lib"
			"$<TARGET_PROPERTY:openssl_build,BINARY_DIR>/build/lib/ssleay32.lib"
		)
	else()
		target_link_libraries(${_trgt} INTERFACE
			"$<TARGET_PROPERTY:openssl_build,BINARY_DIR>/build/lib/libcrypto.a"
			"$<TARGET_PROPERTY:openssl_build,BINARY_DIR>/build/lib/libssl.a"
		)
	endif()

	target_include_directories(${_trgt} PUBLIC "$<TARGET_PROPERTY:openssl_build,BINARY_DIR>/build/include")
endmacro()
macro(nbl_target_link_shaderc _trgt)
	add_dependencies(${_trgt} shaderc)
	target_link_libraries(${_trgt} INTERFACE
		shaderc
	)
	target_include_directories(${_trgt} PUBLIC ${THIRD_PARTY_SOURCE_DIR}/shaderc/libshaderc/include)
endmacro()
macro(nbl_target_link_libjpeg _trgt)
	add_dependencies(${_trgt} jpeg)
	target_link_libraries(${_trgt} INTERFACE
		jpeg
	)
	target_include_directories(${_trgt} PUBLIC ${THIRD_PARTY_SOURCE_DIR}/libjpeg)
endmacro()
macro(nbl_target_link_libpng _trgt)
	add_dependencies(${_trgt} png_static)
	target_link_libraries(${_trgt} INTERFACE
		png_static
	)
	target_include_directories(${_trgt} PUBLIC ${THIRD_PARTY_SOURCE_DIR}/libpng)
endmacro()
macro(nbl_target_link_openexr _trgt)
	add_dependencies(${_trgt} OpenEXR::IlmImf)
	target_link_libraries(${_trgt} INTERFACE 
		OpenEXR::IlmImf
	)
	target_include_directories(${_trgt} PUBLIC ${THIRD_PARTY_SOURCE_DIR}/openexr/OpenEXR/IlmImf
							   ${_trgt} PUBLIC ${THIRD_PARTY_SOURCE_DIR}/openexr/IlmBase/Half
							   ${_trgt} PUBLIC ${THIRD_PARTY_SOURCE_DIR}/openexr/IlmBase/Iex
							   ${_trgt} PUBLIC ${THIRD_PARTY_SOURCE_DIR}/openexr/IlmBase/IexMath
							   ${_trgt} PUBLIC ${THIRD_PARTY_SOURCE_DIR}/openexr/IlmBase/Imath
							   ${_trgt} PUBLIC ${THIRD_PARTY_SOURCE_DIR}/openexr/IlmBase/IlmThread						   
	)
endmacro()
macro(nbl_target_link_gli _trgt)
	add_dependencies(${_trgt} gli)
	target_link_libraries(${_trgt} INTERFACE
		gli
	)
	target_include_directories(${_trgt} PUBLIC ${THIRD_PARTY_SOURCE_DIR}/gli)
endmacro()
macro(nbl_target_link_spirv_tools _trgt)
	add_dependencies(${_trgt} SPIRV)
	add_dependencies(${_trgt} SPIRV-Tools)
	add_dependencies(${_trgt} SPIRV-Tools-opt)
	target_link_libraries(${_trgt} INTERFACE
		SPIRV SPIRV-Tools SPIRV-Tools-opt
	)
	target_include_directories(${_trgt} PUBLIC
		${THIRD_PARTY_SOURCE_DIR}/SPIRV-Tools/include
	)
endmacro()

add_library(Nabla STATIC
	${NABLA_SRCS_COMMON}
	${NABLA_SRCS_NORMAL}
	${NABLA_HEADERS}
	$<TARGET_OBJECTS:aesGladman>
	$<TARGET_OBJECTS:bzip2>
	$<TARGET_OBJECTS:convert_utf>
	$<TARGET_OBJECTS:lz4>
	$<TARGET_OBJECTS:lzma>
	$<TARGET_OBJECTS:spirv_cross>
)

#TODO maybe make a macro to do this (note: it's also used in common.cmake)
set_property(TARGET Nabla PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

add_dependencies(Nabla openssl_build)
nbl_target_link_zlib(Nabla)
nbl_target_link_openssl(Nabla)
nbl_target_link_shaderc(Nabla)
nbl_target_link_libjpeg(Nabla)
nbl_target_link_libpng(Nabla)
nbl_target_link_openexr(Nabla)
if(_NBL_COMPILE_WITH_GLI_)
	nbl_target_link_gli(Nabla)
endif()
nbl_target_include_parallel_hashmap(Nabla)
nbl_target_link_spirv_tools(Nabla)

target_link_libraries(Nabla INTERFACE ${OPENGL_gl_LIBRARY})
if (UNIX)
	target_link_libraries(Nabla INTERFACE 
		${X11_LIBRARIES} 
		${X11_Xrandr_LIB} 
		${X11_Xxf86vm_LIB} 
		${CMAKE_THREAD_LIBS_INIT}
		${CMAKE_DL_LIBS}
		$<$<CONFIG:DEBUG>:-lunwind>
	)
endif()

target_include_directories(Nabla PUBLIC 
	${NBL_ROOT_PATH}/include
	${COMMON_INCLUDE_DIRS}
	${THIRD_PARTY_SOURCE_DIR}
	#those also goes as PUBLIC because of examples
	"$<$<CONFIG:DEBUG>:${NABLA_CONF_DIR_DEBUG}>"
	"$<$<CONFIG:RELEASE>:${NABLA_CONF_DIR_RELEASE}>"
	"$<$<CONFIG:RELWITHDEBINFO>:${NABLA_CONF_DIR_RELWITHDEBINFO}>"
)
target_include_directories(Nabla PUBLIC ${NBL_ROOT_PATH}/src ${NBL_ROOT_PATH}/source/Nabla ${NBL_ROOT_PATH}/src/3rdparty) # this should be PRIVATE, but things from /src (or /source) are sometimes included in things in /include and so examples have to put source dirs into theirs Include Path
target_include_directories(Nabla PRIVATE ${CMAKE_BINARY_DIR}/include)
if (UNIX)
	target_include_directories(Nabla PUBLIC 
		${X11_INCLUDE_DIR} 
		${X11_Xrandr_INCLUDE_PATH} 
		${X11_xf86vmode_INCLUDE_PATH})
endif()

if (NBL_COMPILE_WITH_CUDA)
	target_include_directories(Nabla PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
endif()

target_compile_definitions(Nabla 
	PRIVATE	_NBL_STATIC_LIB_ SPIRV_CROSS_EXCEPTIONS_TO_ASSERTIONS
	PUBLIC	UNW_LOCAL_ONLY
)

if (MSVC) #on MSVC it won't compile without this option!
	target_compile_options(Nabla PUBLIC /bigobj)
endif()

#precompiled headers
if(NBL_PCH)
	target_precompile_headers(Nabla PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/pch.h")
endif()
	

if(NBL_EMBED_BUILTIN_RESOURCES)
	add_custom_target(builtin_resources
	   DEPENDS ${CMAKE_BINARY_DIR}/include/nbl/builtin/builtinResources.h ${CMAKE_CURRENT_BINARY_DIR}/builtin/builtinResourceData.cpp)
	  add_dependencies(Nabla builtin_resources)
elseif()
	foreach(X IN LISTS nbl_resources_to_embed)
		list(APPEND NABLA_HEADERS_PUBLIC  ${NBL_ROOT_PATH}/include/${X})
	endforeach()
endif()


# extensions
start_tracking_variables_for_propagation_to_parent()
add_subdirectory(ext)
propagate_changed_variables_to_parent_scope()


set(NABLA_HEADERS_TO_INSTALL 
	${NABLA_HEADERS_PUBLIC}
)
nbl_install_headers("${NABLA_HEADERS_TO_INSTALL}" "${NBL_ROOT_PATH}/include")
nbl_install_config_header(BuildConfigOptions.h)


macro(nbl_install_lib _TRGT)
	install(TARGETS ${_TRGT} ARCHIVE DESTINATION lib CONFIGURATIONS Release)
	install(TARGETS ${_TRGT} ARCHIVE DESTINATION debug/lib CONFIGURATIONS Debug)
	install(TARGETS ${_TRGT} ARCHIVE DESTINATION relwithdebinfo/lib CONFIGURATIONS RelWithDebInfo)
endmacro()
# Install main lib
set_target_properties(Nabla PROPERTIES DEBUG_POSTFIX _debug)
set_target_properties(Nabla PROPERTIES RELWITHDEBDINFO_POSTFIX _relwithdebinfo)
nbl_install_lib(Nabla)

#install dependencies
nbl_install_lib(glslang)
nbl_install_lib(jpeg)
nbl_install_lib(Half)
nbl_install_lib(HLSL)
nbl_install_lib(Iex)
nbl_install_lib(IexMath)
nbl_install_lib(IlmImf)
nbl_install_lib(IlmThread)
nbl_install_lib(Imath)
nbl_install_lib(png_static)
if(WIN32)
	install(FILES
			"$<TARGET_PROPERTY:openssl_build,BINARY_DIR>/build/lib/libeay32.lib"
			"$<TARGET_PROPERTY:openssl_build,BINARY_DIR>/build/lib/ssleay32.lib"
		DESTINATION lib
	)
else()
	install(FILES
			"$<TARGET_PROPERTY:openssl_build,BINARY_DIR>/build/lib/libcrypto.a"
			"$<TARGET_PROPERTY:openssl_build,BINARY_DIR>/build/lib/libssl.a"
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