# Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
# This file is part of the "Nabla Engine".
# For conditions of distribution and use, see copyright notice in nabla.h

add_subdirectory(ext)

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

irr_get_conf_dir(NABLA_CONF_DIR_DEBUG Debug)
irr_get_conf_dir(NABLA_CONF_DIR_RELEASE Release)
if(NBL_EMBED_BUILTIN_RESOURCES)	
	add_subdirectory(builtin)
	irr_add_builtin_resources("${irr_resources_to_embed}")
endif()
irr_get_conf_dir(NABLA_CONF_DIR_RELWITHDEBINFO RelWithDebInfo)

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
configure_file("${NBL_ROOT_PATH}/include/irr/config/BuildConfigOptions.h.in" "${NABLA_CONF_DIR_RELEASE}/BuildConfigOptions.h")
configure_file("${NBL_ROOT_PATH}/include/irr/config/BuildConfigOptions.h.in" "${NABLA_CONF_DIR_RELWITHDEBINFO}/BuildConfigOptions.h")
set(_NBL_DEBUG 1)
configure_file("${NBL_ROOT_PATH}/include/irr/config/BuildConfigOptions.h.in" "${NABLA_CONF_DIR_DEBUG}/BuildConfigOptions.h")

set(NABLA_HEADERS_PUBLIC
	"${NBL_ROOT_PATH}/source/Irrlicht/FW_Mutex.h"
	"${NBL_ROOT_PATH}/source/Irrlicht/os.h"
	"${NBL_ROOT_PATH}/source/Irrlicht/CLogger.h" 
)
file(GLOB_RECURSE TEMP_GLOB_RES "${NBL_ROOT_PATH}/include/*.h")
set(NABLA_HEADERS_PUBLIC ${NABLA_HEADERS_PUBLIC} ${TEMP_GLOB_RES})
file(GLOB_RECURSE TEMP_GLOB_RES "${NBL_ROOT_PATH}/include/*.tcc")
set(NABLA_HEADERS_PUBLIC ${NABLA_HEADERS_PUBLIC} ${TEMP_GLOB_RES})

file(GLOB_RECURSE NABLA_HEADERS_PRIV1 "*.h")
file(GLOB_RECURSE NABLA_HEADERS_PRIV2 "${NBL_ROOT_PATH}/src/irr/*.h")
# just gather all the header files and later put them into project so it's easy to search for things using IDE
set(NABLA_HEADERS ${NABLA_HEADERS_PUBLIC} ${NABLA_HEADERS_PIRV1} ${NABLA_HEADERS_PRIV2})


set(NBL_CORE_SOURCES
	${NBL_ROOT_PATH}/src/irr/core/IReferenceCounted.cpp
# Core Memory
	${NBL_ROOT_PATH}/src/irr/core/memory/CLeakDebugger.cpp
)
set(NBL_SYSTEM_SOURCES
# Junk to refactor
	${NBL_ROOT_PATH}/source/Irrlicht/CFileList.cpp
	${NBL_ROOT_PATH}/source/Irrlicht/CFileSystem.cpp
	${NBL_ROOT_PATH}/source/Irrlicht/CLimitReadFile.cpp
	${NBL_ROOT_PATH}/source/Irrlicht/CMemoryFile.cpp
	${NBL_ROOT_PATH}/source/Irrlicht/CReadFile.cpp
	${NBL_ROOT_PATH}/source/Irrlicht/CWriteFile.cpp
	${NBL_ROOT_PATH}/source/Irrlicht/CMountPointReader.cpp
	${NBL_ROOT_PATH}/source/Irrlicht/CPakReader.cpp
	${NBL_ROOT_PATH}/source/Irrlicht/CTarReader.cpp
	${NBL_ROOT_PATH}/source/Irrlicht/CZipReader.cpp
	${NBL_ROOT_PATH}/source/Irrlicht/CLogger.cpp
	${NBL_ROOT_PATH}/source/Irrlicht/COSOperator.cpp
	${NBL_ROOT_PATH}/source/Irrlicht/os.cpp
)
set(NBL_ASSET_SOURCES	
# Assets
	${NBL_ROOT_PATH}/src/irr/asset/IAsset.cpp
	${NBL_ROOT_PATH}/src/irr/asset/IAssetManager.cpp
	${NBL_ROOT_PATH}/src/irr/asset/IAssetWriter.cpp
	${NBL_ROOT_PATH}/src/irr/asset/IAssetLoader.cpp
	
# Builtin include loaders
	${NBL_ROOT_PATH}/src/irr/asset/CGLSLScanBuiltinIncludeLoader.cpp
	${NBL_ROOT_PATH}/src/irr/asset/CGLSLSkinningBuiltinIncludeLoader.cpp
	
# Shaders
	${NBL_ROOT_PATH}/src/irr/asset/IGLSLCompiler.cpp
	${NBL_ROOT_PATH}/src/irr/asset/CShaderIntrospector.cpp
	${NBL_ROOT_PATH}/src/irr/asset/CGLSLLoader.cpp
	${NBL_ROOT_PATH}/src/irr/asset/CSPVLoader.cpp
	
# Pipeline loaders
	${NBL_ROOT_PATH}/src/irr/asset/CGraphicsPipelineLoaderMTL.cpp

# Meshes
	${NBL_ROOT_PATH}/src/irr/asset/CForsythVertexCacheOptimizer.cpp
	${NBL_ROOT_PATH}/src/irr/asset/CSmoothNormalGenerator.cpp
	${NBL_ROOT_PATH}/src/irr/asset/CQuantNormalCache.cpp
	${NBL_ROOT_PATH}/src/irr/asset/CGeometryCreator.cpp
	${NBL_ROOT_PATH}/src/irr/asset/CMeshManipulator.cpp
	${NBL_ROOT_PATH}/src/irr/asset/COverdrawMeshOptimizer.cpp
	${NBL_ROOT_PATH}/src/irr/asset/CSmoothNormalGenerator.cpp

# Mesh loaders
	${NBL_ROOT_PATH}/src/irr/asset/CBAWMeshFileLoader.cpp
	${NBL_ROOT_PATH}/src/irr/asset/COBJMeshFileLoader.cpp
	${NBL_ROOT_PATH}/src/irr/asset/CPLYMeshFileLoader.cpp
	${NBL_ROOT_PATH}/src/irr/asset/CSTLMeshFileLoader.cpp
	${NBL_ROOT_PATH}/src/irr/asset/CBufferLoaderBIN.cpp

# Mesh writers
	${NBL_ROOT_PATH}/src/irr/asset/CBAWMeshWriter.cpp
	${NBL_ROOT_PATH}/src/irr/asset/CPLYMeshWriter.cpp
	${NBL_ROOT_PATH}/src/irr/asset/CSTLMeshWriter.cpp

# BaW Format
	${NBL_ROOT_PATH}/src/irr/asset/bawformat/TypedBlob.cpp
	${NBL_ROOT_PATH}/src/irr/asset/bawformat/CBAWFile.cpp
	${NBL_ROOT_PATH}/src/irr/asset/bawformat/legacy/CBAWLegacy.cpp
	${NBL_ROOT_PATH}/src/irr/asset/bawformat/legacy/CBAWVersionUpFunctions.cpp
	${NBL_ROOT_PATH}/src/irr/asset/bawformat/CBlobsLoadingManager.cpp

# Images
	${NBL_ROOT_PATH}/src/irr/asset/IImageAssetHandlerBase.cpp
	${NBL_ROOT_PATH}/src/irr/asset/filters/CBasicImageFilterCommon.cpp

# Image loaders
	${NBL_ROOT_PATH}/src/irr/asset/IImageLoader.cpp
	${NBL_ROOT_PATH}/src/irr/asset/CImageLoaderJPG.cpp
	${NBL_ROOT_PATH}/src/irr/asset/CImageLoaderPNG.cpp
	${NBL_ROOT_PATH}/src/irr/asset/CImageLoaderTGA.cpp
	${NBL_ROOT_PATH}/src/irr/asset/CImageLoaderOpenEXR.cpp
	${NBL_ROOT_PATH}/src/irr/asset/CGLILoader.cpp

# Image writers
	${NBL_ROOT_PATH}/src/irr/asset/IImageWriter.cpp
	${NBL_ROOT_PATH}/src/irr/asset/CImageWriterJPG.cpp
	${NBL_ROOT_PATH}/src/irr/asset/CImageWriterPNG.cpp
	${NBL_ROOT_PATH}/src/irr/asset/CImageWriterTGA.cpp
	${NBL_ROOT_PATH}/src/irr/asset/CImageWriterOpenEXR.cpp
	${NBL_ROOT_PATH}/src/irr/asset/CGLIWriter.cpp
	
# Material compiler
	${NBL_ROOT_PATH}/src/irr/asset/material_compiler/CMaterialCompilerGLSLBackendCommon.cpp
	${NBL_ROOT_PATH}/src/irr/asset/material_compiler/CMaterialCompilerGLSLRasterBackend.cpp
)
set(NBL_VIDEO_SOURCES
# Allocators
<<<<<<< HEAD
	${NBL_ROOT_PATH}/src/irr/video/alloc/GPUMemoryAllocatorBase.cpp
	${NBL_ROOT_PATH}/src/irr/video/alloc/SimpleGPUBufferAllocator.cpp

# Junk to refactor
${NBL_ROOT_PATH}/source/Irrlicht/IDriver.cpp
${NBL_ROOT_PATH}/source/Irrlicht/CNullDriver.cpp
	${NBL_ROOT_PATH}/src/irr/video/COpenGLSpecializedShader.cpp
	${NBL_ROOT_PATH}/source/Irrlicht/FW_Mutex.cpp
	${NBL_ROOT_PATH}/source/Irrlicht/CFPSCounter.cpp
=======
	${IRR_ROOT_PATH}/src/irr/video/alloc/GPUMemoryAllocatorBase.cpp
	${IRR_ROOT_PATH}/src/irr/video/alloc/SimpleGPUBufferAllocator.cpp
	
# Shaders
	${IRR_ROOT_PATH}/src/irr/video/COpenGLSpecializedShader.cpp

# Property Pools
	${IRR_ROOT_PATH}/src/irr/video/CPropertyPoolHandler.cpp

# Junk to refactor
${IRR_ROOT_PATH}/source/Irrlicht/IDriver.cpp
${IRR_ROOT_PATH}/source/Irrlicht/CNullDriver.cpp
	${IRR_ROOT_PATH}/source/Irrlicht/FW_Mutex.cpp
	${IRR_ROOT_PATH}/source/Irrlicht/CFPSCounter.cpp
>>>>>>> 161124a675076a453cebd61216dbc4a713b49049
)
set(NBL_SCENE_SOURCES
# Junk to refactor
${NBL_ROOT_PATH}/source/Irrlicht/CSkinnedMeshSceneNode.cpp
${NBL_ROOT_PATH}/source/Irrlicht/CCameraSceneNode.cpp
${NBL_ROOT_PATH}/source/Irrlicht/CSceneManager.cpp

# Animators
${NBL_ROOT_PATH}/source/Irrlicht/CSceneNodeAnimatorCameraFPS.cpp
	${NBL_ROOT_PATH}/source/Irrlicht/CSceneNodeAnimatorCameraMaya.cpp
	${NBL_ROOT_PATH}/source/Irrlicht/CSceneNodeAnimatorCameraModifiedMaya.cpp
	${NBL_ROOT_PATH}/source/Irrlicht/CSceneNodeAnimatorDelete.cpp
	${NBL_ROOT_PATH}/source/Irrlicht/CSceneNodeAnimatorFlyCircle.cpp
	${NBL_ROOT_PATH}/source/Irrlicht/CSceneNodeAnimatorFlyStraight.cpp
	${NBL_ROOT_PATH}/source/Irrlicht/CSceneNodeAnimatorFollowSpline.cpp
	${NBL_ROOT_PATH}/source/Irrlicht/CSceneNodeAnimatorRotation.cpp
)
set(NABLA_SRCS_COMMON
	${NBL_CORE_SOURCES}
	${NBL_SYSTEM_SOURCES}
	${NBL_ASSET_SOURCES}
	${NBL_VIDEO_SOURCES}
	${NBL_SCENE_SOURCES}

# Junk to refactor
	${NBL_ROOT_PATH}/source/Irrlicht/IrrlichtDevice.cpp
	${NBL_ROOT_PATH}/source/Irrlicht/CIrrDeviceLinux.cpp
	${NBL_ROOT_PATH}/source/Irrlicht/CIrrDeviceConsole.cpp
	${NBL_ROOT_PATH}/source/Irrlicht/CIrrDeviceStub.cpp
	${NBL_ROOT_PATH}/source/Irrlicht/CIrrDeviceWin32.cpp
	${NBL_ROOT_PATH}/source/Irrlicht/Irrlicht.cpp
)

if (NBL_EMBED_BUILTIN_RESOURCES)
	list(APPEND NABLA_SRCS_COMMON "${CMAKE_BINARY_DIR}/src/irr/builtin/builtinResourceData.cpp")
endif()


set(NABLA_SRCS_NORMAL
	# Vulkan

	#opengl
	${NBL_ROOT_PATH}/source/Irrlicht/COpenCLHandler.cpp
	${NBL_ROOT_PATH}/source/Irrlicht/COpenGLDriver.cpp
	${NBL_ROOT_PATH}/source/Irrlicht/COpenGLExtensionHandler.cpp
	${NBL_ROOT_PATH}/source/Irrlicht/COpenGLFrameBuffer.cpp
	${NBL_ROOT_PATH}/source/Irrlicht/COpenGLQuery.cpp
	${NBL_ROOT_PATH}/src/irr/video/COpenGLPipelineCache.cpp
	
	# CUDA
	${NBL_ROOT_PATH}/src/irr/video/CCUDAHandler.cpp

	# OpenCL
	${NBL_ROOT_PATH}/source/Irrlicht/COpenCLHandler.cpp
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

irr_adjust_flags() # macro defined in root CMakeLists
irr_adjust_definitions() # macro defined in root CMakeLists

macro(irr_target_link_zlib _trgt)
	add_dependencies(${_trgt} zlibstatic)

	target_link_libraries(${_trgt} INTERFACE
		$<$<CONFIG:Debug>:${ZLIB_LIBRARY_DEBUG}>
		$<$<CONFIG:Release>:${ZLIB_LIBRARY_RELEASE}>
		$<$<CONFIG:RelWithDebInfo>:${ZLIB_LIBRARY_RELWITHDEBINFO}>
	)
	target_include_directories(${_trgt} PUBLIC ${THIRD_PARTY_SOURCE_DIR}/zlib)
endmacro()

#Parallel hashmap
macro(irr_target_include_parallel_hashmap _trgt)
	target_include_directories(${_trgt} PUBLIC ${THIRD_PARTY_SOURCE_DIR}/parallel-hashmap/parallel_hashmap)
endmacro()

macro(irr_target_link_openssl _trgt)
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
macro(irr_target_link_shaderc _trgt)
	add_dependencies(${_trgt} shaderc)
	target_link_libraries(${_trgt} INTERFACE
		shaderc
	)
	target_include_directories(${_trgt} PUBLIC ${THIRD_PARTY_SOURCE_DIR}/shaderc/libshaderc/include)
endmacro()
macro(irr_target_link_libjpeg _trgt)
	add_dependencies(${_trgt} jpeg)
	target_link_libraries(${_trgt} INTERFACE
		jpeg
	)
	target_include_directories(${_trgt} PUBLIC ${THIRD_PARTY_SOURCE_DIR}/libjpeg)
endmacro()
macro(irr_target_link_libpng _trgt)
	add_dependencies(${_trgt} png_static)
	target_link_libraries(${_trgt} INTERFACE
		png_static
	)
	target_include_directories(${_trgt} PUBLIC ${THIRD_PARTY_SOURCE_DIR}/libpng)
endmacro()
macro(irr_target_link_openexr _trgt)
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
macro(irr_target_link_gli _trgt)
	add_dependencies(${_trgt} gli)
	target_link_libraries(${_trgt} INTERFACE
		gli
	)
	target_include_directories(${_trgt} PUBLIC ${THIRD_PARTY_SOURCE_DIR}/gli)
endmacro()

add_library(Irrlicht STATIC
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
set_property(TARGET Irrlicht PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

add_dependencies(Irrlicht openssl_build)
irr_target_link_zlib(Irrlicht)
irr_target_link_openssl(Irrlicht)
irr_target_link_shaderc(Irrlicht)
irr_target_link_libjpeg(Irrlicht)
irr_target_link_libpng(Irrlicht)
irr_target_link_openexr(Irrlicht)
if(_NBL_COMPILE_WITH_GLI_)
	irr_target_link_gli(Irrlicht)
endif()
irr_target_include_parallel_hashmap(Irrlicht)

target_link_libraries(Irrlicht INTERFACE ${OPENGL_gl_LIBRARY})
if (UNIX)
	target_link_libraries(Irrlicht INTERFACE 
		${X11_LIBRARIES} 
		${X11_Xrandr_LIB} 
		${X11_Xxf86vm_LIB} 
		${CMAKE_THREAD_LIBS_INIT}
		${CMAKE_DL_LIBS}
		$<$<CONFIG:DEBUG>:-lunwind>
	)
endif()

target_include_directories(Irrlicht PUBLIC 
	${NBL_ROOT_PATH}/include
	${COMMON_INCLUDE_DIRS}
	${THIRD_PARTY_SOURCE_DIR}
	#those also goes as PUBLIC because of examples
	"$<$<CONFIG:DEBUG>:${NABLA_CONF_DIR_DEBUG}>"
	"$<$<CONFIG:RELEASE>:${NABLA_CONF_DIR_RELEASE}>"
	"$<$<CONFIG:RELWITHDEBINFO>:${NABLA_CONF_DIR_RELWITHDEBINFO}>"
)
target_include_directories(Irrlicht PUBLIC ${NBL_ROOT_PATH}/src ${NBL_ROOT_PATH}/source/Irrlicht ${NBL_ROOT_PATH}/src/3rdparty) # this should be PRIVATE, but things from /src (or /source) are sometimes included in things in /include and so examples have to put source dirs into theirs Include Path
target_include_directories(Irrlicht PRIVATE ${CMAKE_BINARY_DIR}/include)
if (UNIX)
	target_include_directories(Irrlicht PUBLIC 
		${X11_INCLUDE_DIR} 
		${X11_Xrandr_INCLUDE_PATH} 
		${X11_xf86vmode_INCLUDE_PATH})
endif()


if (NBL_COMPILE_WITH_CUDA)
	target_include_directories(Irrlicht PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

target_compile_definitions(Irrlicht 
	PRIVATE	_NBL_STATIC_LIB_ SPIRV_CROSS_EXCEPTIONS_TO_ASSERTIONS
	PUBLIC	UNW_LOCAL_ONLY
)

if (MSVC) #on MSVC it won't compile without this option!
	target_compile_options(Irrlicht PUBLIC /bigobj)
endif()

#precompiled headers
if(NBL_PCH)
	target_precompile_headers(Irrlicht PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/pch.h")
endif()
	

if(NBL_EMBED_BUILTIN_RESOURCES)
	add_custom_target(builtin_resources
	   DEPENDS ${CMAKE_BINARY_DIR}/include/irr/builtin/builtinResources.h ${CMAKE_CURRENT_BINARY_DIR}/builtin/builtinResourceData.cpp)
	  add_dependencies(Irrlicht builtin_resources)
elseif()
	foreach(X IN LISTS irr_resources_to_embed)
		list(APPEND NABLA_HEADERS_PUBLIC  ${NBL_ROOT_PATH}/include/${X})
	endforeach()
endif()


set(NABLA_HEADERS_TO_INSTALL 
	${NABLA_HEADERS_PUBLIC}
)
irr_install_headers("${NABLA_HEADERS_TO_INSTALL}" "${NBL_ROOT_PATH}/include")
irr_install_config_header(BuildConfigOptions.h)


macro(irr_install_lib _TRGT)
	install(TARGETS ${_TRGT} ARCHIVE DESTINATION lib CONFIGURATIONS Release)
	install(TARGETS ${_TRGT} ARCHIVE DESTINATION debug/lib CONFIGURATIONS Debug)
	install(TARGETS ${_TRGT} ARCHIVE DESTINATION relwithdebinfo/lib CONFIGURATIONS RelWithDebInfo)
endmacro()
# Install main lib
set_target_properties(Irrlicht PROPERTIES DEBUG_POSTFIX _debug)
set_target_properties(Irrlicht PROPERTIES RELWITHDEBDINFO_POSTFIX _relwithdebinfo)
irr_install_lib(Irrlicht)

#install dependencies
irr_install_lib(glslang)
irr_install_lib(jpeg)
irr_install_lib(Half)
irr_install_lib(HLSL)
irr_install_lib(Iex)
irr_install_lib(IexMath)
irr_install_lib(IlmImf)
irr_install_lib(IlmThread)
irr_install_lib(Imath)
irr_install_lib(png_static)
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
irr_install_lib(shaderc)
irr_install_lib(shaderc_util)
irr_install_lib(SPIRV)
irr_install_lib(SPIRV-Tools)
irr_install_lib(SPIRV-Tools-opt)
irr_install_lib(OGLCompiler)
irr_install_lib(OSDependent)
irr_install_lib(zlibstatic)