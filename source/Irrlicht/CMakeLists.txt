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

	set(CMAKE_INSTALL_PREFIX "${IRR_ROOT_PATH}/install/${PLATFORM}" CACHE PATH "Install path" FORCE)
endif()

# include utility macros/functions
include(common)

irr_get_conf_dir(IRRLICHT_CONF_DIR_DEBUG Debug)
irr_get_conf_dir(IRRLICHT_CONF_DIR_RELEASE Release)

set(_IRR_COMPILE_WITH_OPENGL_ ${IRR_COMPILE_WITH_OPENGL})
if (IRR_COMPILE_WITH_OPENCL)
	find_package(OpenCL)

	if (OpenCL_FOUND)
		message(STATUS "Building with OpenCL interop")
		
		set(_IRR_COMPILE_WITH_OPENCL_ ${IRR_COMPILE_WITH_OPENCL})
	else()
		message(STATUS "OpenCL not found")
	endif()
endif()
if (IRR_COMPILE_WITH_CUDA)
	message(STATUS "Building with CUDA interop")
	set(_IRR_COMPILE_WITH_CUDA_ ${IRR_COMPILE_WITH_CUDA})
	if (IRR_BUILD_OPTIX)
		set(_IRR_BUILD_OPTIX_ ${IRR_BUILD_OPTIX})
	endif()
endif()
#set(_IRR_TARGET_ARCH_ARM_ ${IRR_TARGET_ARCH_ARM}) #uncomment in the future
set(__IRR_FAST_MATH ${IRR_FAST_MATH})
set(_IRR_DEBUG 0)
configure_file("${IRR_ROOT_PATH}/include/irr/config/BuildConfigOptions.h.in" "${IRRLICHT_CONF_DIR_RELEASE}/BuildConfigOptions.h")
set(_IRR_DEBUG 1)
configure_file("${IRR_ROOT_PATH}/include/irr/config/BuildConfigOptions.h.in" "${IRRLICHT_CONF_DIR_DEBUG}/BuildConfigOptions.h")


file(GLOB_RECURSE IRRLICHT_HEADERS_PUBLIC "${IRR_ROOT_PATH}/include/*.h")
file(GLOB_RECURSE IRRLICHT_HEADERS_PRIV1 "*.h")
file(GLOB_RECURSE IRRLICHT_HEADERS_PRIV2 "${IRR_ROOT_PATH}/src/irr/*.h")
# just gather all the header files and later put them into project so it's easy to search for things using IDE
set(IRRLICHT_HEADERS ${IRRLICHT_HEADERS_PUBLIC} ${IRRLICHT_HEADERS_PIRV1} ${IRRLICHT_HEADERS_PRIV2})


set(IRRLICHT_SRCS_COMMON
# Core Memory
	${IRR_ROOT_PATH}/src/irr/core/memory/CLeakDebugger.cpp

# Pixel Formats
	${IRR_ROOT_PATH}/src/irr/asset/format/convertColor.cpp

# Mesh loaders
	${IRR_ROOT_PATH}/src/irr/asset/CBAWMeshFileLoader.cpp
	${IRR_ROOT_PATH}/src/irr/asset/COBJMeshFileLoader.cpp
	${IRR_ROOT_PATH}/src/irr/asset/CPLYMeshFileLoader.cpp
	${IRR_ROOT_PATH}/src/irr/asset/CSTLMeshFileLoader.cpp
	${IRR_ROOT_PATH}/src/irr/asset/CXMeshFileLoader.cpp
	${IRR_ROOT_PATH}/src/irr/asset/CBufferLoaderBIN.cpp

# Mesh writers
	${IRR_ROOT_PATH}/src/irr/asset/CBAWMeshWriter.cpp
	${IRR_ROOT_PATH}/src/irr/asset/CPLYMeshWriter.cpp
	${IRR_ROOT_PATH}/src/irr/asset/CSTLMeshWriter.cpp
	
# Assets
	${IRR_ROOT_PATH}/src/irr/asset/IAsset.cpp
	${IRR_ROOT_PATH}/src/irr/asset/IAssetManager.cpp
	${IRR_ROOT_PATH}/src/irr/asset/IAssetWriter.cpp
	${IRR_ROOT_PATH}/src/irr/asset/IAssetLoader.cpp
	
# Builtin include loaders
	${IRR_ROOT_PATH}/src/irr/asset/CGLSLScanBuiltinIncludeLoader.cpp
	${IRR_ROOT_PATH}/src/irr/asset/CGLSLSkinningBuiltinIncludeLoader.cpp
	
# Shaders
	${IRR_ROOT_PATH}/src/irr/asset/IGLSLCompiler.cpp
	${IRR_ROOT_PATH}/src/irr/asset/ICPUShader.cpp

# Other mesh-related stuff
	${IRR_ROOT_PATH}/src/irr/asset/bawformat/CBAWFile.cpp
	${IRR_ROOT_PATH}/src/irr/asset/bawformat/legacy/CBAWLegacy.cpp
	${IRR_ROOT_PATH}/src/irr/asset/bawformat/legacy/CBAWVersionUpFunctions.cpp
	${IRR_ROOT_PATH}/src/irr/asset/bawformat/CBlobsLoadingManager.cpp
	${IRR_ROOT_PATH}/src/irr/asset/CForsythVertexCacheOptimizer.cpp
	${IRR_ROOT_PATH}/src/irr/asset/CSmoothNormalGenerator.cpp
	${IRR_ROOT_PATH}/src/irr/asset/CMeshManipulator.cpp
	CMeshSceneNode.cpp
	CMeshSceneNodeInstanced.cpp
	${IRR_ROOT_PATH}/src/irr/asset/COverdrawMeshOptimizer.cpp
	CSkinnedMeshSceneNode.cpp
	${IRR_ROOT_PATH}/src/irr/asset/bawformat/TypedBlob.cpp
	${IRR_ROOT_PATH}/src/irr/asset/CCPUSkinnedMesh.cpp

# Scene objects
	CCameraSceneNode.cpp
	${IRR_ROOT_PATH}/src/irr/asset/CGeometryCreator.cpp
	CSceneManager.cpp
	CSkyBoxSceneNode.cpp
	CSkyDomeSceneNode.cpp

# Animators
	CSceneNodeAnimatorCameraFPS.cpp
	CSceneNodeAnimatorCameraMaya.cpp
	CSceneNodeAnimatorCameraModifiedMaya.cpp
	CSceneNodeAnimatorDelete.cpp
	CSceneNodeAnimatorFlyCircle.cpp
	CSceneNodeAnimatorFlyStraight.cpp
	CSceneNodeAnimatorFollowSpline.cpp
	CSceneNodeAnimatorRotation.cpp

# Driver-specific objects
	IDriver.cpp
	CNullDriver.cpp
	COpenGLDriver.cpp
	COpenGLExtensionHandler.cpp
	COpenGLFrameBuffer.cpp
	COpenGLSLMaterialRenderer.cpp
	COpenGLTexture.cpp
	STextureSamplingParams.cpp
	FW_Mutex.cpp

# Image processing
	${IRR_ROOT_PATH}/src/irr/asset/CImageLoaderDDS.cpp
	${IRR_ROOT_PATH}/src/irr/asset/CImageLoaderJPG.cpp
	${IRR_ROOT_PATH}/src/irr/asset/CImageLoaderPNG.cpp
	${IRR_ROOT_PATH}/src/irr/asset/CImageLoaderTGA.cpp
	${IRR_ROOT_PATH}/src/irr/asset/CImageWriterJPG.cpp
	${IRR_ROOT_PATH}/src/irr/asset/CImageWriterPNG.cpp
	${IRR_ROOT_PATH}/src/irr/asset/CImageWriterTGA.cpp

# Video
	CFPSCounter.cpp
	CVideoModeList.cpp
	${IRR_ROOT_PATH}/src/irr/video/alloc/GPUMemoryAllocatorBase.cpp
	${IRR_ROOT_PATH}/src/irr/video/alloc/SimpleGPUBufferAllocator.cpp

# Input/output
	CFileList.cpp
	CFileSystem.cpp
	CLimitReadFile.cpp
	CMemoryFile.cpp
	CReadFile.cpp
	CWriteFile.cpp
	CMountPointReader.cpp
	CPakReader.cpp
	CTarReader.cpp
	CZipReader.cpp

# Other
	IrrlichtDevice.cpp
	CIrrDeviceSDL.cpp
	CIrrDeviceLinux.cpp
	CIrrDeviceConsole.cpp
	CIrrDeviceStub.cpp
	CIrrDeviceWin32.cpp
	CLogger.cpp
	COSOperator.cpp
	Irrlicht.cpp
	os.cpp
	IReferenceCounted.cpp
)
if (MSVC)
	list(APPEND IRRLICHT_SRCS_COMMON pch.cpp) # @crisspl why are only some sources appended? also why sources?
endif()

set(IRRLICHT_SRCS_NORMAL
	# Vulkan

	# OpenCL
	COpenGL2DTextureArray.cpp
	COpenGL2DTexture.cpp
	COpenGL3DTexture.cpp
	COpenGLCubemapTexture.cpp
	COpenGLMultisampleTextureArray.cpp
	COpenGLMultisampleTexture.cpp
	COpenGLQuery.cpp
	COpenGLTransformFeedback.cpp
	COpenGLVAOSpec.cpp
	COpenGL1DTexture.cpp
	COpenGL1DTextureArray.cpp
	COpenGLCubemapArrayTexture.cpp
	COpenGLCubemapTexture.cpp
	
	# CUDA
	${IRR_ROOT_PATH}/src/irr/video/CCUDAHandler.cpp

	# OpenCL
	COpenCLHandler.cpp

	# abstract (move this out eventually)
	${IRR_ROOT_PATH}/src/irr/video/CDerivativeMapCreator.cpp
)

set(IRRLICHT_SRCS_SERVER
)

set(COMMON_INCLUDE_DIRS
	${THIRD_PARTY_SOURCE_DIR}/convertutf
	${THIRD_PARTY_SOURCE_DIR}/utfcpp/source
	$<TARGET_PROPERTY:zlibstatic,BINARY_DIR> #for dynamically generated zconf.h
	$<TARGET_PROPERTY:png_static,BINARY_DIR> #for dynamically generated pnglibconf.h
	$<TARGET_PROPERTY:jpeg,BINARY_DIR> #for dynamically generated jconfig.h
)

irr_adjust_flags() # macro defined in root CMakeLists
irr_adjust_definitions() # macro defined in root CMakeLists

macro(irr_target_link_zlib _trgt)
	add_dependencies(${_trgt} zlibstatic)
	target_link_libraries(${_trgt} INTERFACE
		zlibstatic
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
	target_include_directories(${_trgt} PUBLIC ${IRR_ROOT_PATH}/3rdparty/shaderc/libshaderc/include)
endmacro()
macro(irr_target_link_libjpeg _trgt)
	add_dependencies(${_trgt} jpeg)
	target_link_libraries(${_trgt} INTERFACE
		jpeg
	)
	target_include_directories(${_trgt} PUBLIC ${IRR_ROOT_PATH}/3rdparty/libjpeg)
endmacro()
macro(irr_target_link_libpng _trgt)
	add_dependencies(${_trgt} png_static)
	target_link_libraries(${_trgt} INTERFACE
		png_static
	)
	target_include_directories(${_trgt} PUBLIC ${IRR_ROOT_PATH}/3rdparty/libpng)
endmacro()

# get function for cross-compiler PCH configuration
include(add_pch)

add_library(Irrlicht STATIC
	${IRRLICHT_SRCS_COMMON}
	${IRRLICHT_SRCS_NORMAL}
	${IRRLICHT_HEADERS}
	$<TARGET_OBJECTS:aesGladman>
	$<TARGET_OBJECTS:bzip2>
	$<TARGET_OBJECTS:convert_utf>
	$<TARGET_OBJECTS:lz4>
	$<TARGET_OBJECTS:lzma>
	$<TARGET_OBJECTS:spirv_cross>
)

add_dependencies(Irrlicht openssl_build)
irr_target_link_zlib(Irrlicht)
irr_target_link_openssl(Irrlicht)
irr_target_link_shaderc(Irrlicht)
irr_target_link_libjpeg(Irrlicht)
irr_target_link_libpng(Irrlicht)
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
	${IRR_ROOT_PATH}/include
	${COMMON_INCLUDE_DIRS}
	${THIRD_PARTY_SOURCE_DIR}
	#those also goes as PUBLIC because of examples
	"$<$<CONFIG:DEBUG>:${IRRLICHT_CONF_DIR_DEBUG}>"
	"$<$<CONFIG:RELEASE>:${IRRLICHT_CONF_DIR_RELEASE}>"
)
target_include_directories(Irrlicht PUBLIC ${IRR_ROOT_PATH}/src ${IRR_ROOT_PATH}/source/Irrlicht ${IRR_ROOT_PATH}/src/3rdparty) # this should be PRIVATE, but things from /src (or /source) are sometimes included in things in /include and so examples have to put source dirs into theirs Include Path
if (UNIX)
	target_include_directories(Irrlicht PUBLIC 
		${X11_INCLUDE_DIR} 
		${X11_Xrandr_INCLUDE_PATH} 
		${X11_xf86vmode_INCLUDE_PATH})
endif()


if (IRR_COMPILE_WITH_CUDA)
	if (WIN32)
		target_link_libraries(Irrlicht INTERFACE
			${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../lib/x64/nvrtc.lib
		)
	endif()
	target_include_directories(Irrlicht PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

if (IRR_COMPILE_WITH_OPENCL AND OpenCL_FOUND)
	target_link_libraries(Irrlicht PUBLIC OpenCL::OpenCL)
endif()

find_package(Vulkan)
if (Vulkan_FOUND)
	message(STATUS "Building with Vulkan driver")
else()
	message(STATUS "Vulkan driver is not enabled")
endif()

target_compile_definitions(Irrlicht 
	PRIVATE	_IRR_STATIC_LIB_
	PUBLIC	UNW_LOCAL_ONLY
)

if(IRR_PCH)
    if (UNIX)
        if (${CMAKE_BUILD_TYPE} STREQUAL "Release")
            ADD_PRECOMPILED_HEADER(Irrlicht pch.h "${IRRLICHT_CONF_DIR_RELEASE}" FORCEINCLUDE)
	elseif (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
            ADD_PRECOMPILED_HEADER(Irrlicht pch.h "${IRRLICHT_CONF_DIR_DEBUG}" FORCEINCLUDE)
        endif()
    else()
        ADD_PRECOMPILED_HEADER(Irrlicht pch.h FORCEINCLUDE)
    endif()
endif()

add_library(IrrlichtServer STATIC
	${IRRLICHT_SRCS_COMMON}
	${IRRLICHT_SRCS_SERVER}
	${IRRLICHT_HEADERS}
	$<TARGET_OBJECTS:aesGladman>
	$<TARGET_OBJECTS:bzip2>
	$<TARGET_OBJECTS:convert_utf>
	$<TARGET_OBJECTS:lz4>
	$<TARGET_OBJECTS:lzma>
)

add_dependencies(IrrlichtServer openssl_build)
irr_target_link_zlib(IrrlichtServer)
irr_target_link_openssl(IrrlichtServer)
irr_target_link_shaderc(IrrlichtServer)
irr_target_link_libjpeg(IrrlichtServer)
irr_target_link_libpng(IrrlichtServer)
irr_target_include_parallel_hashmap(IrrlichtServer)

target_include_directories(IrrlichtServer PUBLIC 
	${IRR_ROOT_PATH}/include
	${COMMON_INCLUDE_DIRS}
	${THIRD_PARTY_SOURCE_DIR}
	#those also goes as PUBLIC because of examples
	"$<$<CONFIG:DEBUG>:${IRRLICHT_CONF_DIR_DEBUG}>"
	"$<$<CONFIG:RELEASE>:${IRRLICHT_CONF_DIR_RELEASE}>"
)
target_include_directories(IrrlichtServer PRIVATE ${IRR_ROOT_PATH}/src ${IRR_ROOT_PATH}/source/Irrlicht ${IRR_ROOT_PATH}/src/3rdparty)

target_compile_definitions(IrrlichtServer PRIVATE
	_IRR_STATIC_LIB_
	_IRR_SERVER_
)


if(IRR_PCH)
    if (UNIX)
        # "CMake Error: Attempt to add a custom rule to output ".../source/Irrlicht/pch.h.gch/.c++.rule" which already has a custom rule."
        # so there need to be another pch file to create custom_command for it. Or maybe share PCH target between Irrlicht and IrrlichtServer? (not sure if possible)
        add_custom_command(
          OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/server_pch.h"
          COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/pch.h" "${CMAKE_CURRENT_BINARY_DIR}/server_pch.h"
          DEPENDS pch.h
        )
        file(RELATIVE_PATH SERVER_PCH_INPUT_RELDIR "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/server_pch.h")
        if (${CMAKE_BUILD_TYPE} STREQUAL "Release")
            ADD_PRECOMPILED_HEADER(IrrlichtServer "${SERVER_PCH_INPUT_RELDIR}" "${IRRLICHT_CONF_DIR_RELEASE}" FORCEINCLUDE)
	elseif (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
            ADD_PRECOMPILED_HEADER(IrrlichtServer "${SERVER_PCH_INPUT_RELDIR}" "${IRRLICHT_CONF_DIR_DEBUG}" FORCEINCLUDE)
        endif()
	#add_target_byproducts(IrrlichtServer "${CMAKE_CURRENT_BINARY_DIR}/server_pch.h") # how to resolve this?
    else()
        ADD_PRECOMPILED_HEADER(IrrlichtServer pch.h FORCEINCLUDE)
    endif()
endif()

if (MSVC) #on MSVC it won't compile without this option!
	target_compile_options(Irrlicht PUBLIC /bigobj)
	target_compile_options(IrrlichtServer PUBLIC /bigobj)
endif()

set_target_properties(Irrlicht IrrlichtServer PROPERTIES DEBUG_POSTFIX _debug)

macro(irr_install_static_lib _TRGT)
	install(TARGETS ${_TRGT} ARCHIVE DESTINATION lib CONFIGURATIONS Release)
	install(TARGETS ${_TRGT} ARCHIVE DESTINATION debug/lib CONFIGURATIONS Debug)
endmacro()

irr_install_static_lib(Irrlicht)
irr_install_static_lib(IrrlichtServer)
irr_install_headers("${IRRLICHT_HEADERS_PUBLIC}" "${IRR_ROOT_PATH}/include")
irr_install_config_header(BuildConfigOptions.h)