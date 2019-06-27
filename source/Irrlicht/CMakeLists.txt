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

	set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install/${PLATFORM}" CACHE PATH "Install path" FORCE)
endif()

# include utility macros/functions
include(common)

irr_get_conf_dir(IRRLICHT_CONF_DIR_DEBUG Debug)
irr_get_conf_dir(IRRLICHT_CONF_DIR_RELEASE Release)

set(_IRR_COMPILE_WITH_OPENGL_ ${IRR_COMPILE_WITH_OPENGL})
set(_IRR_COMPILE_WITH_BURNINGSVIDEO_ ${IRR_COMPILE_WITH_BURNINGSVIDEO})
#set(_IRR_TARGET_ARCH_ARM_ ${IRR_TARGET_ARCH_ARM}) #uncomment in the future
set(__IRR_FAST_MATH ${IRR_FAST_MATH})
set(_IRR_DEBUG 0)
configure_file("${CMAKE_SOURCE_DIR}/include/irr/config/BuildConfigOptions.h.in" "${IRRLICHT_CONF_DIR_RELEASE}/BuildConfigOptions.h")
set(_IRR_DEBUG 1)
configure_file("${CMAKE_SOURCE_DIR}/include/irr/config/BuildConfigOptions.h.in" "${IRRLICHT_CONF_DIR_DEBUG}/BuildConfigOptions.h")


file(GLOB_RECURSE IRRLICHT_HEADERS_PUBLIC "${CMAKE_SOURCE_DIR}/include/*.h")
file(GLOB_RECURSE IRRLICHT_HEADERS_PRIV1 "*.h")
file(GLOB_RECURSE IRRLICHT_HEADERS_PRIV2 "${CMAKE_SOURCE_DIR}/src/irr/*.h")
# just gather all the header files and later put them into project so it's easy to search for things using IDE
set(IRRLICHT_HEADERS ${IRRLICHT_HEADERS_PUBLIC} ${IRRLICHT_HEADERS_PIRV1} ${IRRLICHT_HEADERS_PRIV2})


set(IRRLICHT_SRCS_COMMON
# Mesh loaders
	${CMAKE_SOURCE_DIR}/src/irr/asset/CBAWMeshFileLoader.cpp
	${CMAKE_SOURCE_DIR}/src/irr/asset/COBJMeshFileLoader.cpp
	${CMAKE_SOURCE_DIR}/src/irr/asset/CPLYMeshFileLoader.cpp
	${CMAKE_SOURCE_DIR}/src/irr/asset/CSTLMeshFileLoader.cpp
	${CMAKE_SOURCE_DIR}/src/irr/asset/CXMeshFileLoader.cpp

# Mesh writers
	${CMAKE_SOURCE_DIR}/src/irr/asset/CBAWMeshWriter.cpp
	${CMAKE_SOURCE_DIR}/src/irr/asset/COBJMeshWriter.cpp
	${CMAKE_SOURCE_DIR}/src/irr/asset/CPLYMeshWriter.cpp
	${CMAKE_SOURCE_DIR}/src/irr/asset/CSTLMeshWriter.cpp
	
# Assets
	${CMAKE_SOURCE_DIR}/src/irr/asset/IAsset.cpp
	${CMAKE_SOURCE_DIR}/src/irr/asset/IAssetManager.cpp
	${CMAKE_SOURCE_DIR}/src/irr/asset/IAssetWriter.cpp
	${CMAKE_SOURCE_DIR}/src/irr/asset/IAssetLoader.cpp
	
# Builtin include loaders
	${CMAKE_SOURCE_DIR}/src/irr/asset/CGLSLScanBuiltinIncludeLoader.cpp
	${CMAKE_SOURCE_DIR}/src/irr/asset/CGLSLSkinningBuiltinIncludeLoader.cpp
	
# Shaders
	${CMAKE_SOURCE_DIR}/src/irr/asset/IGLSLCompiler.cpp
	${CMAKE_SOURCE_DIR}/src/irr/asset/ICPUShader.cpp

# Other mesh-related stuff
	${CMAKE_SOURCE_DIR}/src/irr/asset/bawformat/CBAWFile.cpp
	${CMAKE_SOURCE_DIR}/src/irr/asset/bawformat/legacy/CBAWLegacy.cpp
	${CMAKE_SOURCE_DIR}/src/irr/asset/bawformat/CBlobsLoadingManager.cpp
	${CMAKE_SOURCE_DIR}/src/irr/asset/CForsythVertexCacheOptimizer.cpp
	${CMAKE_SOURCE_DIR}/src/irr/asset/CSmoothNormalGenerator.cpp
	${CMAKE_SOURCE_DIR}/src/irr/asset/CMeshManipulator.cpp
	CMeshSceneNode.cpp
	CMeshSceneNodeInstanced.cpp
	${CMAKE_SOURCE_DIR}/src/irr/asset/COverdrawMeshOptimizer.cpp
	CSkinnedMeshSceneNode.cpp
	${CMAKE_SOURCE_DIR}/src/irr/asset/bawformat/TypedBlob.cpp
	CCPUSkinnedMesh.cpp

# Scene objects
	CBillboardSceneNode.cpp
	CCameraSceneNode.cpp
	CCubeSceneNode.cpp
	${CMAKE_SOURCE_DIR}/src/irr/asset/CGeometryCreator.cpp
	CSceneManager.cpp
	CSkyBoxSceneNode.cpp
	CSkyDomeSceneNode.cpp
	CSphereSceneNode.cpp

# Animators
	CSceneNodeAnimatorCameraFPS.cpp
	CSceneNodeAnimatorCameraMaya.cpp
	CSceneNodeAnimatorDelete.cpp
	CSceneNodeAnimatorFlyCircle.cpp
	CSceneNodeAnimatorFlyStraight.cpp
	CSceneNodeAnimatorFollowSpline.cpp
	CSceneNodeAnimatorRotation.cpp
	CSceneNodeAnimatorTexture.cpp

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
	CColorConverter.cpp
	CImage.cpp
	${CMAKE_SOURCE_DIR}/src/irr/video/convertColor.cpp
	${CMAKE_SOURCE_DIR}/src/irr/asset/CImageLoaderBMP.cpp
	${CMAKE_SOURCE_DIR}/src/irr/asset/CImageLoaderDDS.cpp
	${CMAKE_SOURCE_DIR}/src/irr/asset/CImageLoaderJPG.cpp
	${CMAKE_SOURCE_DIR}/src/irr/asset/CImageLoaderPNG.cpp
	${CMAKE_SOURCE_DIR}/src/irr/asset/CImageLoaderTGA.cpp
	${CMAKE_SOURCE_DIR}/src/irr/asset/CImageWriterBMP.cpp
	${CMAKE_SOURCE_DIR}/src/irr/asset/CImageWriterJPG.cpp
	${CMAKE_SOURCE_DIR}/src/irr/asset/CImageWriterPNG.cpp
	${CMAKE_SOURCE_DIR}/src/irr/asset/CImageWriterTGA.cpp

# Video
	CFPSCounter.cpp
	CVideoModeList.cpp
	${CMAKE_SOURCE_DIR}/src/irr/video/GPUMemoryAllocatorBase.cpp
	${CMAKE_SOURCE_DIR}/src/irr/video/SimpleGPUBufferAllocator.cpp

# Software renderer
	CBurningShader_Raster_Reference.cpp
	CDepthBuffer.cpp
	CSoftwareDriver2.cpp
	CSoftwareTexture2.cpp
	CTRGouraud2.cpp
	CTRTextureGouraud2.cpp
	CTRTextureGouraudNoZ2.cpp
	CTRTextureGouraudAdd2.cpp
	CTRTextureGouraudAddNoZ2.cpp
	CTRTextureGouraudVertexAlpha2.cpp
	IBurningShader.cpp

# Input/output
	CFileList.cpp
	CFileSystem.cpp
	CLimitReadFile.cpp
	CMemoryFile.cpp
	CReadFile.cpp
	CWriteFile.cpp
	CMountPointReader.cpp
	CNPKReader.cpp
	CPakReader.cpp
	CTarReader.cpp
	CWADReader.cpp
	CZipReader.cpp

# Other
	coreutil.cpp
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
	COpenCLHandler.cpp
	COpenGL2DTextureArray.cpp
	COpenGL2DTexture.cpp
	COpenGL3DTexture.cpp
	COpenGLCubemapTexture.cpp
	COpenGLMultisampleTextureArray.cpp
	COpenGLMultisampleTexture.cpp
	COpenGLQuery.cpp
	COpenGLTransformFeedback.cpp
	COpenGLVAOSpec.cpp
	CTRTextureGouraudAlpha.cpp
	CTRTextureGouraudAlphaNoZ.cpp
	COpenGL1DTexture.cpp
	COpenGL1DTextureArray.cpp
	COpenGLCubemapArrayTexture.cpp
	COpenGLCubemapTexture.cpp
)

set(IRRLICHT_SRCS_SERVER
)

set(COMMON_INCLUDE_DIRS
	${THIRD_PARTY_SOURCE_DIR}/convertutf
	${THIRD_PARTY_SOURCE_DIR}/utfcpp/source
	$<TARGET_PROPERTY:png_static,BINARY_DIR> #for dynamically generated pnglibconf.h
	$<TARGET_PROPERTY:jpeg,BINARY_DIR> #for dynamically generated jconfig.h
)

irr_adjust_flags() # macro defined in root CMakeLists
irr_adjust_definitions() # macro defined in root CMakeLists

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
	target_include_directories(${_trgt} PUBLIC ${CMAKE_SOURCE_DIR}/3rdparty/shaderc/libshaderc/include)
endmacro()
macro(irr_target_link_libjpeg _trgt)
	add_dependencies(${_trgt} jpeg)
	target_link_libraries(${_trgt} INTERFACE
		jpeg
	)
	target_include_directories(${_trgt} PUBLIC ${CMAKE_SOURCE_DIR}/3rdparty/libjpeg)
endmacro()
macro(irr_target_link_libpng _trgt)
	add_dependencies(${_trgt} png_static)
	target_link_libraries(${_trgt} INTERFACE
		png_static
	)
	target_include_directories(${_trgt} PUBLIC ${CMAKE_SOURCE_DIR}/3rdparty/libpng)
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
irr_target_link_openssl(Irrlicht)
irr_target_link_shaderc(Irrlicht)
irr_target_link_libjpeg(Irrlicht)
irr_target_link_libpng(Irrlicht)

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
	${CMAKE_SOURCE_DIR}/include
	${COMMON_INCLUDE_DIRS}
	${THIRD_PARTY_SOURCE_DIR}
	#those also goes as PUBLIC because of examples
	"$<$<CONFIG:DEBUG>:${IRRLICHT_CONF_DIR_DEBUG}>"
	"$<$<CONFIG:RELEASE>:${IRRLICHT_CONF_DIR_RELEASE}>"
)
target_include_directories(Irrlicht PUBLIC ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/source/Irrlicht) # this should be PRIVATE, but things from /src (or /source) are sometimes included in things in /include and so examples have to put source dirs into theirs Include Path
if (UNIX)
	target_include_directories(Irrlicht PUBLIC 
		${X11_INCLUDE_DIR} 
		${X11_Xrandr_INCLUDE_PATH} 
		${X11_xf86vmode_INCLUDE_PATH})
endif()

find_package(Vulkan)

if (Vulkan_FOUND)
	message(STATUS "Building with Vulkan driver")
	#target_compile_definitions(Irrlicht PRIVATE
	#	_IRR_COMPILE_WITH_VULKAN_
	#)
	#target_link_libraries(Irrlicht INTERFACE Vulkan::Vulkan) # this should work, but it doesn't
	#target_link_libraries(Irrlicht INTERFACE ${Vulkan_LIBRARY})
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
	$<TARGET_OBJECTS:zlib>
)

add_dependencies(IrrlichtServer openssl_build)
irr_target_link_openssl(IrrlichtServer)
irr_target_link_shaderc(IrrlichtServer)
irr_target_link_libjpeg(IrrlichtServer)
irr_target_link_libpng(IrrlichtServer)

target_include_directories(IrrlichtServer PUBLIC 
	${CMAKE_SOURCE_DIR}/include
	${COMMON_INCLUDE_DIRS}
	${THIRD_PARTY_SOURCE_DIR}
	#those also goes as PUBLIC because of examples
	"$<$<CONFIG:DEBUG>:${IRRLICHT_CONF_DIR_DEBUG}>"
	"$<$<CONFIG:RELEASE>:${IRRLICHT_CONF_DIR_RELEASE}>"
)
target_include_directories(IrrlichtServer PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/source/Irrlicht)

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
irr_install_headers("${IRRLICHT_HEADERS_PUBLIC}" "${CMAKE_SOURCE_DIR}/include")
irr_install_config_header(BuildConfigOptions.h)
