#find_package(OpenSSL REQUIRED)
find_package(OpenGL REQUIRED)
if(UNIX)
	find_package(X11 REQUIRED)
	set(CMAKE_THREAD_PREFER_PTHREAD 1)
	find_package(Threads REQUIRED)
endif()

file(GLOB_RECURSE IRRLICHT_HEADERS "../../*.h") # just gather all the header files and later put them into project so it's easy to search for things using IDE

set(IRRLICHT_SRCS_COMMON
# Mesh loaders
	../../src/irr/asset/CBAWMeshFileLoader.cpp
	../../src/irr/asset/COBJMeshFileLoader.cpp
	../../src/irr/asset/CPLYMeshFileLoader.cpp
	../../src/irr/asset/CSTLMeshFileLoader.cpp
	../../src/irr/asset/CXMeshFileLoader.cpp

# Mesh writers
	../../src/irr/asset/CBAWMeshWriter.cpp
	../../src/irr/asset/COBJMeshWriter.cpp
	../../src/irr/asset/CPLYMeshWriter.cpp
	../../src/irr/asset/CSTLMeshWriter.cpp
	
# Assets
	../../src/irr/asset/IAsset.cpp
	../../src/irr/asset/IAssetManager.cpp
	../../src/irr/asset/IAssetWriter.cpp
	../../src/irr/asset/IAssetLoader.cpp
	
# Shaders
	../../src/irr/asset/IGLSLCompiler.cpp
	../../src/irr/asset/ICPUShader.cpp

# Other mesh-related stuff
	../../src/irr/asset/bawformat/CBAWFile.cpp
	../../src/irr/asset/bawformat/legacy/CBAWLegacy.cpp
	../../src/irr/asset/bawformat/CBlobsLoadingManager.cpp
	../../src/irr/asset/CForsythVertexCacheOptimizer.cpp
	../../src/irr/asset/CMeshManipulator.cpp
	CMeshSceneNode.cpp
	CMeshSceneNodeInstanced.cpp
	../../src/irr/asset/COverdrawMeshOptimizer.cpp
	CSkinnedMeshSceneNode.cpp
	../../src/irr/asset/bawformat/TypedBlob.cpp
	CCPUSkinnedMesh.cpp
	
# Shader loading and processing
	CGLSLFunctionGenerator.cpp

# Scene objects
	CBillboardSceneNode.cpp
	CCameraSceneNode.cpp
	CCubeSceneNode.cpp
	../../src/irr/asset/CGeometryCreator.cpp
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
	../../src/irr/asset/CImageLoaderBMP.cpp
	../../src/irr/asset/CImageLoaderDDS.cpp
	../../src/irr/asset/CImageLoaderJPG.cpp
	../../src/irr/asset/CImageLoaderPNG.cpp
	../../src/irr/asset/CImageLoaderTGA.cpp
	../../src/irr/asset/CImageWriterBMP.cpp
	../../src/irr/asset/CImageWriterJPG.cpp
	../../src/irr/asset/CImageWriterPNG.cpp
	../../src/irr/asset/CImageWriterTGA.cpp

# Video
	CFPSCounter.cpp
	CVideoModeList.cpp
	../../src/irr/video/GPUMemoryAllocatorBase.cpp
	../../src/irr/video/SimpleGPUBufferAllocator.cpp

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
	
# PCH
	pch.cpp
)

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
	add_dependencies(Irrlicht shaderc)
	target_link_libraries(${_trgt} INTERFACE
		shaderc
	)
	target_include_directories(${_trgt} PUBLIC ../../3rdparty/shaderc/libshaderc/include)
endmacro()
macro(irr_target_link_libjpeg _trgt)
	add_dependencies(Irrlicht jpeg)
	target_link_libraries(${_trgt} INTERFACE
		jpeg
	)
	target_include_directories(${_trgt} PUBLIC ../../3rdparty/libjpeg)
endmacro()
macro(irr_target_link_libpng _trgt)
	add_dependencies(Irrlicht png_static $<TARGET_OBJECTS:zlib>)
	target_link_libraries(${_trgt} INTERFACE
		png_static $<TARGET_OBJECTS:zlib>
	)
	target_include_directories(${_trgt} PUBLIC ../../3rdparty/libpng)
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
	)
endif()

target_include_directories(Irrlicht PUBLIC 
	../../include
	${COMMON_INCLUDE_DIRS}
	${THIRD_PARTY_SOURCE_DIR}
)
target_include_directories(Irrlicht PUBLIC ../../src ../../source/Irrlicht) # this should be PRIVATE, but things from /src (or /source) are sometimes included in things in /include and so examples have to put source dirs into theirs Include Path
if (UNIX)
	target_include_directories(Irrlicht PUBLIC 
		${X11_INCLUDE_DIR} 
		${X11_Xrandr_INCLUDE_PATH} 
		${X11_xf86vmode_INCLUDE_PATH})
endif()

find_package(Vulkan)

if (Vulkan_FOUND)
	message(STATUS "Building with Vulkan driver")
	target_compile_definitions(Irrlicht PRIVATE
		_IRR_COMPILE_WITH_VULKAN_
	)
	#target_link_libraries(Irrlicht INTERFACE Vulkan::Vulkan) # this should work, but it doesn't
	#target_link_libraries(Irrlicht INTERFACE ${Vulkan_LIBRARY})
else()
	message(STATUS "Vulkan driver is not enabled")
endif()

target_compile_definitions(Irrlicht PRIVATE
	_IRR_STATIC_LIB_
	_IRR_COMPILE_WITH_OPENGL_
)

if(IRR_PCH)
  ADD_PRECOMPILED_HEADER(Irrlicht pch.h FORCEINCLUDE)
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

if (NOT MSVC)
	target_compile_options(Irrlicht PRIVATE -fexceptions)
	target_compile_options(IrrlichtServer PRIVATE -fexceptions)
endif()

add_dependencies(IrrlichtServer openssl_build)
irr_target_link_openssl(IrrlichtServer)
irr_target_link_shaderc(IrrlichtServer)
irr_target_link_libjpeg(IrrlichtServer)
irr_target_link_libpng(IrrlichtServer)

target_include_directories(IrrlichtServer PUBLIC 
	../../include
	${COMMON_INCLUDE_DIRS}
	${THIRD_PARTY_SOURCE_DIR}
)
target_include_directories(IrrlichtServer PRIVATE ../../src ../../source/Irrlicht)

target_compile_definitions(IrrlichtServer PRIVATE
	_IRR_STATIC_LIB_
	BAW_SERVER
	NO_IRR_COMPILE_WITH_BURNINGSVIDEO_
	NO_IRR_COMPILE_WITH_OPENGL_
)

if(IRR_PCH)
  ADD_PRECOMPILED_HEADER(IrrlichtServer pch.h FORCEINCLUDE)
endif()

if(MSVC)
	set (OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/../../lib/Win64-visualStudio")
elseif(WIN32)
	set (OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/../../lib/Win64-gcc")
else()
	if (APPLE)
		set (OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/../../lib/MacOSX")
	elseif(UNIX)
		set (OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/../../lib/Linux")
	else()
		set (OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/../../lib/unknown_platform")
	endif()
endif()

macro(irr_set_irrlicht_output_dir _TRGT_NAME _OUTPATH)
	set_target_properties(${_TRGT_NAME}
		PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY "${_OUTPATH}"
	)
	if(MSVC)
		set_target_properties(${_TRGT_NAME}
			PROPERTIES
			RUNTIME_OUTPUT_DIRECTORY_DEBUG "${_OUTPATH}"
			RUNTIME_OUTPUT_DIRECTORY_RELEASE "${_OUTPATH}"
		)
	endif()
endmacro()
irr_set_irrlicht_output_dir(Irrlicht "${OUTPUT_DIR}")
irr_set_irrlicht_output_dir(IrrlichtServer "${OUTPUT_DIR}")
set_target_properties(Irrlicht PROPERTIES DEBUG_POSTFIX _debug)
set_target_properties(IrrlichtServer PROPERTIES DEBUG_POSTFIX _debug)

install(TARGETS Irrlicht DESTINATION .)
install(TARGETS IrrlichtServer DESTINATION .)
