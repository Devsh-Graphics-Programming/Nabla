
if(NOT NASTY_OPENSSL_WORKAROUND)
	find_package(OpenSSL REQUIRED)
endif()

find_package(OpenGL REQUIRED)
if(UNIX)
	find_package(X11 REQUIRED)
	set(CMAKE_THREAD_PREFER_PTHREAD 1)
	find_package(Threads REQUIRED)
endif()

set(IRRLICHT_SRCS_COMMON
# Mesh loaders
	C3DSMeshFileLoader.cpp
	CB3DMeshFileLoader.cpp
	CBAWMeshFileLoader.cpp
	CLMTSMeshFileLoader.cpp
	CLWOMeshFileLoader.cpp
	CMS3DMeshFileLoader.cpp
	CMY3DMeshFileLoader.cpp
	COBJMeshFileLoader.cpp
	COCTLoader.cpp
	COgreMeshFileLoader.cpp
	CPLYMeshFileLoader.cpp
	CSTLMeshFileLoader.cpp
	CXMeshFileLoader.cpp

# Mesh writers
	CBAWMeshWriter.cpp
	COBJMeshWriter.cpp
	CPLYMeshWriter.cpp
	CSTLMeshWriter.cpp

# Other mesh-related stuff
	CAnimatedMeshSceneNode.cpp
	CBAWFile.cpp
	CBlobsLoadingManager.cpp
	CForsythVertexCacheOptimizer.cpp
	CMeshCache.cpp
	CMeshManipulator.cpp
	CMeshSceneNode.cpp
	CMeshSceneNodeInstanced.cpp
	COverdrawMeshOptimizer.cpp
	CSkinnedMesh.cpp
	CSkinnedMeshSceneNode.cpp
	TypedBlob.cpp

# Scene objects
	CBillboardSceneNode.cpp
	CCameraSceneNode.cpp
	CCubeSceneNode.cpp
	CGeometryCreator.cpp
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
	CImageLoaderBMP.cpp
	CImageLoaderDDS.cpp
	CImageLoaderJPG.cpp
	CImageLoaderPNG.cpp
	CImageLoaderRGB.cpp
	CImageLoaderTGA.cpp
	CImageWriterBMP.cpp
	CImageWriterJPG.cpp
	CImageWriterPNG.cpp
	CImageWriterTGA.cpp

# Video
	CFPSCounter.cpp
	CVideoModeList.cpp
	../../src/irr/video/GPUMemoryAllocatorBase.cpp
	../../src/irr/video/StreamingGPUBufferAllocator.cpp

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
	convert_utf
	libpng
	utf8/source
	zlib
)

irr_adjust_flags() # macro defined in root CMakeLists
irr_adjust_definitions() # macro defined in root CMakeLists

add_library(aesGladman OBJECT
	aesGladman/aes_ni.c
	aesGladman/aescrypt.c
	aesGladman/aeskey.c
	aesGladman/aestab.c
	aesGladman/fileenc.c
	aesGladman/hmac.c
	aesGladman/prng.c
	aesGladman/pwd2key.c
	aesGladman/sha1.c
)
target_include_directories(aesGladman PUBLIC ${COMMON_INCLUDE_DIRS})

add_library(bzip2 OBJECT
	bzip2/blocksort.c
	bzip2/bzlib.c
	bzip2/compress.c
	bzip2/crctable.c
	bzip2/decompress.c
	bzip2/huffman.c
	bzip2/randtable.c
)
target_include_directories(bzip2 PUBLIC ${COMMON_INCLUDE_DIRS})

add_library(convert_utf OBJECT
	convert_utf/ConvertUTF.c
)
target_include_directories(convert_utf PUBLIC ${COMMON_INCLUDE_DIRS})

add_library(jpeglib OBJECT
	jpeglib/jaricom.c
	jpeglib/jcapimin.c
	jpeglib/jcapistd.c
	jpeglib/jcarith.c
	jpeglib/jccoefct.c
	jpeglib/jccolor.c
	jpeglib/jcdctmgr.c
	jpeglib/jchuff.c
	jpeglib/jcinit.c
	jpeglib/jcmainct.c
	jpeglib/jcmarker.c
	jpeglib/jcmaster.c
	jpeglib/jcomapi.c
	jpeglib/jcparam.c
	jpeglib/jcprepct.c
	jpeglib/jcsample.c
	jpeglib/jctrans.c
	jpeglib/jdapimin.c
	jpeglib/jdapistd.c
	jpeglib/jdarith.c
	jpeglib/jdatadst.c
	jpeglib/jdatasrc.c
	jpeglib/jdcoefct.c
	jpeglib/jdcolor.c
	jpeglib/jddctmgr.c
	jpeglib/jdhuff.c
	jpeglib/jdinput.c
	jpeglib/jdmainct.c
	jpeglib/jdmarker.c
	jpeglib/jdmaster.c
	jpeglib/jdmerge.c
	jpeglib/jdpostct.c
	jpeglib/jdsample.c
	jpeglib/jdtrans.c
	jpeglib/jerror.c
	jpeglib/jfdctflt.c
	jpeglib/jfdctfst.c
	jpeglib/jfdctint.c
	jpeglib/jidctflt.c
	jpeglib/jidctfst.c
	jpeglib/jidctint.c
	jpeglib/jmemmgr.c
	jpeglib/jmemnobs.c
	jpeglib/jquant1.c
	jpeglib/jquant2.c
	jpeglib/jutils.c
)
target_include_directories(jpeglib PUBLIC ${COMMON_INCLUDE_DIRS})

add_library(libpng OBJECT
	libpng/png.c
	libpng/pngerror.c
	libpng/pngget.c
	libpng/pngmem.c
	libpng/pngpread.c
	libpng/pngread.c
	libpng/pngrio.c
	libpng/pngrtran.c
	libpng/pngrutil.c
	libpng/pngset.c
	libpng/pngtrans.c
	libpng/pngwio.c
	libpng/pngwrite.c
	libpng/pngwtran.c
	libpng/pngwutil.c
)
target_include_directories(libpng PUBLIC ${COMMON_INCLUDE_DIRS})

add_library(lz4 OBJECT
	lz4/lz4.c
	lz4/lz4frame.c
	lz4/lz4hc.c
	lz4/xxhash.c
)
target_include_directories(lz4 PUBLIC ${COMMON_INCLUDE_DIRS})

add_library(lzma OBJECT
	lzma/Alloc.c
	lzma/LzFind.c
	lzma/LzmaDec.c
	lzma/LzmaEnc.c
)
target_compile_definitions(lzma PRIVATE _7ZIP_ST)
target_include_directories(lzma PUBLIC ${COMMON_INCLUDE_DIRS})

add_library(zlib OBJECT
	zlib/adler32.c
	zlib/crc32.c
	zlib/deflate.c
	zlib/infback.c
	zlib/inffast.c
	zlib/inflate.c
	zlib/inftrees.c
	zlib/trees.c
	zlib/uncompr.c
	zlib/zcompress.c
	zlib/zutil.c
)
target_include_directories(zlib PUBLIC ${COMMON_INCLUDE_DIRS})

add_library(spirv_cross OBJECT
	spirv_cross/spirv_cfg.cpp
	spirv_cross/spirv_cross.cpp
	spirv_cross/spirv_glsl.cpp
	spirv_cross/spirv_reflect.cpp
)
target_compile_definitions(spirv_cross PRIVATE SPIRV_CROSS_EXCEPTIONS_TO_ASSERTIONS)
target_include_directories(spirv_cross PUBLIC ${COMMON_INCLUDE_DIRS})

add_library(libshaderc OBJECT
	libshaderc/compiler.cc
	libshaderc/file_finder.cc
	libshaderc/io.cc
	libshaderc/message.cc
	libshaderc/resources.cc
	libshaderc/shader_stage.cc
	libshaderc/shaderc.cc
	libshaderc/spirv_tools_wrapper.cc
	libshaderc/version_profile.cc
)
target_include_directories(libshaderc PUBLIC ${COMMON_INCLUDE_DIRS})

macro(irr_target_link_openssl _trgt)
	if(NOT NASTY_OPENSSL_WORKAROUND)
		target_link_libraries(${_trgt} INTERFACE
			OpenSSL::Crypto
			OpenSSL::SSL
		)
		target_include_directories(${_trgt} PUBLIC ${OPENSSL_INCLUDE_DIR})
	else()
		target_include_directories(${_trgt} PUBLIC ../../../openssl/include/)
		if (UNIX)
			target_link_libraries(${_trgt} INTERFACE
				../../../openssl/crypto.a
			)
		elseif (WIN32)
			target_link_libraries(${_trgt} INTERFACE
				../../../openssl/lib/libeay32.lib
			)
		endif()
	endif()
endmacro()

add_library(Irrlicht STATIC
	${IRRLICHT_SRCS_COMMON}
	${IRRLICHT_SRCS_NORMAL}
	$<TARGET_OBJECTS:aesGladman>
	$<TARGET_OBJECTS:bzip2>
	$<TARGET_OBJECTS:convert_utf>
	$<TARGET_OBJECTS:jpeglib>
	$<TARGET_OBJECTS:libpng>
	$<TARGET_OBJECTS:lz4>
	$<TARGET_OBJECTS:lzma>
	$<TARGET_OBJECTS:zlib>
	$<TARGET_OBJECTS:spirv_cross>
	$<TARGET_OBJECTS:libshaderc>
)

irr_target_link_openssl(Irrlicht)

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
)
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
	target_include_directories(Irrlicht INTERFACE ${Vulkan_INCLUDE_DIR})
	target_link_libraries(Irrlicht INTERFACE ${Vulkan_LIBRARY})
else()
	message(STATUS "Vulkan driver is not enabled")
endif()

target_compile_definitions(Irrlicht PRIVATE
	_IRR_STATIC_LIB_
	_IRR_COMPILE_WITH_OPENGL_
)

if(BUILD_EXAMPLES)
	add_custom_command(TARGET Irrlicht POST_BUILD
					   COMMAND ${CMAKE_COMMAND} -E 
					   copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/../../examples_tests/media" "${CMAKE_CURRENT_BINARY_DIR}/../../examples_tests/media"
	)
endif()

add_library(IrrlichtServer STATIC
	${IRRLICHT_SRCS_COMMON}
	${IRRLICHT_SRCS_SERVER}
	$<TARGET_OBJECTS:aesGladman>
	$<TARGET_OBJECTS:bzip2>
	$<TARGET_OBJECTS:convert_utf>
	$<TARGET_OBJECTS:jpeglib>
	$<TARGET_OBJECTS:libpng>
	$<TARGET_OBJECTS:lz4>
	$<TARGET_OBJECTS:lzma>
	$<TARGET_OBJECTS:zlib>
)

irr_target_link_openssl(IrrlichtServer)

target_include_directories(IrrlichtServer PUBLIC 
	../../include
	${COMMON_INCLUDE_DIRS}
)

target_compile_definitions(IrrlichtServer PRIVATE
	_IRR_STATIC_LIB_
	BAW_SERVER
	NO_IRR_COMPILE_WITH_BURNINGSVIDEO_
	NO_IRR_COMPILE_WITH_OPENGL_
)

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
