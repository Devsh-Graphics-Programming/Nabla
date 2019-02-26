find_package(OpenSSL REQUIRED)
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
	CAnimatedMeshSceneNode.cpp
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
	../../src/irr/asset/CImageLoaderRGB.cpp
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
	spirv_cross/spirv_parser.cpp
	spirv_cross/spirv_cross_parsed_ir.cpp
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
	target_link_libraries(${_trgt} INTERFACE
		OpenSSL::Crypto
		OpenSSL::SSL
	)
	target_include_directories(${_trgt} PUBLIC ${OPENSSL_INCLUDE_DIR})
endmacro()
macro(irr_target_link_shaderc _trgt)
	add_dependencies(Irrlicht shaderc)
	target_link_libraries(${_trgt} INTERFACE
		shaderc
	)
	target_include_directories(${_trgt} PUBLIC ../../3rdparty/shaderc/libshaderc/include)
endmacro()

add_library(Irrlicht STATIC
	${IRRLICHT_SRCS_COMMON}
	${IRRLICHT_SRCS_NORMAL}
	${IRRLICHT_HEADERS}
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
irr_target_link_shaderc(Irrlicht)

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

add_library(IrrlichtServer STATIC
	${IRRLICHT_SRCS_COMMON}
	${IRRLICHT_SRCS_SERVER}
	${IRRLICHT_HEADERS}
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
irr_target_link_shaderc(IrrlichtServer)

target_include_directories(IrrlichtServer PUBLIC 
	../../include
	${COMMON_INCLUDE_DIRS}
)
target_include_directories(IrrlichtServer PRIVATE ../../src ../../source/Irrlicht)

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
