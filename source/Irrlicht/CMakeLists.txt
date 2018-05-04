cmake_minimum_required(VERSION 3.0)
project(Irrlicht)

set(CMAKE_CXX_STANDARD 11)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type: Debug or Release" FORCE)
endif()

option(FAST_MATH "Enable fast low-precision math" ON)

find_package(OpenSSL REQUIRED)

set(IRRLICHT_SRCS_COMMON
# Mesh loaders
	C3DSMeshFileLoader.cpp
	CB3DMeshFileLoader.cpp
	CBAWMeshFileLoader.cpp
	CCSMLoader.cpp
	CDMFLoader.cpp
	CLMTSMeshFileLoader.cpp
	CLWOMeshFileLoader.cpp
	CMS3DMeshFileLoader.cpp
	CMY3DMeshFileLoader.cpp
	COBJMeshFileLoader.cpp
	COCTLoader.cpp
	COgreMeshFileLoader.cpp
	CPLYMeshFileLoader.cpp
	CSMFMeshFileLoader.cpp
	CSTLMeshFileLoader.cpp
	CXMeshFileLoader.cpp

# Mesh writers
	CBAWMeshWriter.cpp
	COBJMeshWriter.cpp
	CPLYMeshWriter.cpp
	CSTLMeshWriter.cpp

# Other mesh-related stuff
	CAnimatedMeshHalfLife.cpp
	CAnimatedMeshSceneNode.cpp
	CBAWFile.cpp
	CBlobsLoadingManager.cpp
	CMeshSceneNode.cpp
	CMeshSceneNodeInstanced.cpp
	CSkinnedMesh.cpp
	CSkinnedMeshSceneNode.cpp
	TypedBlob.cpp

# Scene objects
	CBillboardSceneNode.cpp
	CCameraSceneNode.cpp
	CCubeSceneNode.cpp
	CDefaultSceneNodeAnimatorFactory.cpp
	CDefaultSceneNodeFactory.cpp
	CEmptySceneNode.cpp
	CGeometryCreator.cpp
	CMeshCache.cpp
	CMeshManipulator.cpp
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
	COpenGLRenderBuffer.cpp
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
	CImageLoaderPCX.cpp
	CImageLoaderPNG.cpp
	CImageLoaderPPM.cpp
	CImageLoaderPSD.cpp
	CImageLoaderRGB.cpp
	CImageLoaderTGA.cpp
	CImageWriterBMP.cpp
	CImageWriterJPG.cpp
	CImageWriterPCX.cpp
	CImageWriterPNG.cpp
	CImageWriterPPM.cpp
	CImageWriterPSD.cpp
	CImageWriterTGA.cpp

# Video
	CFPSCounter.cpp
	CVideoModeList.cpp

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
	CIrrDeviceSDL.cpp
	CIrrDeviceLinux.cpp
	CIrrDeviceConsole.cpp
	CIrrDeviceStub.cpp
	CIrrDeviceWin32.cpp
	CLogger.cpp
	COSOperator.cpp
	Irrlicht.cpp
	os.cpp
)

set(IRRLICHT_SRCS_NORMAL
	CGPUTransientBuffer.cpp
	COpenCLHandler.cpp
	COpenGL2DTextureArray.cpp
	COpenGL2DTexture.cpp
	COpenGL3DTexture.cpp
	COpenGLCubemapTexture.cpp
	COpenGLMultisampleTextureArray.cpp
	COpenGLMultisampleTexture.cpp
	COpenGLOcclusionQuery.cpp
	COpenGLPersistentlyMappedBuffer.cpp
	COpenGLQuery.cpp
	COpenGLTransformFeedback.cpp
	COpenGLVAO.cpp
	CTRTextureGouraudAlpha.cpp
	CTRTextureGouraudAlphaNoZ.cpp
	coreutil.cpp
)

set(IRRLICHT_SRCS_SERVER
)

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

add_library(bzip2 OBJECT
	bzip2/blocksort.c
	bzip2/bzlib.c
	bzip2/compress.c
	bzip2/crctable.c
	bzip2/decompress.c
	bzip2/huffman.c
	bzip2/randtable.c
)

add_library(convert_utf OBJECT
	convert_utf/ConvertUTF.c
)

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

add_library(lz4 OBJECT
	lz4/lz4.c
	lz4/lz4frame.c
	lz4/lz4hc.c
	lz4/xxhash.c
)

add_library(lzma OBJECT
	lzma/Alloc.c
	lzma/LzFind.c
	lzma/LzmaDec.c
	lzma/LzmaEnc.c
)
target_compile_definitions(lzma PRIVATE _7ZIP_ST)

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

if(WIN32)
	add_definitions(
		"WIN32"
		__GNUWIN32__
		_CRT_SECURE_NO_DEPRECATE
	)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
	add_compile_options(
		-W
		-Wall
		-Wextra
		-Wno-unused-parameter
		-fno-exceptions
		-msse3
	)
	if($<CONFIG:Debug>)
		add_compile_options(
			-fno-omit-frame-pointer
			-fstack-protector-strong
		)
	endif()
	if(FAST_MATH)
		add_compile_options(
			-fexpensive-optimizations
			-ffast-math
			-mfpmath=sse
		)
	endif()
endif()

link_libraries(
	OpenSSL::Crypto
	OpenSSL::SSL
)

include_directories(
	convert_utf
	libpng
	utf8/source
	zlib
)

add_definitions(
	$<$<CONFIG:Debug>:_DEBUG>
	$<$<BOOL:FAST_MATH>:IRRLICHT_FAST_MATH>
)

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
)

target_include_directories(Irrlicht PUBLIC ../../include)

target_compile_definitions(Irrlicht PRIVATE
	_IRR_STATIC_LIB_
	_IRR_COMPILE_WITH_OPENGL_
)

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

target_include_directories(IrrlichtServer PUBLIC ../../include)

target_compile_definitions(IrrlichtServer PRIVATE
	_IRR_STATIC_LIB_
	BAW_SERVER
	NO_IRR_COMPILE_WITH_BURNINGSVIDEO_
	NO_IRR_COMPILE_WITH_OPENGL_
)
