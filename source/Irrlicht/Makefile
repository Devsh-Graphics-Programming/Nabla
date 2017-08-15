VERSION_MAJOR = 1
VERSION_MINOR = 8
VERSION_RELEASE = 3
# Irrlicht Engine 1.8.3
# Makefile for Linux
#
# To use, just run:
#
# make
#
# This will compile Irrlicht, create a static lib (libIrrlicht.a), and copy it
# into the subdirectory lib/Linux. That's all.
#
# If you want Irrlicht to be compiled as shared lib (libIrrlicht.so.versionnumber), then run:
#
# make sharedlib
# make install
#
# If you want to compile in release mode run:
#
# make NDEBUG=1
#
# For cross-compilation for Win32 under Linux, just use the win32 targets. You have to set
# at least CXX, CC, and AR to the proper binaries.
#
# For compiling on MinGW you can run it with:
#
# make win32
#

#List of object files, separated based on engine architecture
IRRMESHLOADER = CMS3DMeshFileLoader.o CB3DMeshFileLoader.o C3DSMeshFileLoader.o COgreMeshFileLoader.o COBJMeshFileLoader.o CCSMLoader.o CDMFLoader.o CLMTSMeshFileLoader.o CMY3DMeshFileLoader.o COCTLoader.o CXMeshFileLoader.o CSTLMeshFileLoader.o CLWOMeshFileLoader.o CPLYMeshFileLoader.o CSMFMeshFileLoader.o
IRRMESHWRITER = CSTLMeshWriter.o COBJMeshWriter.o CPLYMeshWriter.o
IRRMESHOBJ = $(IRRMESHLOADER) $(IRRMESHWRITER) \
	CSkinnedMesh.o CMeshSceneNode.o CMeshSceneNodeInstanced.o \
	CSkinnedMeshSceneNode.o \
	CAnimatedMeshSceneNode.o \
	CAnimatedMeshHalfLife.o
IRROBJ = CBillboardSceneNode.o CCameraSceneNode.o CEmptySceneNode.o CGeometryCreator.o CMeshManipulator.o CSceneManager.o CSkyBoxSceneNode.o CSkyDomeSceneNode.o CCubeSceneNode.o CSphereSceneNode.o CMeshCache.o CDefaultSceneNodeAnimatorFactory.o CDefaultSceneNodeFactory.o
IRRANIMOBJ = CSceneNodeAnimatorCameraFPS.o CSceneNodeAnimatorCameraMaya.o CSceneNodeAnimatorDelete.o CSceneNodeAnimatorFlyCircle.o CSceneNodeAnimatorFlyStraight.o CSceneNodeAnimatorFollowSpline.o CSceneNodeAnimatorRotation.o CSceneNodeAnimatorTexture.o
IRRDRVROBJ = FW_Mutex.o CNullDriver.o COpenGLDriver.o COpenGLTexture.o COpenGLSLMaterialRenderer.o COpenGLExtensionHandler.o COpenGLFrameBuffer.o COpenGLRenderBuffer.o STextureSamplingParams.o
IRRIMAGEOBJ = CColorConverter.o CImage.o CImageLoaderBMP.o CImageLoaderDDS.o CImageLoaderJPG.o CImageLoaderPCX.o CImageLoaderPNG.o CImageLoaderPSD.o CImageLoaderTGA.o CImageLoaderPPM.o CImageLoaderWAL.o CImageLoaderRGB.o \
	CImageWriterBMP.o CImageWriterJPG.o CImageWriterPCX.o CImageWriterPNG.o CImageWriterPPM.o CImageWriterPSD.o CImageWriterTGA.o
IRRVIDEOOBJ = CVideoModeList.o CFPSCounter.o $(IRRDRVROBJ) $(IRRIMAGEOBJ)
IRRSWRENDEROBJ = CSoftwareDriver.o CSoftwareTexture.o CTRFlat.o CTRFlatWire.o CTRGouraud.o CTRGouraudWire.o CTRNormalMap.o CTRStencilShadow.o CTRTextureFlat.o CTRTextureFlatWire.o CTRTextureGouraud.o CTRTextureGouraudAdd.o CTRTextureGouraudNoZ.o CTRTextureGouraudWire.o CZBuffer.o CTRTextureGouraudVertexAlpha2.o CTRTextureGouraudNoZ2.o CTRTextureLightMap2_M2.o CTRTextureLightMap2_M4.o CTRTextureLightMap2_M1.o CSoftwareDriver2.o CSoftwareTexture2.o CTRTextureGouraud2.o CTRGouraud2.o CTRGouraudAlpha2.o CTRGouraudAlphaNoZ2.o CTRTextureDetailMap2.o CTRTextureGouraudAdd2.o CTRTextureGouraudAddNoZ2.o CTRTextureWire2.o CTRTextureLightMap2_Add.o CTRTextureLightMapGouraud2_M4.o IBurningShader.o CTRTextureBlend.o CTRTextureGouraudAlpha.o CTRTextureGouraudAlphaNoZ.o CDepthBuffer.o CBurningShader_Raster_Reference.o
IRRIOOBJ = CFileList.o CFileSystem.o CLimitReadFile.o CMemoryFile.o CReadFile.o CWriteFile.o CWADReader.o CZipReader.o CPakReader.o CNPKReader.o CTarReader.o CMountPointReader.o lzma/LzmaDec.o
IRROTHEROBJ = CIrrDeviceSDL.o CIrrDeviceLinux.o CIrrDeviceConsole.o CIrrDeviceStub.o CIrrDeviceWin32.o CIrrDeviceFB.o CLogger.o COSOperator.o Irrlicht.o os.o
ZLIBOBJ = zlib/adler32.o zlib/compress.o zlib/crc32.o zlib/deflate.o zlib/inffast.o zlib/inflate.o zlib/inftrees.o zlib/trees.o zlib/uncompr.o zlib/zutil.o
JPEGLIBOBJ = jpeglib/jcapimin.o jpeglib/jcapistd.o jpeglib/jccoefct.o jpeglib/jccolor.o jpeglib/jcdctmgr.o jpeglib/jchuff.o jpeglib/jcinit.o jpeglib/jcmainct.o jpeglib/jcmarker.o jpeglib/jcmaster.o jpeglib/jcomapi.o jpeglib/jcparam.o jpeglib/jcprepct.o jpeglib/jcsample.o jpeglib/jctrans.o jpeglib/jdapimin.o jpeglib/jdapistd.o jpeglib/jdatadst.o jpeglib/jdatasrc.o jpeglib/jdcoefct.o jpeglib/jdcolor.o jpeglib/jddctmgr.o jpeglib/jdhuff.o jpeglib/jdinput.o jpeglib/jdmainct.o jpeglib/jdmarker.o jpeglib/jdmaster.o jpeglib/jdmerge.o jpeglib/jdpostct.o jpeglib/jdsample.o jpeglib/jdtrans.o jpeglib/jerror.o jpeglib/jfdctflt.o jpeglib/jfdctfst.o jpeglib/jfdctint.o jpeglib/jidctflt.o jpeglib/jidctfst.o jpeglib/jidctint.o jpeglib/jmemmgr.o jpeglib/jmemnobs.o jpeglib/jquant1.o jpeglib/jquant2.o jpeglib/jutils.o jpeglib/jcarith.o jpeglib/jdarith.o jpeglib/jaricom.o
LIBPNGOBJ = libpng/png.o libpng/pngerror.o libpng/pngget.o libpng/pngmem.o libpng/pngpread.o libpng/pngread.o libpng/pngrio.o libpng/pngrtran.o libpng/pngrutil.o libpng/pngset.o libpng/pngtrans.o libpng/pngwio.o libpng/pngwrite.o libpng/pngwtran.o libpng/pngwutil.o
LIBAESGM = aesGladman/aescrypt.o aesGladman/aeskey.o aesGladman/aestab.o aesGladman/fileenc.o aesGladman/hmac.o aesGladman/prng.o aesGladman/pwd2key.o aesGladman/sha1.o aesGladman/sha2.o
BZIP2OBJ = bzip2/blocksort.o bzip2/huffman.o bzip2/crctable.o bzip2/randtable.o bzip2/bzcompress.o bzip2/decompress.o bzip2/bzlib.o
UTFOBJ = coreutil.o convert_utf/ConvertUTF.o

# Next variable is for additional scene nodes etc. of customized Irrlicht versions
EXTRAOBJ =
LINKOBJ = $(IRRMESHOBJ) $(IRROBJ) $(IRRANIMOBJ) \
	$(IRRVIDEOOBJ) $(IRRSWRENDEROBJ) $(IRRIOOBJ) $(IRROTHEROBJ) \
	$(ZLIBOBJ) $(JPEGLIBOBJ) $(LIBPNGOBJ) $(LIBAESGM) \
	$(BZIP2OBJ) $(UTFOBJ) $(EXTRAOBJ)

###############
#Compiler flags
CXXINCS = -I../../include -Izlib -Ijpeglib -Ilibpng -Iconvert_utf -Iutf8/source
CPPFLAGS += $(CXXINCS) -DIRRLICHT_EXPORTS=1
CXXFLAGS += -std=c++03 -Wall -pipe -fno-exceptions -fno-rtti -fstrict-aliasing
ifndef NDEBUG
CXXFLAGS += -g -D_DEBUG
else
CXXFLAGS += -fexpensive-optimizations -O3 -msse3
endif
ifdef INTELCC
CXXFLAGS += -axCORE-AVX2,CORE-AVX-I,AVX -msse4.2 -opt-prefetch -multiple-processes=8 -prec-sqrt -fast -rcd -ip -ipo -ipo-jobs8 -fma -fp-relaxed -m64 -opt-loadpair 
endif
ifdef BAW_SERVER
CXXFLAGS += -DBAW_SERVER
ifndef INTELCC
CXXFLAGS += -march=native
endif
endif
ifdef BAW_SERVER_RELEASE
CXXFLAGS += -DBAW_SERVER
ifndef INTELCC
CXXFLAGS +=
CPPFLAGS +=
endif
endif
ifdef PROFILE
CXXFLAGS += -pg
endif
CFLAGS := -O3 -fexpensive-optimizations -DPNG_THREAD_UNSAFE_OK -DPNG_NO_MMX_CODE -DPNG_NO_MNG_FEATURES

sharedlib sharedlib_osx: CXXFLAGS += -fPIC
sharedlib sharedlib_osx: CFLAGS += -fPIC

#multilib handling
ifeq ($(HOSTTYPE), x86_64)
LIBSELECT=64
endif

#Linux specific options
staticlib serverlib sharedlib install: SYSTEM = Linux
ifndef BAW_SERVER
ifndef BAW_SERVER_RELEASE
STATIC_LIB = libIrrlicht.a
else
STATIC_LIB = libIrrlichtServer.a
endif
else
STATIC_LIB = libIrrlichtServer.a
endif
LIB_PATH = ../../lib/$(SYSTEM)
INSTALL_DIR = /usr/local/lib
sharedlib install: SHARED_LIB = libIrrlicht.so
#staticlib sharedlib: LDFLAGS += --no-export-all-symbols --add-stdcall-alias
sharedlib: LDFLAGS += -fuse-ld=gold -L/usr/X11R6/lib$(LIBSELECT) -lGL -lXxf86vm
#might lose the X11 for serverlib
serverlib staticlib sharedlib: CXXINCS += -I/usr/X11R6/include

#OSX specific options
staticlib_osx sharedlib_osx install_osx: SYSTEM = MacOSX
staticlib_osx sharedlib_osx: IRROTHEROBJ += MacOSX/CIrrDeviceMacOSX.o MacOSX/OSXClipboard.o MacOSX/AppDelegate.o
staticlib_osx sharedlib_osx: CXXINCS += -IMacOSX -I/usr/X11R6/include
sharedlib_osx install_osx: SHARED_LIB = libIrrlicht.dylib
staticlib_osx sharedlib_osx: LDFLAGS += --no-export-all-symbols --add-stdcall-alias
sharedlib_osx: LDFLAGS += -L/usr/X11R6/lib$(LIBSELECT) -lGL -lXxf86vm
# for non-X11 app
#sharedlib_osx: LDFLAGS += -framework cocoa -framework carbon -framework opengl -framework IOKit

#Windows specific options
IRRLICHT_DLL := ../../bin/Win32-gcc/Irrlicht.dll
sharedlib_win32 staticlib_win32: SYSTEM = Win32-gcc
sharedlib_win32: LDFLAGS += -lgdi32 -lopengl32 -lwinmm -Wl,--add-stdcall-alias
#choose either -DIRR_COMPILE_WITH_DX9_DEV_PACK or -DNO_IRR_COMPILE_WITH_DIRECT3D_9_ depending if you need dx9
#sharedlib_win32 staticlib_win32: CPPFLAGS += -DIRR_COMPILE_WITH_DX9_DEV_PACK
sharedlib_win32 staticlib_win32: CPPFLAGS += -DNO_IRR_COMPILE_WITH_DIRECTINPUT_JOYSTICK_ -DNO_IRR_COMPILE_WITH_DIRECT3D_9_
sharedlib_win32 staticlib_win32: CPPFLAGS += -DIRR_COMPILE_WITH_DX9_DEV_PACK -D__GNUWIN32__ -D_WIN32 -DWIN32 -D_WINDOWS -D_MBCS -D_USRDLL
staticlib_win32: CPPFLAGS += -D_IRR_STATIC_LIB_

VERSION = $(VERSION_MAJOR).$(VERSION_MINOR).$(VERSION_RELEASE)
SHARED_FULLNAME = $(SHARED_LIB).$(VERSION)
SONAME = $(SHARED_LIB).$(VERSION_MAJOR).$(VERSION_MINOR)

####################
# All target, builds Irrlicht as static lib (libIrrlicht.a) and copies it into lib/Linux
all linux: staticlib

# Builds Irrlicht as shared lib (libIrrlicht.so.versionNumber) and copies it into lib/Linux
sharedlib: $(LINKOBJ)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -shared -Wl,-soname,$(SONAME) -o $(SHARED_FULLNAME) $^ $(LDFLAGS)
	mkdir -p $(LIB_PATH)
	cp $(SHARED_FULLNAME) $(LIB_PATH)

# Builds Irrlicht as static lib (libIrrlicht.a)
$(STATIC_LIB): $(LINKOBJ)
	$(AR) rs $@ $^

# Copies static lib into lib/Linux
serverlib staticlib staticlib_osx: $(STATIC_LIB)
	mkdir -p $(LIB_PATH)
	cp $^ $(LIB_PATH)

# Builds Irrlicht as dll (Irrlicht.dll) into ../../bin/Win32-gcc
all_win32 win32: sharedlib_win32
sharedlib_win32: $(IRRLICHT_DLL)
../../bin/Win32-gcc/Irrlicht.dll: $(LINKOBJ)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -shared -o $@ $^ $(LDFLAGS) -Wl,--out-implib,../../lib/Win32-gcc/$(STATIC_LIB)
# Copies static lib into /lib/Win32-gcc
staticlib_win32: $(STATIC_LIB)
	cp $^ $(LIB_PATH)

# Builds Irrlicht as shared lib (libIrrlicht.so.versionNumber) and copies it into /lib/MacOSX
sharedlib_osx: $(LINKOBJ)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -dynamiclib -Wl,-install_name,$(SONAME) -o $(SHARED_FULLNAME) $^ $(LDFLAGS)
	cp $(SHARED_FULLNAME) $(LIB_PATH)

# Installs Irrlicht if it was created as shared lib
install install_osx:
	$(RM) -r $(INSTALL_DIR)/../include/irrlicht
	mkdir -p $(INSTALL_DIR)/../include/irrlicht
	cp ../../include/*.h $(INSTALL_DIR)/../include/irrlicht/
	cp $(LIB_PATH)/$(SHARED_FULLNAME) $(INSTALL_DIR)
	cd $(INSTALL_DIR) && ln -s -f $(SHARED_FULLNAME) $(SONAME)
	cd $(INSTALL_DIR) && ln -s -f $(SONAME) $(SHARED_LIB)
#	ldconfig -n $(INSTALL_DIR)

TAGS:
	ctags *.cpp ../../include/*.h *.h

# Create dependency files for automatic recompilation
%.d:%.cpp
	$(CXX) $(CPPFLAGS) -MM -MF $@ $<

# Create dependency files for automatic recompilation
%.d:%.c
	$(CC) $(CPPFLAGS) -MM -MF $@ $<

# Create object files from objective-c code
%.o:%.mm
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

ifneq ($(MAKECMDGOALS),clean)
-include $(LINKOBJ:.o=.d)
endif

help:
	@echo "Available targets for Irrlicht"
	@echo " sharedlib: Build shared library Irrlicht.so for Linux"
	@echo " staticlib: Build static library Irrlicht.a for Linux"
	@echo " serverlib: Build BAWServer static library Irrlicht.a for Linux"
	@echo " install: Copy shared library to /usr/local/lib"
	@echo ""
	@echo " sharedlib_win32: Build shared library Irrlicht.dll for Windows"
	@echo " staticlib_win32: Build static library Irrlicht.a for Windows"
	@echo ""
	@echo " clean: Clean up directory"

# Cleans all temporary files and compilation results.
clean:
	$(RM) $(LINKOBJ) $(SHARED_FULLNAME) $(STATIC_LIB) $(LINKOBJ:.o=.d)

.PHONY: all sharedlib staticlib serverlib sharedlib_win32 staticlib_win32 help install clean
 
