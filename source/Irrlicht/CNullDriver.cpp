// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "CNullDriver.h"
#include "os.h"
#include "CImage.h"
#include "IReadFile.h"
#include "IWriteFile.h"
#include "IImageLoader.h"
#include "IImageWriter.h"
#include "IMaterialRenderer.h"
#include "IAnimatedMeshSceneNode.h"
#include "CColorConverter.h"
#include "CMeshManipulator.h"
#include "CMeshSceneNodeInstanced.h"
#include "FW_Mutex.h"


namespace irr
{
namespace video
{

FW_AtomicCounter CNullDriver::ReallocationCounter = 0;

FW_AtomicCounter CNullDriver::incrementAndFetchReallocCounter()
{
#if _MSC_VER && !__INTEL_COMPILER
    return InterlockedIncrement(&ReallocationCounter);
#elif defined(__GNUC__)
    return __sync_add_and_fetch(&ReallocationCounter,int32_t(1));
#endif // _MSC_VER
}


//! creates a loader which is able to load windows bitmaps
IImageLoader* createImageLoaderBMP();

//! creates a loader which is able to load jpeg images
IImageLoader* createImageLoaderJPG();

//! creates a loader which is able to load targa images
IImageLoader* createImageLoaderTGA();

//! creates a loader which is able to load psd images
IImageLoader* createImageLoaderPSD();

//! creates a loader which is able to load dds images
IImageLoader* createImageLoaderDDS();

//! creates a loader which is able to load pcx images
IImageLoader* createImageLoaderPCX();

//! creates a loader which is able to load png images
IImageLoader* createImageLoaderPNG();

//! creates a loader which is able to load WAL images
IImageLoader* createImageLoaderWAL();

//! creates a loader which is able to load halflife images
IImageLoader* createImageLoaderHalfLife();

//! creates a loader which is able to load lmp images
IImageLoader* createImageLoaderLMP();

//! creates a loader which is able to load ppm/pgm/pbm images
IImageLoader* createImageLoaderPPM();

//! creates a loader which is able to load rgb images
IImageLoader* createImageLoaderRGB();


//! creates a writer which is able to save bmp images
IImageWriter* createImageWriterBMP();

//! creates a writer which is able to save jpg images
IImageWriter* createImageWriterJPG();

//! creates a writer which is able to save tga images
IImageWriter* createImageWriterTGA();

//! creates a writer which is able to save psd images
IImageWriter* createImageWriterPSD();

//! creates a writer which is able to save pcx images
IImageWriter* createImageWriterPCX();

//! creates a writer which is able to save png images
IImageWriter* createImageWriterPNG();

//! creates a writer which is able to save ppm images
IImageWriter* createImageWriterPPM();

//! constructor
CNullDriver::CNullDriver(io::IFileSystem* io, const core::dimension2d<u32>& screenSize)
: FileSystem(io), ViewPort(0,0,0,0), ScreenSize(screenSize), boxLineMesh(0),
	PrimitivesDrawn(0), MinVertexCountForVBO(500), TextureCreationFlags(0),
	OverrideMaterial2DEnabled(false), AllowZWriteOnTransparent(false),
	matrixModifiedBits(0)
{
	#ifdef _DEBUG
	setDebugName("CNullDriver");
	#endif

	for (size_t i=0; i<EQOT_COUNT; i++)
    for (size_t j=0; j<_IRR_XFORM_FEEDBACK_MAX_STREAMS_; j++)
        currentQuery[i][j] = NULL;

	setTextureCreationFlag(ETCF_ALWAYS_32_BIT, true);
	setTextureCreationFlag(ETCF_CREATE_MIP_MAPS, true);

	ViewPort = core::rect<s32>(core::position2d<s32>(0,0), core::dimension2di(screenSize));


	if (FileSystem)
		FileSystem->grab();

	// create surface loader
#ifdef _IRR_COMPILE_WITH_WAL_LOADER_
	SurfaceLoader.push_back(video::createImageLoaderHalfLife());
	SurfaceLoader.push_back(video::createImageLoaderWAL());
#endif
#ifdef _IRR_COMPILE_WITH_LMP_LOADER_
	SurfaceLoader.push_back(video::createImageLoaderLMP());
#endif
#ifdef _IRR_COMPILE_WITH_PPM_LOADER_
	SurfaceLoader.push_back(video::createImageLoaderPPM());
#endif
#ifdef _IRR_COMPILE_WITH_RGB_LOADER_
	SurfaceLoader.push_back(video::createImageLoaderRGB());
#endif
#ifdef _IRR_COMPILE_WITH_PSD_LOADER_
	SurfaceLoader.push_back(video::createImageLoaderPSD());
#endif
#ifdef _IRR_COMPILE_WITH_DDS_LOADER_
	SurfaceLoader.push_back(video::createImageLoaderDDS());
#endif
#ifdef _IRR_COMPILE_WITH_PCX_LOADER_
	SurfaceLoader.push_back(video::createImageLoaderPCX());
#endif
#ifdef _IRR_COMPILE_WITH_TGA_LOADER_
	SurfaceLoader.push_back(video::createImageLoaderTGA());
#endif
#ifdef _IRR_COMPILE_WITH_PNG_LOADER_
	SurfaceLoader.push_back(video::createImageLoaderPNG());
#endif
#ifdef _IRR_COMPILE_WITH_JPG_LOADER_
	SurfaceLoader.push_back(video::createImageLoaderJPG());
#endif
#ifdef _IRR_COMPILE_WITH_BMP_LOADER_
	SurfaceLoader.push_back(video::createImageLoaderBMP());
#endif


#ifdef _IRR_COMPILE_WITH_PPM_WRITER_
	SurfaceWriter.push_back(video::createImageWriterPPM());
#endif
#ifdef _IRR_COMPILE_WITH_PCX_WRITER_
	SurfaceWriter.push_back(video::createImageWriterPCX());
#endif
#ifdef _IRR_COMPILE_WITH_PSD_WRITER_
	SurfaceWriter.push_back(video::createImageWriterPSD());
#endif
#ifdef _IRR_COMPILE_WITH_TGA_WRITER_
	SurfaceWriter.push_back(video::createImageWriterTGA());
#endif
#ifdef _IRR_COMPILE_WITH_JPG_WRITER_
	SurfaceWriter.push_back(video::createImageWriterJPG());
#endif
#ifdef _IRR_COMPILE_WITH_PNG_WRITER_
	SurfaceWriter.push_back(video::createImageWriterPNG());
#endif
#ifdef _IRR_COMPILE_WITH_BMP_WRITER_
	SurfaceWriter.push_back(video::createImageWriterBMP());
#endif

    MaxTextureSizes[ITexture::ETT_1D][0] = 0x80u;
    MaxTextureSizes[ITexture::ETT_1D][1] = 0x1u;
    MaxTextureSizes[ITexture::ETT_1D][2] = 0x1u;

    MaxTextureSizes[ITexture::ETT_2D][0] = 0x80u;
    MaxTextureSizes[ITexture::ETT_2D][1] = 0x80u;
    MaxTextureSizes[ITexture::ETT_2D][2] = 0x1u;

    MaxTextureSizes[ITexture::ETT_3D][0] = 0x80u;
    MaxTextureSizes[ITexture::ETT_3D][1] = 0x80u;
    MaxTextureSizes[ITexture::ETT_3D][2] = 0x80u;

    MaxTextureSizes[ITexture::ETT_1D_ARRAY][0] = 0x80u;
    MaxTextureSizes[ITexture::ETT_1D_ARRAY][1] = 0x1u;
    MaxTextureSizes[ITexture::ETT_1D_ARRAY][2] = 0x800u;

    MaxTextureSizes[ITexture::ETT_2D_ARRAY][0] = 0x80u;
    MaxTextureSizes[ITexture::ETT_2D_ARRAY][1] = 0x80u;
    MaxTextureSizes[ITexture::ETT_2D_ARRAY][2] = 0x800u;

    MaxTextureSizes[ITexture::ETT_CUBE_MAP][0] = 0x80u;
    MaxTextureSizes[ITexture::ETT_CUBE_MAP][1] = 0x80u;
    MaxTextureSizes[ITexture::ETT_CUBE_MAP][2] = 0x6u;

    MaxTextureSizes[ITexture::ETT_CUBE_MAP_ARRAY][0] = 0x80u;
    MaxTextureSizes[ITexture::ETT_CUBE_MAP_ARRAY][1] = 0x80u;
    MaxTextureSizes[ITexture::ETT_CUBE_MAP_ARRAY][2] = 0x800u*6;

    MaxTextureSizes[ITexture::ETT_TEXTURE_BUFFER][0] = 0x80u;
    MaxTextureSizes[ITexture::ETT_TEXTURE_BUFFER][1] = 0x1u;
    MaxTextureSizes[ITexture::ETT_TEXTURE_BUFFER][2] = 0x1u;


	// set ExposedData to 0
	memset(&ExposedData, 0, sizeof(ExposedData));
	for (u32 i=0; i<video::EVDF_COUNT; ++i)
		FeatureEnabled[i]=true;

	InitMaterial2D.AntiAliasing=video::EAAM_OFF;
	InitMaterial2D.ZWriteEnable=false;
	InitMaterial2D.ZBuffer=video::ECFN_NEVER;
	for (u32 i=0; i<video::MATERIAL_MAX_TEXTURES; ++i)
	{
		InitMaterial2D.TextureLayer[i].SamplingParams.MinFilter = video::ETFT_NEAREST_NEARESTMIP;
		InitMaterial2D.TextureLayer[i].SamplingParams.MaxFilter = video::ETFT_NEAREST_NO_MIP;
		InitMaterial2D.TextureLayer[i].SamplingParams.TextureWrapU = video::ETC_REPEAT;
		InitMaterial2D.TextureLayer[i].SamplingParams.TextureWrapV = video::ETC_REPEAT;
        InitMaterial2D.TextureLayer[i].SamplingParams.UseMipmaps = false;
	}
	OverrideMaterial2D=InitMaterial2D;
}


//! destructor
CNullDriver::~CNullDriver()
{
    if (boxLineMesh)
        boxLineMesh->drop();

	if (FileSystem)
		FileSystem->drop();

    removeAllFrameBuffers();
    removeAllRenderBuffers();
	deleteAllTextures();

	u32 i;
	for (i=0; i<SurfaceLoader.size(); ++i)
		SurfaceLoader[i]->drop();

	for (i=0; i<SurfaceWriter.size(); ++i)
		SurfaceWriter[i]->drop();

	// delete material renderers
	deleteMaterialRenders();
}

void CNullDriver::bufferCopy(IGPUBuffer* readBuffer, IGPUBuffer* writeBuffer, const size_t& readOffset, const size_t& writeOffset, const size_t& length)
{
    os::Printer::log("Copying Buffers Not supported by this Driver!\n",ELL_ERROR);
}


//! Adds an external surface loader to the engine.
void CNullDriver::addExternalImageLoader(IImageLoader* loader)
{
	if (!loader)
		return;

	loader->grab();
	SurfaceLoader.push_back(loader);
}


//! Adds an external surface writer to the engine.
void CNullDriver::addExternalImageWriter(IImageWriter* writer)
{
	if (!writer)
		return;

	writer->grab();
	SurfaceWriter.push_back(writer);
}


//! Retrieve the number of image loaders
u32 CNullDriver::getImageLoaderCount() const
{
	return SurfaceLoader.size();
}


//! Retrieve the given image loader
IImageLoader* CNullDriver::getImageLoader(u32 n)
{
	if (n < SurfaceLoader.size())
		return SurfaceLoader[n];
	return 0;
}


//! Retrieve the number of image writers
u32 CNullDriver::getImageWriterCount() const
{
	return SurfaceWriter.size();
}


//! Retrieve the given image writer
IImageWriter* CNullDriver::getImageWriter(u32 n)
{
	if (n < SurfaceWriter.size())
		return SurfaceWriter[n];
	return 0;
}


//! deletes all textures
void CNullDriver::deleteAllTextures()
{
	// we need to remove previously set textures which might otherwise be kept in the
	// last set material member. Could be optimized to reduce state changes.
	setMaterial(SMaterial());

	for (u32 i=0; i<Textures.size(); ++i)
		Textures[i].Surface->drop();

	Textures.clear();
}



//! applications must call this method before performing any rendering. returns false if failed.
bool CNullDriver::beginScene(bool backBuffer, bool zBuffer, SColor color,
		const SExposedVideoData& videoData, core::rect<s32>* sourceRect)
{
	core::clearFPUException();

	scene::CMeshSceneNodeInstanced::recullOrder = 0;

	PrimitivesDrawn = 0;
	return true;
}


//! applications must call this method after performing any rendering. returns false if failed.
bool CNullDriver::endScene()
{
	FPSCounter.registerFrame(os::Timer::getRealTime(), PrimitivesDrawn);

	return true;
}


//! Disable a feature of the driver.
void CNullDriver::disableFeature(E_VIDEO_DRIVER_FEATURE feature, bool flag)
{
	FeatureEnabled[feature]=!flag;
}


//! queries the features of the driver, returns true if feature is available
bool CNullDriver::queryFeature(E_VIDEO_DRIVER_FEATURE feature) const
{
	return false;
}


//! sets transformation
void CNullDriver::setTransform(const E_4X3_TRANSFORMATION_STATE& state, const core::matrix4x3& mat)
{
    if (state>E4X3TS_WORLD)
        return;


	const uint32_t commonBits = (0x1u<<E4X3TS_WORLD_VIEW)|(0x1u<<E4X3TS_WORLD_VIEW_INVERSE)|(0x1u<<E4X3TS_NORMAL_MATRIX)|(0x1u<<(E4X3TS_COUNT+EPTS_PROJ_VIEW_WORLD))|(0x1u<<(E4X3TS_COUNT+EPTS_PROJ_VIEW_WORLD_INVERSE));
    uint32_t modifiedBit;
    switch (state)
    {
        case E4X3TS_VIEW:
            modifiedBit = (0x1u<<E4X3TS_VIEW)|(0x1u<<E4X3TS_VIEW_INVERSE)|(0x1u<<(E4X3TS_COUNT+EPTS_PROJ_VIEW))|(0x1u<<(E4X3TS_COUNT+EPTS_PROJ_VIEW_INVERSE))|commonBits;
            break;
        case E4X3TS_WORLD:
            modifiedBit = (0x1u<<E4X3TS_WORLD)|(0x1u<<E4X3TS_WORLD_INVERSE)|commonBits;
            break;
    }

    //if all bits marked as modified and matrices dont change
    if ((matrixModifiedBits&modifiedBit)==modifiedBit)
        TransformationMatrices[state] = mat;
    else
    {
        if (TransformationMatrices[state]==mat)
            return;
        matrixModifiedBits |= modifiedBit;
        TransformationMatrices[state] = mat;
    }
}

//! sets transformation
void CNullDriver::setTransform(const E_PROJECTION_TRANSFORMATION_STATE& state, const core::matrix4& mat)
{
    if (state>EPTS_PROJ)
        return;


	const uint32_t modifiedBit = ((0x1u<<EPTS_PROJ)|(0x1u<<EPTS_PROJ_VIEW)|(0x1u<<EPTS_PROJ_VIEW_WORLD)|(0x1u<<EPTS_PROJ_INVERSE)|(0x1u<<EPTS_PROJ_VIEW_INVERSE)|(0x1u<<EPTS_PROJ_VIEW_WORLD_INVERSE))<<E4X3TS_COUNT;

    //if all bits marked as modified and matrices dont change
    if ((matrixModifiedBits&modifiedBit)==modifiedBit)
        ProjectionMatrices[state] = mat;
    else
    {
        if (ProjectionMatrices[state]==mat)
            return;
        matrixModifiedBits |= modifiedBit;
        ProjectionMatrices[state] = mat;
    }
}


//! Returns the transformation set by setTransform
const core::matrix4x3& CNullDriver::getTransform(const E_4X3_TRANSFORMATION_STATE& state)
{
    const uint32_t stateBit = 0x1u<<state;

	if (matrixModifiedBits&stateBit)
    {
        switch (state)
        {
            case E4X3TS_WORLD:
            case E4X3TS_VIEW:
                break;
            case E4X3TS_WORLD_VIEW:
                TransformationMatrices[E4X3TS_WORLD_VIEW] = concatenateBFollowedByA(TransformationMatrices[E4X3TS_VIEW],TransformationMatrices[E4X3TS_WORLD]);
                break;
            case E4X3TS_VIEW_INVERSE:
                TransformationMatrices[E4X3TS_VIEW].getInverse(TransformationMatrices[E4X3TS_VIEW_INVERSE]);
                break;
            case E4X3TS_WORLD_INVERSE:
                TransformationMatrices[E4X3TS_WORLD].getInverse(TransformationMatrices[E4X3TS_WORLD_INVERSE]);
                break;
            case E4X3TS_WORLD_VIEW_INVERSE:
                if (matrixModifiedBits&(0x1u<<E4X3TS_WORLD_VIEW))
                {
                    TransformationMatrices[E4X3TS_WORLD_VIEW] = concatenateBFollowedByA(TransformationMatrices[E4X3TS_VIEW],TransformationMatrices[E4X3TS_WORLD]);
                    matrixModifiedBits &= ~(0x1u<<E4X3TS_WORLD_VIEW);
                }

                TransformationMatrices[E4X3TS_WORLD_VIEW].getInverse(TransformationMatrices[E4X3TS_WORLD_VIEW_INVERSE]);
                break;
            case E4X3TS_NORMAL_MATRIX:
                if (matrixModifiedBits&(0x1u<<E4X3TS_WORLD_VIEW_INVERSE))
                {
                    if (matrixModifiedBits&(0x1u<<E4X3TS_WORLD_VIEW))
                    {
                        TransformationMatrices[E4X3TS_WORLD_VIEW] = concatenateBFollowedByA(TransformationMatrices[E4X3TS_VIEW],TransformationMatrices[E4X3TS_WORLD]);
                        matrixModifiedBits &= ~(0x1u<<E4X3TS_WORLD_VIEW);
                    }

                    TransformationMatrices[E4X3TS_WORLD_VIEW].getInverse(TransformationMatrices[E4X3TS_WORLD_VIEW_INVERSE]);
                    matrixModifiedBits &= ~(0x1u<<E4X3TS_WORLD_VIEW_INVERSE);
                }

                TransformationMatrices[E4X3TS_NORMAL_MATRIX](0,0) = TransformationMatrices[E4X3TS_WORLD_VIEW_INVERSE](0,0);
                TransformationMatrices[E4X3TS_NORMAL_MATRIX](0,1) = TransformationMatrices[E4X3TS_WORLD_VIEW_INVERSE](1,0);
                TransformationMatrices[E4X3TS_NORMAL_MATRIX](0,2) = TransformationMatrices[E4X3TS_WORLD_VIEW_INVERSE](2,0);
                TransformationMatrices[E4X3TS_NORMAL_MATRIX](1,0) = TransformationMatrices[E4X3TS_WORLD_VIEW_INVERSE](0,1);
                TransformationMatrices[E4X3TS_NORMAL_MATRIX](1,1) = TransformationMatrices[E4X3TS_WORLD_VIEW_INVERSE](1,1);
                TransformationMatrices[E4X3TS_NORMAL_MATRIX](1,2) = TransformationMatrices[E4X3TS_WORLD_VIEW_INVERSE](2,1);
                TransformationMatrices[E4X3TS_NORMAL_MATRIX](2,0) = TransformationMatrices[E4X3TS_WORLD_VIEW_INVERSE](0,2);
                TransformationMatrices[E4X3TS_NORMAL_MATRIX](2,1) = TransformationMatrices[E4X3TS_WORLD_VIEW_INVERSE](1,2);
                TransformationMatrices[E4X3TS_NORMAL_MATRIX](2,2) = TransformationMatrices[E4X3TS_WORLD_VIEW_INVERSE](2,2);
                break;
        }

        matrixModifiedBits &= ~stateBit;
    }

    return TransformationMatrices[state];
}

//! Returns the transformation set by setTransform
const core::matrix4& CNullDriver::getTransform(const E_PROJECTION_TRANSFORMATION_STATE& state)
{
    const uint32_t stateBit = 0x1u<<(state+E4X3TS_COUNT);

	if (matrixModifiedBits&stateBit)
    {
        switch (state)
        {
            case EPTS_PROJ:
                break;
            case EPTS_PROJ_VIEW:
                ProjectionMatrices[EPTS_PROJ_VIEW] = concatenateBFollowedByA(ProjectionMatrices[EPTS_PROJ],TransformationMatrices[E4X3TS_VIEW]);
                break;
            case EPTS_PROJ_VIEW_WORLD:
                if (matrixModifiedBits&(0x1u<<E4X3TS_WORLD_VIEW))
                {
                    TransformationMatrices[E4X3TS_WORLD_VIEW] = concatenateBFollowedByA(TransformationMatrices[E4X3TS_VIEW],TransformationMatrices[E4X3TS_WORLD]);
                    ///TransformationMatrices[E4X3TS_WORLD_VIEW] = concatenatePreciselyBFollowedByA(TransformationMatrices[E4X3TS_VIEW],TransformationMatrices[E4X3TS_WORLD]);
                    matrixModifiedBits &= ~(0x1u<<E4X3TS_WORLD_VIEW);
                }
                ProjectionMatrices[EPTS_PROJ_VIEW_WORLD] = concatenateBFollowedByA(ProjectionMatrices[EPTS_PROJ],TransformationMatrices[E4X3TS_WORLD_VIEW]);
                break;
            case EPTS_PROJ_INVERSE:
                ProjectionMatrices[EPTS_PROJ].getInverse(ProjectionMatrices[EPTS_PROJ]);
                break;
            case EPTS_PROJ_VIEW_INVERSE:
                if (matrixModifiedBits&(0x1u<<(EPTS_PROJ_VIEW+E4X3TS_COUNT)))
                {
                    ProjectionMatrices[EPTS_PROJ_VIEW] = concatenateBFollowedByA(ProjectionMatrices[EPTS_PROJ],TransformationMatrices[E4X3TS_VIEW]);
                    matrixModifiedBits &= ~(0x1u<<(EPTS_PROJ_VIEW+E4X3TS_COUNT));
                }
                ProjectionMatrices[EPTS_PROJ_VIEW].getInverse(ProjectionMatrices[EPTS_PROJ_VIEW_INVERSE]);
                break;
            case EPTS_PROJ_VIEW_WORLD_INVERSE:
                if (matrixModifiedBits&(0x1u<<(EPTS_PROJ_VIEW_WORLD+E4X3TS_COUNT)))
                {
                    if (matrixModifiedBits&(0x1u<<E4X3TS_WORLD_VIEW))
                    {
                        TransformationMatrices[E4X3TS_WORLD_VIEW] = concatenateBFollowedByA(TransformationMatrices[E4X3TS_VIEW],TransformationMatrices[E4X3TS_WORLD]);
                        ///TransformationMatrices[E4X3TS_WORLD_VIEW] = concatenatePreciselyBFollowedByA(TransformationMatrices[E4X3TS_VIEW],TransformationMatrices[E4X3TS_WORLD]);
                        matrixModifiedBits &= ~(0x1u<<E4X3TS_WORLD_VIEW);
                    }
                    ProjectionMatrices[EPTS_PROJ_VIEW_WORLD] = concatenateBFollowedByA(ProjectionMatrices[EPTS_PROJ],TransformationMatrices[E4X3TS_WORLD_VIEW]);
                    matrixModifiedBits &= ~(0x1u<<(EPTS_PROJ_VIEW_WORLD+E4X3TS_COUNT));
                }
                ProjectionMatrices[EPTS_PROJ_VIEW_WORLD].getInverse(ProjectionMatrices[EPTS_PROJ_VIEW_WORLD_INVERSE]);
                break;
        }

        matrixModifiedBits &= ~stateBit;
    }
    return ProjectionMatrices[state];
}


//! sets a material
void CNullDriver::setMaterial(const SMaterial& material)
{
}


//! Removes a texture from the texture cache and deletes it, freeing lot of
//! memory.
void CNullDriver::removeTexture(ITexture* texture)
{
	if (!texture)
		return;

	for (u32 i=0; i<Textures.size(); ++i)
	{
		if (Textures[i].Surface == texture)
		{
			texture->drop();
			Textures.erase(i);
		}
	}
}

void CNullDriver::removeRenderBuffer(IRenderBuffer* renderbuf)
{
    int32_t ix = RenderBuffers.binary_search(renderbuf);
    if (ix<0)
        return;
    RenderBuffers.erase(ix);

    renderbuf->drop();
}

void CNullDriver::removeFrameBuffer(IFrameBuffer* framebuf)
{
    int32_t ix = FrameBuffers.binary_search(framebuf);
    if (ix<0)
        return;
    FrameBuffers.erase(ix);

    framebuf->drop();
}


//! Removes all texture from the texture cache and deletes them, freeing lot of
//! memory.
void CNullDriver::removeAllTextures()
{
	setMaterial ( SMaterial() );
	deleteAllTextures();
}

void CNullDriver::removeAllRenderBuffers()
{
	for (u32 i=0; i<RenderBuffers.size(); ++i)
		RenderBuffers[i]->drop();
}

void CNullDriver::removeAllFrameBuffers()
{
	for (u32 i=0; i<FrameBuffers.size(); ++i)
		FrameBuffers[i]->drop();
}


//! Returns a texture by index
ITexture* CNullDriver::getTextureByIndex(u32 i)
{
	if ( i < Textures.size() )
		return Textures[i].Surface;

	return 0;
}


//! Returns amount of textures currently loaded
u32 CNullDriver::getTextureCount() const
{
	return Textures.size();
}


//! Renames a texture
void CNullDriver::renameTexture(ITexture* texture, const io::path& newName)
{
	// we can do a const_cast here safely, the name of the ITexture interface
	// is just readonly to prevent the user changing the texture name without invoking
	// this method, because the textures will need resorting afterwards

	io::SNamedPath& name = const_cast<io::SNamedPath&>(texture->getName());
	name.setPath(newName);

	Textures.sort();
}


//! loads a Texture
ITexture* CNullDriver::getTexture(const io::path& filename)
{
	// Identify textures by their absolute filenames if possible.
	const io::path absolutePath = FileSystem->getAbsolutePath(filename);

	ITexture* texture = findTexture(absolutePath);
	if (texture)
		return texture;

	// Then try the raw filename, which might be in an Archive
	texture = findTexture(filename);
	if (texture)
		return texture;

	// Now try to open the file using the complete path.
	io::IReadFile* file = FileSystem->createAndOpenFile(absolutePath);

	if (!file)
	{
		// Try to open it using the raw filename.
		file = FileSystem->createAndOpenFile(filename);
	}

	if (file)
	{
		// Re-check name for actual archive names
		texture = findTexture(file->getFileName());
		if (texture)
		{
			file->drop();
			return texture;
		}

		texture = loadTextureFromFile(file);
		file->drop();

		if (texture)
		{
			addToTextureCache(texture);
			texture->drop(); // drop it because we created it, one grab too much
		}
		else
			os::Printer::log("Could not load texture", filename, ELL_ERROR);
		return texture;
	}
	else
	{
		os::Printer::log("Could not open file of texture", filename, ELL_WARNING);
		return 0;
	}
}


//! loads a Texture
ITexture* CNullDriver::getTexture(io::IReadFile* file)
{
	ITexture* texture = 0;

	if (file)
	{
		texture = findTexture(file->getFileName());

		if (texture)
			return texture;

		texture = loadTextureFromFile(file);

		if (texture)
		{
			addToTextureCache(texture);
			texture->drop(); // drop it because we created it, one grab too much
		}

		if (!texture)
			os::Printer::log("Could not load texture", file->getFileName(), ELL_WARNING);
	}

	return texture;
}


//! opens the file and loads it into the surface
video::ITexture* CNullDriver::loadTextureFromFile(io::IReadFile* file, const io::path& hashName )
{
	ITexture* texture = 0;
	IImage* image = createImageFromFile(file);

	if (image)
	{
		// create texture from surface
		texture = createDeviceDependentTexture(image, hashName.size() ? hashName : file->getFileName() );
		os::Printer::log("Loaded texture", file->getFileName());
		image->drop();
	}

	return texture;
}


//! adds a surface, not loaded or created by the Irrlicht Engine
void CNullDriver::addToTextureCache(video::ITexture* texture)
{
	if (texture)
	{
		SSurface s;
		s.Surface = texture;
		texture->grab();

		Textures.push_back(s);

		// the new texture is now at the end of the texture list. when searching for
		// the next new texture, the texture array will be sorted and the index of this texture
		// will be changed. to let the order be more consistent to the user, sort
		// the textures now already although this isn't necessary:

		Textures.sort();
	}
}


//! looks if the image is already loaded
video::ITexture* CNullDriver::findTexture(const io::path& filename)
{
	SSurface s;
	SDummyTexture dummy(filename);
	s.Surface = &dummy;

	s32 index = Textures.binary_search(s);
	if (index != -1)
		return Textures[index].Surface;

	return 0;
}


//! Creates a texture from a loaded IImage.
ITexture* CNullDriver::addTexture(const io::path& name, IImage* image, void* mipmapData)
{
	if ( 0 == name.size() || !image)
		return 0;

	ITexture* t = createDeviceDependentTexture(image, name, mipmapData);
	if (t)
	{
		addToTextureCache(t);
		t->drop();
	}
	return t;
}


//! creates a Texture
ITexture* CNullDriver::addTexture(const ITexture::E_TEXTURE_TYPE& type, const uint32_t* size, uint32_t mipMapLevels,
				  const io::path& name, ECOLOR_FORMAT format)
{
	if ( 0 == name.size () )
		return 0;

	ITexture* t = createDeviceDependentTexture(type,size,mipMapLevels,name,format);
	addToTextureCache(t);

	if (t)
		t->drop();

	return t;
}



//! returns a device dependent texture from a software surface (IImage)
//! THIS METHOD HAS TO BE OVERRIDDEN BY DERIVED DRIVERS WITH OWN TEXTURES
ITexture* CNullDriver::createDeviceDependentTexture(IImage* surface, const io::path& name, void* mipmapData)
{
	return new SDummyTexture(name);
}
ITexture* CNullDriver::createDeviceDependentTexture(const ITexture::E_TEXTURE_TYPE& type, const uint32_t* size, uint32_t mipmapLevels,
			const io::path& name, ECOLOR_FORMAT format)
{
	return new SDummyTexture(name);
}


//! sets a render target
bool CNullDriver::setRenderTarget(video::IFrameBuffer* texture, bool setNewViewport)
{
	return false;
}


//! sets a viewport
void CNullDriver::setViewPort(const core::rect<s32>& area)
{
}


//! gets the area of the current viewport
const core::rect<s32>& CNullDriver::getViewPort() const
{
	return ViewPort;
}



//! Draws a 3d line.
void CNullDriver::draw3DLine(const core::vector3df& start,
				const core::vector3df& end, SColor color)
{
}

//! Draws a 3d axis aligned box.
void CNullDriver::draw3DBox(const core::aabbox3d<f32>& box, SColor color)
{
	core::vector3df edges[8];
	box.getEdges(edges);
	uint32_t colors[8];
	for (u32 i=0; i<8; i++)
    {
        colors[8] = color.color;
    }
    u16 indices[24] = {5,1,1,3,3,7,7,5,0,2,2,6,6,4,4,0,1,0,3,2,7,6,5,4};

    if (!boxLineMesh)
    {
        video::IGPUBuffer* indexBuf = createGPUBuffer(sizeof(indices),indices);
        video::IGPUBuffer* vposBuf = createGPUBuffer(sizeof(edges),edges,true,true);
        video::IGPUBuffer* vcolBuf = createGPUBuffer(sizeof(colors),colors,true,true);
        scene::IGPUMeshDataFormatDesc* desc = createGPUMeshDataFormatDesc();
        if (!indexBuf||!vposBuf||!vcolBuf||!desc)
            return;

        scene::IGPUMeshBuffer* boxLineMesh =  new scene::IGPUMeshBuffer();
        boxLineMesh->setIndexCount(24);
        boxLineMesh->setIndexType(EIT_16BIT);
        boxLineMesh->setMeshDataAndFormat(desc);
        desc->drop();
        boxLineMesh->setPrimitiveType(scene::EPT_LINES);

        desc->mapVertexAttrBuffer(vposBuf,scene::EVAI_ATTR0,scene::ECPA_THREE,scene::ECT_FLOAT);
        vposBuf->drop();
        desc->mapVertexAttrBuffer(vcolBuf,scene::EVAI_ATTR2,scene::ECPA_REVERSED_OR_BGRA,scene::ECT_NORMALIZED_UNSIGNED_BYTE);
        vposBuf->drop();
        desc->mapIndexBuffer(indexBuf);
        indexBuf->drop();
    }
    else
    {
        const_cast<IGPUBuffer*>(boxLineMesh->getMeshDataAndFormat()->getMappedBuffer(scene::EVAI_ATTR0))->updateSubRange(0,sizeof(edges),edges);
        const_cast<IGPUBuffer*>(boxLineMesh->getMeshDataAndFormat()->getMappedBuffer(scene::EVAI_ATTR2))->updateSubRange(0,sizeof(colors),colors);
    }

    if (boxLineMesh)
        drawMeshBuffer(boxLineMesh,NULL);
}



//! draws an 2d image
void CNullDriver::draw2DImage(const video::ITexture* texture, const core::position2d<s32>& destPos)
{
	if (!texture)
		return;

	draw2DImage(texture,destPos, core::rect<s32>(core::position2d<s32>(0,0),
												core::dimension2di(*reinterpret_cast<const core::dimension2du*>(texture->getSize()))));
}



//! draws a set of 2d images, using a color and the alpha channel of the
//! texture if desired. The images are drawn beginning at pos and concatenated
//! in one line. All drawings are clipped against clipRect (if != 0).
//! The subtextures are defined by the array of sourceRects and are chosen
//! by the indices given.
void CNullDriver::draw2DImageBatch(const video::ITexture* texture,
				const core::position2d<s32>& pos,
				const core::array<core::rect<s32> >& sourceRects,
				const core::array<s32>& indices,
				s32 kerningWidth,
				const core::rect<s32>* clipRect, SColor color,
				bool useAlphaChannelOfTexture)
{
	core::position2d<s32> target(pos);

	for (u32 i=0; i<indices.size(); ++i)
	{
		draw2DImage(texture, target, sourceRects[indices[i]],
				clipRect, color, useAlphaChannelOfTexture);
		target.X += sourceRects[indices[i]].getWidth();
		target.X += kerningWidth;
	}
}

//! draws a set of 2d images, using a color and the alpha channel of the
//! texture if desired.
void CNullDriver::draw2DImageBatch(const video::ITexture* texture,
				const core::array<core::position2d<s32> >& positions,
				const core::array<core::rect<s32> >& sourceRects,
				const core::rect<s32>* clipRect,
				SColor color,
				bool useAlphaChannelOfTexture)
{
	const irr::u32 drawCount = core::min_<u32>(positions.size(), sourceRects.size());

	for (u32 i=0; i<drawCount; ++i)
	{
		draw2DImage(texture, positions[i], sourceRects[i],
				clipRect, color, useAlphaChannelOfTexture);
	}
}


//! Draws a part of the texture into the rectangle.
void CNullDriver::draw2DImage(const video::ITexture* texture, const core::rect<s32>& destRect,
	const core::rect<s32>& sourceRect, const core::rect<s32>* clipRect,
	const video::SColor* const colors, bool useAlphaChannelOfTexture)
{
	if (destRect.isValid())
		draw2DImage(texture, core::position2d<s32>(destRect.UpperLeftCorner),
				sourceRect, clipRect, colors?colors[0]:video::SColor(0xffffffff),
				useAlphaChannelOfTexture);
}


//! Draws a 2d image, using a color (if color is other then Color(255,255,255,255)) and the alpha channel of the texture if wanted.
void CNullDriver::draw2DImage(const video::ITexture* texture, const core::position2d<s32>& destPos,
				const core::rect<s32>& sourceRect,
				const core::rect<s32>* clipRect, SColor color,
				bool useAlphaChannelOfTexture)
{
}


//! Draws the outline of a 2d rectangle
void CNullDriver::draw2DRectangleOutline(const core::recti& pos, SColor color)
{
	draw2DLine(pos.UpperLeftCorner, core::position2di(pos.LowerRightCorner.X, pos.UpperLeftCorner.Y), color);
	draw2DLine(core::position2di(pos.LowerRightCorner.X, pos.UpperLeftCorner.Y), pos.LowerRightCorner, color);
	draw2DLine(pos.LowerRightCorner, core::position2di(pos.UpperLeftCorner.X, pos.LowerRightCorner.Y), color);
	draw2DLine(core::position2di(pos.UpperLeftCorner.X, pos.LowerRightCorner.Y), pos.UpperLeftCorner, color);
}


//! Draw a 2d rectangle
void CNullDriver::draw2DRectangle(SColor color, const core::rect<s32>& pos, const core::rect<s32>* clip)
{
	draw2DRectangle(pos, color, color, color, color, clip);
}



//! Draws a 2d rectangle with a gradient.
void CNullDriver::draw2DRectangle(const core::rect<s32>& pos,
	SColor colorLeftUp, SColor colorRightUp, SColor colorLeftDown, SColor colorRightDown,
	const core::rect<s32>* clip)
{
}



//! Draws a 2d line.
void CNullDriver::draw2DLine(const core::position2d<s32>& start,
				const core::position2d<s32>& end, SColor color)
{
}


//! returns color format
ECOLOR_FORMAT CNullDriver::getColorFormat() const
{
	return ECF_R5G6B5;
}


//! returns screen size
const core::dimension2d<u32>& CNullDriver::getScreenSize() const
{
	return ScreenSize;
}


//! returns the current render target size,
//! or the screen size if render targets are not implemented
const core::dimension2d<u32>& CNullDriver::getCurrentRenderTargetSize() const
{
	return ScreenSize;
}


// returns current frames per second value
s32 CNullDriver::getFPS() const
{
	return FPSCounter.getFPS();
}



//! returns amount of primitives (mostly triangles) were drawn in the last frame.
//! very useful method for statistics.
u32 CNullDriver::getPrimitiveCountDrawn( u32 param ) const
{
	return (0 == param) ? FPSCounter.getPrimitive() : (1 == param) ? FPSCounter.getPrimitiveAverage() : FPSCounter.getPrimitiveTotal();
}



//! \return Returns the name of the video driver. Example: In case of the DIRECT3D8
//! driver, it would return "Direct3D8".

const wchar_t* CNullDriver::getName() const
{
	return L"Irrlicht NullDevice";
}




//! deletes all dynamic lights there are
void CNullDriver::deleteAllDynamicLights()
{
	Lights.set_used(0);
}


//! adds a dynamic light
s32 CNullDriver::addDynamicLight(const SLight& light)
{
	Lights.push_back(light);
	return Lights.size() - 1;
}

//! Turns a dynamic light on or off
//! \param lightIndex: the index returned by addDynamicLight
//! \param turnOn: true to turn the light on, false to turn it off
void CNullDriver::turnLightOn(s32 lightIndex, bool turnOn)
{
	// Do nothing
}


//! returns the maximal amount of dynamic lights the device can handle
u32 CNullDriver::getMaximalDynamicLightAmount() const
{
	return 0;
}


//! Returns current amount of dynamic lights set
//! \return Current amount of dynamic lights set
u32 CNullDriver::getDynamicLightCount() const
{
	return Lights.size();
}


//! Returns light data which was previously set by IVideoDriver::addDynamicLight().
//! \param idx: Zero based index of the light. Must be greater than 0 and smaller
//! than IVideoDriver()::getDynamicLightCount.
//! \return Light data.
const SLight& CNullDriver::getDynamicLight(u32 idx) const
{
	if ( idx < Lights.size() )
		return Lights[idx];
	else
		return *((SLight*)0);
}



//! Creates a normal map from a height map texture.
//! \param amplitude: Constant value by which the height information is multiplied.
void CNullDriver::makeNormalMapTexture(video::ITexture* texture, f32 amplitude) const
{
    /*
	if (!texture)
		return;

	if (texture->getColorFormat() != ECF_A1R5G5B5 &&
		texture->getColorFormat() != ECF_A8R8G8B8 )
	{
		os::Printer::log("Error: Unsupported texture color format for making normal map.", ELL_ERROR);
		return;
	}

	core::dimension2d<u32> dim = texture->getSize();
	amplitude = amplitude / 255.0f;
	f32 vh = dim.Height / (f32)dim.Width;
	f32 hh = dim.Width / (f32)dim.Height;

	if (texture->getColorFormat() == ECF_A8R8G8B8)
	{
		// ECF_A8R8G8B8 version

		s32 *p = (s32*)texture->lock();

		if (!p)
		{
			os::Printer::log("Could not lock texture for making normal map.", ELL_ERROR);
			return;
		}

		// copy texture

		u32 pitch = texture->getPitch() / 4;

		s32* in = new s32[dim.Height * pitch];
		memcpy(in, p, dim.Height * pitch * 4);

		for (s32 x=0; x < s32(pitch); ++x)
			for (s32 y=0; y < s32(dim.Height); ++y)
			{
				// TODO: this could be optimized really a lot

				core::vector3df h1((x-1)*hh, nml32(x-1, y, pitch, dim.Height, in)*amplitude, y*vh);
				core::vector3df h2((x+1)*hh, nml32(x+1, y, pitch, dim.Height, in)*amplitude, y*vh);
				//core::vector3df v1(x*hh, nml32(x, y-1, pitch, dim.Height, in)*amplitude, (y-1)*vh);
				//core::vector3df v2(x*hh, nml32(x, y+1, pitch, dim.Height, in)*amplitude, (y+1)*vh);
				core::vector3df v1(x*hh, nml32(x, y+1, pitch, dim.Height, in)*amplitude, (y-1)*vh);
				core::vector3df v2(x*hh, nml32(x, y-1, pitch, dim.Height, in)*amplitude, (y+1)*vh);

				core::vector3df v = v1-v2;
				core::vector3df h = h1-h2;

				core::vector3df n = v.crossProduct(h);
				n.normalize();
				n *= 0.5f;
				n += core::vector3df(0.5f,0.5f,0.5f); // now between 0 and 1
				n *= 255.0f;

				s32 height = (s32)nml32(x, y, pitch, dim.Height, in);
				p[y*pitch + x] = video::SColor(
					height, // store height in alpha
					(s32)n.X, (s32)n.Z, (s32)n.Y).color;
			}

		delete [] in;
		texture->unlock();
	}
	else
	{
		// ECF_A1R5G5B5 version

		s16 *p = (s16*)texture->lock();

		if (!p)
		{
			os::Printer::log("Could not lock texture for making normal map.", ELL_ERROR);
			return;
		}

		u32 pitch = texture->getPitch() / 2;

		// copy texture

		s16* in = new s16[dim.Height * pitch];
		memcpy(in, p, dim.Height * pitch * 2);

		for (s32 x=0; x < s32(pitch); ++x)
			for (s32 y=0; y < s32(dim.Height); ++y)
			{
				// TODO: this could be optimized really a lot

				core::vector3df h1((x-1)*hh, nml16(x-1, y, pitch, dim.Height, in)*amplitude, y*vh);
				core::vector3df h2((x+1)*hh, nml16(x+1, y, pitch, dim.Height, in)*amplitude, y*vh);
				core::vector3df v1(x*hh, nml16(x, y-1, pitch, dim.Height, in)*amplitude, (y-1)*vh);
				core::vector3df v2(x*hh, nml16(x, y+1, pitch, dim.Height, in)*amplitude, (y+1)*vh);

				core::vector3df v = v1-v2;
				core::vector3df h = h1-h2;

				core::vector3df n = v.crossProduct(h);
				n.normalize();
				n *= 0.5f;
				n += core::vector3df(0.5f,0.5f,0.5f); // now between 0 and 1
				n *= 255.0f;

				p[y*pitch + x] = video::RGBA16((u32)n.X, (u32)n.Z, (u32)n.Y);
			}

		delete [] in;
		texture->unlock();
	}

	texture->regenerateMipMapLevels();*/

		os::Printer::log("DevSH thinks you're a retard for using the CPU for processing pixels.", ELL_ERROR);
		return;
}


//! Returns the maximum amount of primitives (mostly vertices) which
//! the device is able to render with one drawIndexedTriangleList
//! call.
u32 CNullDriver::getMaximalIndicesCount() const
{
	return 0xFFFFFFFF;
}


//! Enables or disables a texture creation flag.
void CNullDriver::setTextureCreationFlag(E_TEXTURE_CREATION_FLAG flag, bool enabled)
{
	if (enabled && ((flag == ETCF_ALWAYS_16_BIT) || (flag == ETCF_ALWAYS_32_BIT)
		|| (flag == ETCF_OPTIMIZED_FOR_QUALITY) || (flag == ETCF_OPTIMIZED_FOR_SPEED)))
	{
		// disable other formats
		setTextureCreationFlag(ETCF_ALWAYS_16_BIT, false);
		setTextureCreationFlag(ETCF_ALWAYS_32_BIT, false);
		setTextureCreationFlag(ETCF_OPTIMIZED_FOR_QUALITY, false);
		setTextureCreationFlag(ETCF_OPTIMIZED_FOR_SPEED, false);
	}

	// set flag
	TextureCreationFlags = (TextureCreationFlags & (~flag)) |
		((((u32)!enabled)-1) & flag);
}


//! Returns if a texture creation flag is enabled or disabled.
bool CNullDriver::getTextureCreationFlag(E_TEXTURE_CREATION_FLAG flag) const
{
	return (TextureCreationFlags & flag)!=0;
}


//! Creates a software image from a file.
IImage* CNullDriver::createImageFromFile(const io::path& filename)
{
	if (!filename.size())
		return 0;

	IImage* image = 0;
	io::IReadFile* file = FileSystem->createAndOpenFile(filename);

	if (file)
	{
		image = createImageFromFile(file);
		file->drop();
	}
//	else
//		os::Printer::log("Could not open file of image", filename, ELL_WARNING);	// sodan

	return image;
}


//! Creates a software image from a file.
IImage* CNullDriver::createImageFromFile(io::IReadFile* file)
{
	if (!file)
		return 0;

	IImage* image = 0;

	s32 i;

	// try to load file based on file extension
	for (i=SurfaceLoader.size()-1; i>=0; --i)
	{
		if (SurfaceLoader[i]->isALoadableFileExtension(file->getFileName()))
		{
			// reset file position which might have changed due to previous loadImage calls
			file->seek(0);
			image = SurfaceLoader[i]->loadImage(file);
			if (image)
				return image;
		}
	}

	// try to load file based on what is in it
	for (i=SurfaceLoader.size()-1; i>=0; --i)
	{
		// dito
		file->seek(0);
		if (SurfaceLoader[i]->isALoadableFileFormat(file))
		{
			file->seek(0);
			image = SurfaceLoader[i]->loadImage(file);
			if (image)
				return image;
		}
	}

	return 0; // failed to load
}


//! Writes the provided image to disk file
bool CNullDriver::writeImageToFile(IImage* image, const io::path& filename,u32 param)
{
	io::IWriteFile* file = FileSystem->createAndWriteFile(filename);
	if(!file)
		return false;

	bool result = writeImageToFile(image, file, param);
	file->drop();

	return result;
}

//! Writes the provided image to a file.
bool CNullDriver::writeImageToFile(IImage* image, io::IWriteFile * file, u32 param)
{
	if(!file)
		return false;

	for (s32 i=SurfaceWriter.size()-1; i>=0; --i)
	{
		if (SurfaceWriter[i]->isAWriteableFileExtension(file->getFileName()))
		{
			bool written = SurfaceWriter[i]->writeImage(file, image, param);
			if (written)
				return true;
		}
	}
	return false; // failed to write
}


//! Creates a software image from a byte array.
IImage* CNullDriver::createImageFromData(ECOLOR_FORMAT format,
					const core::dimension2d<u32>& size,
					void *data, bool ownForeignMemory,
					bool deleteMemory)
{
	return new CImage(format, size, data, ownForeignMemory, deleteMemory);
}


//! Creates an empty software image.
IImage* CNullDriver::createImage(ECOLOR_FORMAT format, const core::dimension2d<u32>& size)
{
    return new CImage(format, size);
}


//! Creates a software image from another image.
IImage* CNullDriver::createImage(ECOLOR_FORMAT format, IImage *imageToCopy)
{
	os::Printer::log("Deprecated method, please create an empty image instead and use copyTo().", ELL_WARNING);

	CImage* tmp = new CImage(format, imageToCopy->getDimension());
	imageToCopy->copyTo(tmp);
	return tmp;
}


//! Creates a software image from part of another image.
IImage* CNullDriver::createImage(IImage* imageToCopy, const core::position2d<s32>& pos, const core::dimension2d<u32>& size)
{
	os::Printer::log("Deprecated method, please create an empty image instead and use copyTo().", ELL_WARNING);
	CImage* tmp = new CImage(imageToCopy->getColorFormat(), imageToCopy->getDimension());
	imageToCopy->copyTo(tmp, core::position2di(0,0), core::recti(pos,size));
	return tmp;
}



void CNullDriver::drawMeshBuffer(scene::IGPUMeshBuffer* mb, IOcclusionQuery* query)
{
	if (!mb)
		return;

    uint32_t increment = mb->getInstanceCount();
    switch (mb->getPrimitiveType())
    {
        case scene::EPT_POINTS:
            increment *= mb->getIndexCount();
            break;
        case scene::EPT_LINE_STRIP:
            increment *= mb->getIndexCount()-1;
            break;
        case scene::EPT_LINE_LOOP:
            increment *= mb->getIndexCount();
            break;
        case scene::EPT_LINES:
            increment *= mb->getIndexCount()/2;
            break;
        case scene::EPT_TRIANGLE_STRIP:
            increment *= mb->getIndexCount()-2;
            break;
        case scene::EPT_TRIANGLE_FAN:
            increment *= mb->getIndexCount()-2;
            break;
        case scene::EPT_TRIANGLES:
            increment *= mb->getIndexCount()/3;
            break;
    }
    PrimitivesDrawn += increment;
}


//! Indirect Draw
void CNullDriver::drawArraysIndirect(scene::IGPUMeshDataFormatDesc* vao, scene::E_PRIMITIVE_TYPE& mode, IGPUBuffer* indirectDrawBuff, const size_t& offset, const size_t& count, const size_t& stride, IOcclusionQuery* query)
{
}

void CNullDriver::drawIndexedIndirect(scene::IGPUMeshDataFormatDesc* vao, scene::E_PRIMITIVE_TYPE& mode, const E_INDEX_TYPE& type, IGPUBuffer* indirectDrawBuff, const size_t& offset, const size_t& count, const size_t& stride, IOcclusionQuery* query)
{
}




IOcclusionQuery* CNullDriver::createOcclusionQuery(const E_OCCLUSION_QUERY_TYPE& heuristic)
{
    return NULL;
}

IQueryObject* CNullDriver::createPrimitivesGeneratedQuery()
{
    return NULL;
}
IQueryObject* CNullDriver::createXFormFeedbackPrimitiveQuery()
{
    return NULL;
}
IQueryObject* CNullDriver::createElapsedTimeQuery()
{
    return NULL;
}
IGPUTimestampQuery* CNullDriver::createTimestampQuery()
{
    return NULL;
}


void CNullDriver::beginQuery(IQueryObject* query)
{
    if (!query)
        return; //error

    if (currentQuery[query->getQueryObjectType()][0])
        return; //error

    query->grab();
    currentQuery[query->getQueryObjectType()][0] = query;
}
void CNullDriver::endQuery(IQueryObject* query)
{
    if (!query)
        return; //error
    if (currentQuery[query->getQueryObjectType()][0]!=query)
        return; //error

    if (currentQuery[query->getQueryObjectType()][0])
        currentQuery[query->getQueryObjectType()][0]->drop();
    currentQuery[query->getQueryObjectType()][0] = NULL;
}

void CNullDriver::beginQuery(IQueryObject* query, const size_t& index)
{
    if (index>=_IRR_XFORM_FEEDBACK_MAX_STREAMS_)
        return; //error

    if (!query||(query->getQueryObjectType()!=EQOT_PRIMITIVES_GENERATED&&query->getQueryObjectType()!=EQOT_XFORM_FEEDBACK_PRIMITIVES_WRITTEN))
        return; //error

    if (currentQuery[query->getQueryObjectType()][index])
        return; //error

    query->grab();
    currentQuery[query->getQueryObjectType()][index] = query;
}
void CNullDriver::endQuery(IQueryObject* query, const size_t& index)
{
    if (index>=_IRR_XFORM_FEEDBACK_MAX_STREAMS_)
        return; //error

    if (!query||(query->getQueryObjectType()!=EQOT_PRIMITIVES_GENERATED&&query->getQueryObjectType()!=EQOT_XFORM_FEEDBACK_PRIMITIVES_WRITTEN))
        return; //error
    if (currentQuery[query->getQueryObjectType()][index]!=query)
        return; //error

    if (currentQuery[query->getQueryObjectType()][index])
        currentQuery[query->getQueryObjectType()][index]->drop();
    currentQuery[query->getQueryObjectType()][index] = NULL;
}


//! Only used by the internal engine. Used to notify the driver that
//! the window was resized.
void CNullDriver::OnResize(const core::dimension2d<u32>& size)
{
	if (ViewPort.getWidth() == (s32)ScreenSize.Width &&
		ViewPort.getHeight() == (s32)ScreenSize.Height)
		ViewPort = core::rect<s32>(core::position2d<s32>(0,0),
									core::dimension2di(size));

	ScreenSize = size;
}


// adds a material renderer and drops it afterwards. To be used for internal creation
s32 CNullDriver::addAndDropMaterialRenderer(IMaterialRenderer* m)
{
	s32 i = addMaterialRenderer(m);

	if (m)
		m->drop();

	return i;
}


//! Adds a new material renderer to the video device.
s32 CNullDriver::addMaterialRenderer(IMaterialRenderer* renderer, const char* name)
{
	if (!renderer)
		return -1;

	SMaterialRenderer r;
	r.Renderer = renderer;
	r.Name = name;

	if (name == 0 && (MaterialRenderers.size() < (sizeof(sBuiltInMaterialTypeNames) / sizeof(char*))-1 ))
	{
		// set name of built in renderer so that we don't have to implement name
		// setting in all available renderers.
		r.Name = sBuiltInMaterialTypeNames[MaterialRenderers.size()];
	}

	MaterialRenderers.push_back(r);
	renderer->grab();

	return MaterialRenderers.size()-1;
}


//! Sets the name of a material renderer.
void CNullDriver::setMaterialRendererName(s32 idx, const char* name)
{
	if (idx < s32(sizeof(sBuiltInMaterialTypeNames) / sizeof(char*))-1 ||
		idx >= (s32)MaterialRenderers.size())
		return;

	MaterialRenderers[idx].Name = name;
}


//! Returns driver and operating system specific data about the IVideoDriver.
const SExposedVideoData& CNullDriver::getExposedVideoData()
{
	return ExposedData;
}


//! Returns type of video driver
E_DRIVER_TYPE CNullDriver::getDriverType() const
{
	return EDT_NULL;
}


//! deletes all material renderers
void CNullDriver::deleteMaterialRenders()
{
	// delete material renderers
	for (u32 i=0; i<MaterialRenderers.size(); ++i)
		if (MaterialRenderers[i].Renderer)
			MaterialRenderers[i].Renderer->drop();

	MaterialRenderers.clear();
}


//! Returns pointer to material renderer or null
IMaterialRenderer* CNullDriver::getMaterialRenderer(u32 idx)
{
	if ( idx < MaterialRenderers.size() )
		return MaterialRenderers[idx].Renderer;
	else
		return 0;
}


//! Returns amount of currently available material renderers.
u32 CNullDriver::getMaterialRendererCount() const
{
	return MaterialRenderers.size();
}


//! Returns name of the material renderer
const char* CNullDriver::getMaterialRendererName(u32 idx) const
{
	if ( idx < MaterialRenderers.size() )
		return MaterialRenderers[idx].Name.c_str();

	return 0;
}


//! Returns pointer to the IGPUProgrammingServices interface.
IGPUProgrammingServices* CNullDriver::getGPUProgrammingServices()
{
	return this;
}

s32 CNullDriver::addHighLevelShaderMaterial(
    const c8* vertexShaderProgram,
    const c8* controlShaderProgram,
    const c8* evaluationShaderProgram,
    const c8* geometryShaderProgram,
    const c8* pixelShaderProgram,
    u32 patchVertices,
    E_MATERIAL_TYPE baseMaterial,
    IShaderConstantSetCallBack* callback,
    const char** xformFeedbackOutputs,
    const uint32_t& xformFeedbackOutputCount,
    s32 userData,
    const c8* vertexShaderEntryPointName,
    const c8* controlShaderEntryPointName,
    const c8* evaluationShaderEntryPointName,
    const c8* geometryShaderEntryPointName,
    const c8* pixelShaderEntryPointName)
{
	os::Printer::log("High level shader materials not available (yet) in this driver, sorry");
	return -1;
}

bool CNullDriver::replaceHighLevelShaderMaterial(const s32 &materialIDToReplace,
    const c8* vertexShaderProgram,
    const c8* controlShaderProgram,
    const c8* evaluationShaderProgram,
    const c8* geometryShaderProgram,
    const c8* pixelShaderProgram,
    u32 patchVertices,
    E_MATERIAL_TYPE baseMaterial,
    IShaderConstantSetCallBack* callback,
    const char** xformFeedbackOutputs,
    const uint32_t& xformFeedbackOutputCount,
    s32 userData,
    const c8* vertexShaderEntryPointName,
    const c8* controlShaderEntryPointName,
    const c8* evaluationShaderEntryPointName,
    const c8* geometryShaderEntryPointName,
    const c8* pixelShaderEntryPointName)
{
    if (MaterialRenderers.size()<materialIDToReplace)
        return false;

    s32 nr = addHighLevelShaderMaterial(vertexShaderProgram,controlShaderProgram,evaluationShaderProgram,geometryShaderProgram,pixelShaderProgram,
                                        3,baseMaterial,callback,xformFeedbackOutputs,xformFeedbackOutputCount,userData,
                                        vertexShaderEntryPointName,controlShaderEntryPointName,evaluationShaderEntryPointName,geometryShaderEntryPointName,pixelShaderEntryPointName);
    if (nr==-1)
        return false;

	MaterialRenderers[materialIDToReplace].Renderer->drop();
	MaterialRenderers[materialIDToReplace] = MaterialRenderers[MaterialRenderers.size()-1];

	MaterialRenderers.set_used(MaterialRenderers.size()-1);

    return true;
}

s32 CNullDriver::addHighLevelShaderMaterialFromFiles(
    const io::path& vertexShaderProgramFileName,
    const io::path& controlShaderProgramFileName,
    const io::path& evaluationShaderProgramFileName,
    const io::path& geometryShaderProgramFileName,
    const io::path& pixelShaderProgramFileName,
    u32 patchVertices,
    E_MATERIAL_TYPE baseMaterial,
    IShaderConstantSetCallBack* callback,
    const char** xformFeedbackOutputs,
    const uint32_t& xformFeedbackOutputCount,
    s32 userData,
    const c8* vertexShaderEntryPointName,
    const c8* controlShaderEntryPointName,
    const c8* evaluationShaderEntryPointName,
    const c8* geometryShaderEntryPointName,
    const c8* pixelShaderEntryPointName)
{
	io::IReadFile* vsfile = 0;
	io::IReadFile* gsfile = 0;
	io::IReadFile* ctsfile = 0;
	io::IReadFile* etsfile = 0;
	io::IReadFile* psfile = 0;

	if (vertexShaderProgramFileName.size() )
	{
		vsfile = FileSystem->createAndOpenFile(vertexShaderProgramFileName);
		if (!vsfile)
		{
			os::Printer::log("Could not open vertex shader program file",
				vertexShaderProgramFileName, ELL_WARNING);
		}
	}

	if (controlShaderProgramFileName.size() )
	{
		ctsfile = FileSystem->createAndOpenFile(controlShaderProgramFileName);
		if (!ctsfile)
		{
			os::Printer::log("Could not open control shader program file",
				controlShaderProgramFileName, ELL_WARNING);
		}
	}

	if (evaluationShaderProgramFileName.size() )
	{
		etsfile = FileSystem->createAndOpenFile(evaluationShaderProgramFileName);
		if (!etsfile)
		{
			os::Printer::log("Could not open evaluation shader program file",
				evaluationShaderProgramFileName, ELL_WARNING);
		}
	}

	if (geometryShaderProgramFileName.size() )
	{
		gsfile = FileSystem->createAndOpenFile(geometryShaderProgramFileName);
		if (!gsfile)
		{
			os::Printer::log("Could not open geometry shader program file",
				geometryShaderProgramFileName, ELL_WARNING);
		}
	}

	if (pixelShaderProgramFileName.size() )
	{
		psfile = FileSystem->createAndOpenFile(pixelShaderProgramFileName);
		if (!psfile)
		{
			os::Printer::log("Could not open pixel shader program file",
				pixelShaderProgramFileName, ELL_WARNING);
		}
	}

	s32 result = addHighLevelShaderMaterialFromFiles(
		vsfile, ctsfile, etsfile, gsfile, psfile,
		patchVertices, baseMaterial, callback,
		xformFeedbackOutputs,xformFeedbackOutputCount, userData,
		vertexShaderEntryPointName, controlShaderEntryPointName,
		evaluationShaderEntryPointName, geometryShaderEntryPointName,
		pixelShaderEntryPointName);

	if (psfile)
		psfile->drop();

	if (ctsfile)
		ctsfile->drop();

	if (etsfile)
		etsfile->drop();

	if (gsfile)
		gsfile->drop();

	if (vsfile)
		vsfile->drop();

	return result;
}

s32 CNullDriver::addHighLevelShaderMaterialFromFiles(
    io::IReadFile* vertexShaderProgram,
    io::IReadFile* controlShaderProgram,
    io::IReadFile* evaluationShaderProgram,
    io::IReadFile* geometryShaderProgram,
    io::IReadFile* pixelShaderProgram,
    u32 patchVertices,
    E_MATERIAL_TYPE baseMaterial,
    IShaderConstantSetCallBack* callback,
    const char** xformFeedbackOutputs,
    const uint32_t& xformFeedbackOutputCount,
    s32 userData,
    const c8* vertexShaderEntryPointName,
    const c8* controlShaderEntryPointName,
    const c8* evaluationShaderEntryPointName,
    const c8* geometryShaderEntryPointName,
    const c8* pixelShaderEntryPointName)
{
	c8* vs = 0;
	c8* cts = 0;
	c8* ets = 0;
	c8* gs = 0;
	c8* ps = 0;

	if (vertexShaderProgram)
	{
		const long size = vertexShaderProgram->getSize();
		if (size)
		{
			vs = new c8[size+1];
			vertexShaderProgram->read(vs, size);
			vs[size] = 0;
		}
	}

	if (pixelShaderProgram)
	{
		const long size = pixelShaderProgram->getSize();
		if (size)
		{
			// if both handles are the same we must reset the file
			if (pixelShaderProgram==vertexShaderProgram)
				pixelShaderProgram->seek(0);
			ps = new c8[size+1];
			pixelShaderProgram->read(ps, size);
			ps[size] = 0;
		}
	}

	if (geometryShaderProgram)
	{
		const long size = geometryShaderProgram->getSize();
		if (size)
		{
			// if both handles are the same we must reset the file
			if ((geometryShaderProgram==vertexShaderProgram) ||
					(geometryShaderProgram==pixelShaderProgram))
				geometryShaderProgram->seek(0);
			gs = new c8[size+1];
			geometryShaderProgram->read(gs, size);
			gs[size] = 0;
		}
	}

	if (controlShaderProgram)
	{
		const long size = controlShaderProgram->getSize();
		if (size)
		{
			// if both handles are the same we must reset the file
			if ((controlShaderProgram==vertexShaderProgram) ||
					(controlShaderProgram==pixelShaderProgram) ||
                        (controlShaderProgram==geometryShaderProgram))
				controlShaderProgram->seek(0);
			cts = new c8[size+1];
			controlShaderProgram->read(cts, size);
			cts[size] = 0;
		}
	}

	if (evaluationShaderProgram)
	{
		const long size = evaluationShaderProgram->getSize();
		if (size)
		{
			// if both handles are the same we must reset the file
			if ((evaluationShaderProgram==vertexShaderProgram) ||
					(evaluationShaderProgram==pixelShaderProgram) ||
                        (evaluationShaderProgram==geometryShaderProgram) ||
                            (evaluationShaderProgram==controlShaderProgram))
				evaluationShaderProgram->seek(0);
			ets = new c8[size+1];
			evaluationShaderProgram->read(ets, size);
			ets[size] = 0;
		}
	}


	s32 result = this->addHighLevelShaderMaterial(
		vs, cts, ets, gs, ps,patchVertices,
		baseMaterial, callback,
		xformFeedbackOutputs,xformFeedbackOutputCount, userData,
		vertexShaderEntryPointName,
		controlShaderEntryPointName,
		evaluationShaderEntryPointName,
		geometryShaderEntryPointName,
		pixelShaderEntryPointName);

    if (vs)
        delete [] vs;
    if (ps)
        delete [] ps;
    if (ps)
        delete [] gs;
    if (cts)
        delete [] cts;
    if (ets)
        delete [] ets;

	return result;
}




IRenderBuffer* CNullDriver::addRenderBuffer(const core::dimension2d<u32>& size, const ECOLOR_FORMAT format)
{
	return 0;
}

void CNullDriver::addRenderBuffer(IRenderBuffer* buffer)
{
	RenderBuffers.push_back(buffer);
	RenderBuffers.sort();
}

IFrameBuffer* CNullDriver::addFrameBuffer()
{
	return 0;
}

void CNullDriver::addFrameBuffer(IFrameBuffer* framebuffer)
{
    FrameBuffers.push_back(framebuffer);
    FrameBuffers.sort();
}


void CNullDriver::blitRenderTargets(IFrameBuffer* in, IFrameBuffer* out, bool copyDepth,
									core::recti srcRect, core::recti dstRect,
									bool bilinearFilter)
{
}


//! Clears the ZBuffer.
void CNullDriver::clearZBuffer(const float &depth)
{
}
void CNullDriver::clearStencilBuffer(const int32_t &stencil)
{
}
void CNullDriver::clearZStencilBuffers(const float &depth, const int32_t &stencil)
{
}
void CNullDriver::clearColorBuffer(const E_FBO_ATTACHMENT_POINT &attachment, const int32_t* vals)
{
}
void CNullDriver::clearColorBuffer(const E_FBO_ATTACHMENT_POINT &attachment, const uint32_t* vals)
{
}
void CNullDriver::clearColorBuffer(const E_FBO_ATTACHMENT_POINT &attachment, const float* vals)
{
}
void CNullDriver::clearScreen(const E_SCREEN_BUFFERS &buffer, const float* vals)
{
}
void CNullDriver::clearScreen(const E_SCREEN_BUFFERS &buffer, const uint32_t* vals)
{
}

void CNullDriver::bindTransformFeedback(ITransformFeedback* xformFeedback)
{
    os::Printer::log("Transform Feedback Not supported by this Driver!\n",ELL_ERROR);
}

void CNullDriver::beginTransformFeedback(ITransformFeedback* xformFeedback, const E_MATERIAL_TYPE& xformFeedbackShader, const scene::E_PRIMITIVE_TYPE& primType)
{
    os::Printer::log("Transform Feedback Not supported by this Driver!\n",ELL_ERROR);
}

void CNullDriver::pauseTransformFeedback()
{
    os::Printer::log("Transform Feedback Not supported by this Driver!\n",ELL_ERROR);
}

void CNullDriver::resumeTransformFeedback()
{
    os::Printer::log("Transform Feedback Not supported by this Driver!\n",ELL_ERROR);
}

void CNullDriver::endTransformFeedback()
{
    os::Printer::log("Transform Feedback Not supported by this Driver!\n",ELL_ERROR);
}


// prints renderer version
void CNullDriver::printVersion()
{
	core::stringw namePrint = L"Using renderer: ";
	namePrint += getName();
	os::Printer::log(namePrint.c_str(), ELL_INFORMATION);
}


//! creates a video driver
IVideoDriver* createNullDriver(io::IFileSystem* io, const core::dimension2d<u32>& screenSize)
{
	CNullDriver* nullDriver = new CNullDriver(io, screenSize);

	// create empty material renderers
	for(u32 i=0; sBuiltInMaterialTypeNames[i]; ++i)
	{
		IMaterialRenderer* imr = new IMaterialRenderer();
		nullDriver->addMaterialRenderer(imr);
		imr->drop();
	}

	return nullDriver;
}



//! Enable/disable a clipping plane.
void CNullDriver::enableClipPlane(u32 index, bool enable)
{
	// not necessary
}


void CNullDriver::setMinHardwareBufferVertexCount(u32 count)
{
	MinVertexCountForVBO = count;
}


SOverrideMaterial& CNullDriver::getOverrideMaterial()
{
	return OverrideMaterial;
}


//! Get the 2d override material for altering its values
SMaterial& CNullDriver::getMaterial2D()
{
	return OverrideMaterial2D;
}


//! Enable the 2d override material
void CNullDriver::enableMaterial2D(bool enable)
{
	OverrideMaterial2DEnabled=enable;
}


const uint32_t* CNullDriver::getMaxTextureSize(const ITexture::E_TEXTURE_TYPE& type) const
{
    return MaxTextureSizes[type];
}


//! Color conversion convenience function
/** Convert an image (as array of pixels) from source to destination
array, thereby converting the color format. The pixel size is
determined by the color formats.
\param sP Pointer to source
\param sF Color format of source
\param sN Number of pixels to convert, both array must be large enough
\param dP Pointer to destination
\param dF Color format of destination
*/
void CNullDriver::convertColor(const void* sP, ECOLOR_FORMAT sF, s32 sN,
		void* dP, ECOLOR_FORMAT dF) const
{
	video::CColorConverter::convert_viaFormat(sP, sF, sN, dP, dF);
}


} // end namespace
} // end namespace
