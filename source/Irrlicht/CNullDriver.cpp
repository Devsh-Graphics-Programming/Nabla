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
#include "irr/asset/CMeshManipulator.h"
#include "CMeshSceneNodeInstanced.h"

#include <new>
#include "IrrlichtDevice.h"

namespace irr
{
namespace video
{

FW_AtomicCounter CNullDriver::ReallocationCounter(0);
int32_t CNullDriver::incrementAndFetchReallocCounter()
{
// omg this has to be rewritten
#if defined(FW_MUTEX_H_CXX11_IMPL)
	return ReallocationCounter += 1;
#elif _MSC_VER && !__INTEL_COMPILER
    return InterlockedIncrement(&ReallocationCounter);
#elif defined(__GNUC__)
    return __sync_add_and_fetch(&ReallocationCounter,int32_t(1));
#endif // _MSC_VER
}

//! constructor
CNullDriver::CNullDriver(IrrlichtDevice* dev, io::IFileSystem* io, const core::dimension2d<uint32_t>& screenSize)
: IVideoDriver(dev), FileSystem(io), ViewPort(0,0,0,0), ScreenSize(screenSize), boxLineMesh(0),
	PrimitivesDrawn(0), TextureCreationFlags(0),
	OverrideMaterial2DEnabled(false), AllowZWriteOnTransparent(false),
	matrixModifiedBits(0)
{
	#ifdef _IRR_DEBUG
	setDebugName("CNullDriver");
	#endif

	for (size_t i=0; i<EQOT_COUNT; i++)
    for (size_t j=0; j<_IRR_XFORM_FEEDBACK_MAX_STREAMS_; j++)
        currentQuery[i][j] = NULL;

	setTextureCreationFlag(ETCF_ALWAYS_32_BIT, true);
	setTextureCreationFlag(ETCF_CREATE_MIP_MAPS, true);

	ViewPort = core::rect<int32_t>(core::position2d<int32_t>(0,0), core::dimension2di(screenSize));


	if (FileSystem)
		FileSystem->grab();

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


	// set ExposedData to 0
	memset(&ExposedData, 0, sizeof(ExposedData));

	InitMaterial2D.ZWriteEnable=false;
	InitMaterial2D.ZBuffer=video::ECFN_NEVER;
	for (uint32_t i=0; i<video::MATERIAL_MAX_TEXTURES; ++i)
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

	uint32_t i;
	for (i=0; i<SurfaceLoader.size(); ++i)
		SurfaceLoader[i]->drop();

	for (i=0; i<SurfaceWriter.size(); ++i)
		SurfaceWriter[i]->drop();

	// delete material renderers
	deleteMaterialRenders();
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
uint32_t CNullDriver::getImageLoaderCount() const
{
	return SurfaceLoader.size();
}


//! Retrieve the given image loader
IImageLoader* CNullDriver::getImageLoader(uint32_t n)
{
	if (n < SurfaceLoader.size())
		return SurfaceLoader[n];
	return 0;
}


//! Retrieve the number of image writers
uint32_t CNullDriver::getImageWriterCount() const
{
	return SurfaceWriter.size();
}


//! Retrieve the given image writer
IImageWriter* CNullDriver::getImageWriter(uint32_t n)
{
	if (n < SurfaceWriter.size())
		return SurfaceWriter[n];
	return 0;
}

//! applications must call this method before performing any rendering. returns false if failed.
bool CNullDriver::beginScene(bool backBuffer, bool zBuffer, SColor color,
		const SExposedVideoData& videoData, core::rect<int32_t>* sourceRect)
{
	scene::CMeshSceneNodeInstanced::recullOrder = 0;

	PrimitivesDrawn = 0;
	return true;
}


//! applications must call this method after performing any rendering. returns false if failed.
bool CNullDriver::endScene()
{
	FPSCounter.registerFrame(std::chrono::high_resolution_clock::now(), PrimitivesDrawn);

	return true;
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
void CNullDriver::setTransform(const E_PROJECTION_TRANSFORMATION_STATE& state, const core::matrix4SIMD& mat)
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
const core::matrix4SIMD& CNullDriver::getTransform(const E_PROJECTION_TRANSFORMATION_STATE& state)
{
    const uint32_t stateBit = 0x1u<<(state+E4X3TS_COUNT);

	if (matrixModifiedBits&stateBit)
    {
        switch (state)
        {
            case EPTS_PROJ:
                break;
            case EPTS_PROJ_VIEW:
                ProjectionMatrices[EPTS_PROJ_VIEW] = core::concatenateBFollowedByA(ProjectionMatrices[EPTS_PROJ],TransformationMatrices[E4X3TS_VIEW]);
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
                ProjectionMatrices[EPTS_PROJ].getInverseTransform(ProjectionMatrices[EPTS_PROJ]);
                break;
            case EPTS_PROJ_VIEW_INVERSE:
                if (matrixModifiedBits&(0x1u<<(EPTS_PROJ_VIEW+E4X3TS_COUNT)))
                {
                    ProjectionMatrices[EPTS_PROJ_VIEW] = concatenateBFollowedByA(ProjectionMatrices[EPTS_PROJ],TransformationMatrices[E4X3TS_VIEW]);
                    matrixModifiedBits &= ~(0x1u<<(EPTS_PROJ_VIEW+E4X3TS_COUNT));
                }
                ProjectionMatrices[EPTS_PROJ_VIEW].getInverseTransform(ProjectionMatrices[EPTS_PROJ_VIEW_INVERSE]);
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
                ProjectionMatrices[EPTS_PROJ_VIEW_WORLD].getInverseTransform(ProjectionMatrices[EPTS_PROJ_VIEW_WORLD_INVERSE]);
                break;
        }

        matrixModifiedBits &= ~stateBit;
    }
    return ProjectionMatrices[state];
}


//! sets a material
void CNullDriver::setMaterial(const SGPUMaterial& material)
{
}

void CNullDriver::removeMultisampleTexture(IMultisampleTexture* tex)
{
    auto it = std::lower_bound(MultisampleTextures.begin(),MultisampleTextures.end(),tex);
    if (it==MultisampleTextures.end() || tex<*it)
        return;
    MultisampleTextures.erase(it);

    tex->drop();
}

void CNullDriver::removeTextureBufferObject(ITextureBufferObject* tbo)
{
    auto it = std::lower_bound(TextureBufferObjects.begin(),TextureBufferObjects.end(),tbo);
    if (it==TextureBufferObjects.end() || tbo<*it)
        return;
    TextureBufferObjects.erase(it);

    tbo->drop();
}

void CNullDriver::removeFrameBuffer(IFrameBuffer* framebuf)
{
}

void CNullDriver::removeAllMultisampleTextures()
{
	setMaterial ( SGPUMaterial() );

	for (uint32_t i=0; i<MultisampleTextures.size(); ++i)
		MultisampleTextures[i]->drop();
    MultisampleTextures.clear();
}

void CNullDriver::removeAllTextureBufferObjects()
{
	setMaterial ( SGPUMaterial() );

	for (uint32_t i=0; i<TextureBufferObjects.size(); ++i)
		TextureBufferObjects[i]->drop();
    TextureBufferObjects.clear();
}

void CNullDriver::removeAllFrameBuffers()
{
}

ITexture* CNullDriver::createGPUTexture(const ITexture::E_TEXTURE_TYPE& type, const uint32_t* size, uint32_t mipmapLevels, asset::E_FORMAT format)
{
    if (type != ITexture::ETT_2D)
        return nullptr;

    return new SDummyTexture("");
}


//! returns a device dependent texture from a software surface (IImage)
//! THIS METHOD HAS TO BE OVERRIDDEN BY DERIVED DRIVERS WITH OWN TEXTURES
ITexture* CNullDriver::createDeviceDependentTexture(const ITexture::E_TEXTURE_TYPE& type, const uint32_t* size, uint32_t mipmapLevels,
			const io::path& name, asset::E_FORMAT format)
{
    //better safe than sorry
    if (type!=ITexture::ETT_2D)
        return NULL;

	return new SDummyTexture(name);
}


//! sets a render target
bool CNullDriver::setRenderTarget(video::IFrameBuffer* texture, bool setNewViewport)
{
	return false;
}


//! sets a viewport
void CNullDriver::setViewPort(const core::rect<int32_t>& area)
{
}


//! gets the area of the current viewport
const core::rect<int32_t>& CNullDriver::getViewPort() const
{
	return ViewPort;
}



//! draws an 2d image
void CNullDriver::draw2DImage(const video::ITexture* texture, const core::position2d<int32_t>& destPos)
{
	if (!texture)
		return;

	draw2DImage(texture,destPos, core::rect<int32_t>(core::position2d<int32_t>(0,0),
												core::dimension2di(*reinterpret_cast<const core::dimension2du*>(texture->getSize()))));
}



//! draws a set of 2d images, using a color and the alpha channel of the
//! texture if desired. The images are drawn beginning at pos and concatenated
//! in one line. All drawings are clipped against clipRect (if != 0).
//! The subtextures are defined by the array of sourceRects and are chosen
//! by the indices given.
void CNullDriver::draw2DImageBatch(const video::ITexture* texture,
				const core::position2d<int32_t>& pos,
				const core::vector<core::rect<int32_t> >& sourceRects,
				const core::vector<int32_t>& indices,
				int32_t kerningWidth,
				const core::rect<int32_t>* clipRect, SColor color,
				bool useAlphaChannelOfTexture)
{
	core::position2d<int32_t> target(pos);

	for (uint32_t i=0; i<indices.size(); ++i)
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
				const core::vector<core::position2d<int32_t> >& positions,
				const core::vector<core::rect<int32_t> >& sourceRects,
				const core::rect<int32_t>* clipRect,
				SColor color,
				bool useAlphaChannelOfTexture)
{
	const uint32_t drawCount = core::min_<uint32_t>(positions.size(), sourceRects.size());

	for (uint32_t i=0; i<drawCount; ++i)
	{
		draw2DImage(texture, positions[i], sourceRects[i],
				clipRect, color, useAlphaChannelOfTexture);
	}
}


//! Draws a part of the texture into the rectangle.
void CNullDriver::draw2DImage(const video::ITexture* texture, const core::rect<int32_t>& destRect,
	const core::rect<int32_t>& sourceRect, const core::rect<int32_t>* clipRect,
	const video::SColor* const colors, bool useAlphaChannelOfTexture)
{
	if (destRect.isValid())
		draw2DImage(texture, core::position2d<int32_t>(destRect.UpperLeftCorner),
				sourceRect, clipRect, colors?colors[0]:video::SColor(0xffffffff),
				useAlphaChannelOfTexture);
}


//! Draws a 2d image, using a color (if color is other then Color(255,255,255,255)) and the alpha channel of the texture if wanted.
void CNullDriver::draw2DImage(const video::ITexture* texture, const core::position2d<int32_t>& destPos,
				const core::rect<int32_t>& sourceRect,
				const core::rect<int32_t>* clipRect, SColor color,
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
void CNullDriver::draw2DRectangle(SColor color, const core::rect<int32_t>& pos, const core::rect<int32_t>* clip)
{
	draw2DRectangle(pos, color, color, color, color, clip);
}



//! Draws a 2d rectangle with a gradient.
void CNullDriver::draw2DRectangle(const core::rect<int32_t>& pos,
	SColor colorLeftUp, SColor colorRightUp, SColor colorLeftDown, SColor colorRightDown,
	const core::rect<int32_t>* clip)
{
}



//! Draws a 2d line.
void CNullDriver::draw2DLine(const core::position2d<int32_t>& start,
				const core::position2d<int32_t>& end, SColor color)
{
}


//! returns color format
asset::E_FORMAT CNullDriver::getColorFormat() const
{
	return asset::EF_B5G6R5_UNORM_PACK16;
}


//! returns screen size
const core::dimension2d<uint32_t>& CNullDriver::getScreenSize() const
{
	return ScreenSize;
}


//! returns the current render target size,
//! or the screen size if render targets are not implemented
const core::dimension2d<uint32_t>& CNullDriver::getCurrentRenderTargetSize() const
{
	return ScreenSize;
}


// returns current frames per second value
int32_t CNullDriver::getFPS() const
{
	return FPSCounter.getFPS();
}



//! returns amount of primitives (mostly triangles) were drawn in the last frame.
//! very useful method for statistics.
uint32_t CNullDriver::getPrimitiveCountDrawn( uint32_t param ) const
{
	return (0 == param) ? FPSCounter.getPrimitive() : (1 == param) ? FPSCounter.getPrimitiveAverage() : FPSCounter.getPrimitiveTotal();
}



//! \return Returns the name of the video driver. Example: In case of the DIRECT3D8
//! driver, it would return "Direct3D8".

const wchar_t* CNullDriver::getName() const
{
	return L"Irrlicht NullDevice";
}



//! Returns the maximum amount of primitives (mostly vertices) which
//! the device is able to render with one drawIndexedTriangleList
//! call.
uint32_t CNullDriver::getMaximalIndicesCount() const
{
	return 0xFFFFFFFF;
}


//! Enables or disables a texture creation flag.
void CNullDriver::setTextureCreationFlag(E_TEXTURE_CREATION_FLAG flag, bool enabled)
{
	if (enabled && ((flag == ETCF_ALWAYS_16_BIT) || (flag == ETCF_ALWAYS_32_BIT)))
	{
		// disable other formats
		setTextureCreationFlag(ETCF_ALWAYS_16_BIT, false);
		setTextureCreationFlag(ETCF_ALWAYS_32_BIT, false);
	}

	// set flag
	TextureCreationFlags = (TextureCreationFlags & (~flag)) |
		((((uint32_t)!enabled)-1) & flag);
}


//! Returns if a texture creation flag is enabled or disabled.
bool CNullDriver::getTextureCreationFlag(E_TEXTURE_CREATION_FLAG flag) const
{
	return (TextureCreationFlags & flag)!=0;
}


//! Creates a software image from a file.
core::vector<asset::CImageData*> CNullDriver::createImageDataFromFile(const io::path& filename)
{
	if (!filename.size())
		return core::vector<asset::CImageData*>();

	io::IReadFile* file = FileSystem->createAndOpenFile(filename);

	if (file)
	{
		core::vector<asset::CImageData*> imageData = createImageDataFromFile(file);
		file->drop();
		return imageData;
	}
//	else
//		os::Printer::log("Could not open file of image", filename, ELL_WARNING);	// sodan

	return core::vector<asset::CImageData*>();
}


//! Creates a software image from a file.
core::vector<asset::CImageData*> CNullDriver::createImageDataFromFile(io::IReadFile* file)
{
	if (!file)
		return core::vector<asset::CImageData*>();


	int32_t i;

	// try to load file based on file extension
	for (i=SurfaceLoader.size()-1; i>=0; --i)
	{
		if (SurfaceLoader[i]->isALoadableFileExtension(file->getFileName()))
		{
			// reset file position which might have changed due to previous loadImage calls
			file->seek(0);
			core::vector<asset::CImageData*> imageData = SurfaceLoader[i]->loadImage(file);
			if (imageData.size())
				return imageData;
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
			core::vector<asset::CImageData*> imageData = SurfaceLoader[i]->loadImage(file);
			if (imageData.size())
				return imageData;
		}
	}

	return core::vector<asset::CImageData*>(); // failed to load
}


//! Writes the provided image to disk file
bool CNullDriver::writeImageToFile(IImage* image, const io::path& filename,uint32_t param)
{
	io::IWriteFile* file = FileSystem->createAndWriteFile(filename);
	if(!file)
		return false;

	bool result = writeImageToFile(image, file, param);
	file->drop();

	return result;
}

//! Writes the provided image to a file.
bool CNullDriver::writeImageToFile(IImage* image, io::IWriteFile * file, uint32_t param)
{
	if(!file)
		return false;

	for (int32_t i=SurfaceWriter.size()-1; i>=0; --i)
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
IImage* CNullDriver::createImageFromData(asset::CImageData* imageData, bool ownForeignMemory)
{
    core::dimension2du size;
    size.Width = imageData->getSliceMax()[0]-imageData->getSliceMin()[0];
    size.Height = imageData->getSliceMax()[1]-imageData->getSliceMin()[1];
	CImage* img = new CImage(imageData->getColorFormat(), size, imageData->getData(), ownForeignMemory);

	if (ownForeignMemory)
        imageData->forgetAboutData();

	return img;
}

//!
IImage* CNullDriver::createImage(const asset::E_FORMAT& format, const core::dimension2d<uint32_t>& size)
{
    return new CImage(format, size);
}


void CNullDriver::drawMeshBuffer(const IGPUMeshBuffer* mb)
{
	if (!mb)
		return;

    uint32_t increment = mb->getInstanceCount();
    switch (mb->getPrimitiveType())
    {
        case asset::EPT_POINTS:
            increment *= mb->getIndexCount();
            break;
        case asset::EPT_LINE_STRIP:
            increment *= mb->getIndexCount()-1;
            break;
        case asset::EPT_LINE_LOOP:
            increment *= mb->getIndexCount();
            break;
        case asset::EPT_LINES:
            increment *= mb->getIndexCount()/2;
            break;
        case asset::EPT_TRIANGLE_STRIP:
            increment *= mb->getIndexCount()-2;
            break;
        case asset::EPT_TRIANGLE_FAN:
            increment *= mb->getIndexCount()-2;
            break;
        case asset::EPT_TRIANGLES:
            increment *= mb->getIndexCount()/3;
            break;
    }
    PrimitivesDrawn += increment;
}


//! Indirect Draw
void CNullDriver::drawArraysIndirect(const asset::IMeshDataFormatDesc<video::IGPUBuffer>* vao,
                                     const asset::E_PRIMITIVE_TYPE& mode,
                                     const IGPUBuffer* indirectDrawBuff,
                                     const size_t& offset, const size_t& count, const size_t& stride)
{
}

void CNullDriver::drawIndexedIndirect(  const asset::IMeshDataFormatDesc<video::IGPUBuffer>* vao,
                                        const asset::E_PRIMITIVE_TYPE& mode,
                                        const asset::E_INDEX_TYPE& type,
                                        const IGPUBuffer* indirectDrawBuff,
                                        const size_t& offset, const size_t& count, const size_t& stride)
{
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
void CNullDriver::OnResize(const core::dimension2d<uint32_t>& size)
{
	ScreenSize = size;
}


// adds a material renderer and drops it afterwards. To be used for internal creation
int32_t CNullDriver::addAndDropMaterialRenderer(IMaterialRenderer* m)
{
	int32_t i = addMaterialRenderer(m);

	if (m)
		m->drop();

	return i;
}


//! Adds a new material renderer to the video device.
int32_t CNullDriver::addMaterialRenderer(IMaterialRenderer* renderer, const char* name)
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
void CNullDriver::setMaterialRendererName(int32_t idx, const char* name)
{
	if (idx < int32_t(sizeof(sBuiltInMaterialTypeNames) / sizeof(char*))-1 ||
		idx >= (int32_t)MaterialRenderers.size())
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
	for (uint32_t i=0; i<MaterialRenderers.size(); ++i)
    {
		if (MaterialRenderers[i].Renderer)
			MaterialRenderers[i].Renderer->drop();
    }

	MaterialRenderers.clear();
}


//! Returns pointer to material renderer or null
IMaterialRenderer* CNullDriver::getMaterialRenderer(uint32_t idx)
{
	if ( idx < MaterialRenderers.size() )
		return MaterialRenderers[idx].Renderer;
	else
		return 0;
}


//! Returns amount of currently available material renderers.
uint32_t CNullDriver::getMaterialRendererCount() const
{
	return MaterialRenderers.size();
}


//! Returns name of the material renderer
const char* CNullDriver::getMaterialRendererName(uint32_t idx) const
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

int32_t CNullDriver::addHighLevelShaderMaterial(
    const char* vertexShaderProgram,
    const char* controlShaderProgram,
    const char* evaluationShaderProgram,
    const char* geometryShaderProgram,
    const char* pixelShaderProgram,
    uint32_t patchVertices,
    E_MATERIAL_TYPE baseMaterial,
    IShaderConstantSetCallBack* callback,
    const char** xformFeedbackOutputs,
    const uint32_t& xformFeedbackOutputCount,
    int32_t userData,
    const char* vertexShaderEntryPointName,
    const char* controlShaderEntryPointName,
    const char* evaluationShaderEntryPointName,
    const char* geometryShaderEntryPointName,
    const char* pixelShaderEntryPointName)
{
	os::Printer::log("High level shader materials not available (yet) in this driver, sorry");
	return -1;
}

bool CNullDriver::replaceHighLevelShaderMaterial(const int32_t &materialIDToReplace,
    const char* vertexShaderProgram,
    const char* controlShaderProgram,
    const char* evaluationShaderProgram,
    const char* geometryShaderProgram,
    const char* pixelShaderProgram,
    uint32_t patchVertices,
    E_MATERIAL_TYPE baseMaterial,
    IShaderConstantSetCallBack* callback,
    const char** xformFeedbackOutputs,
    const uint32_t& xformFeedbackOutputCount,
    int32_t userData,
    const char* vertexShaderEntryPointName,
    const char* controlShaderEntryPointName,
    const char* evaluationShaderEntryPointName,
    const char* geometryShaderEntryPointName,
    const char* pixelShaderEntryPointName)
{
    if (materialIDToReplace<0 || MaterialRenderers.size()<static_cast<uint32_t>(materialIDToReplace))
        return false;

    int32_t nr = addHighLevelShaderMaterial(vertexShaderProgram,controlShaderProgram,evaluationShaderProgram,geometryShaderProgram,pixelShaderProgram,
                                        3,baseMaterial,callback,xformFeedbackOutputs,xformFeedbackOutputCount,userData,
                                        vertexShaderEntryPointName,controlShaderEntryPointName,evaluationShaderEntryPointName,geometryShaderEntryPointName,pixelShaderEntryPointName);
    if (nr==-1)
        return false;

	MaterialRenderers[materialIDToReplace].Renderer->drop();
	MaterialRenderers[materialIDToReplace] = MaterialRenderers[MaterialRenderers.size()-1];

	MaterialRenderers.resize(MaterialRenderers.size()-1);

    return true;
}

int32_t CNullDriver::addHighLevelShaderMaterialFromFiles(
    const io::path& vertexShaderProgramFileName,
    const io::path& controlShaderProgramFileName,
    const io::path& evaluationShaderProgramFileName,
    const io::path& geometryShaderProgramFileName,
    const io::path& pixelShaderProgramFileName,
    uint32_t patchVertices,
    E_MATERIAL_TYPE baseMaterial,
    IShaderConstantSetCallBack* callback,
    const char** xformFeedbackOutputs,
    const uint32_t& xformFeedbackOutputCount,
    int32_t userData,
    const char* vertexShaderEntryPointName,
    const char* controlShaderEntryPointName,
    const char* evaluationShaderEntryPointName,
    const char* geometryShaderEntryPointName,
    const char* pixelShaderEntryPointName)
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
				vertexShaderProgramFileName.c_str(), ELL_WARNING);
		}
	}

	if (controlShaderProgramFileName.size() )
	{
		ctsfile = FileSystem->createAndOpenFile(controlShaderProgramFileName);
		if (!ctsfile)
		{
			os::Printer::log("Could not open control shader program file",
				controlShaderProgramFileName.c_str(), ELL_WARNING);
		}
	}

	if (evaluationShaderProgramFileName.size() )
	{
		etsfile = FileSystem->createAndOpenFile(evaluationShaderProgramFileName);
		if (!etsfile)
		{
			os::Printer::log("Could not open evaluation shader program file",
				evaluationShaderProgramFileName.c_str(), ELL_WARNING);
		}
	}

	if (geometryShaderProgramFileName.size() )
	{
		gsfile = FileSystem->createAndOpenFile(geometryShaderProgramFileName);
		if (!gsfile)
		{
			os::Printer::log("Could not open geometry shader program file",
				geometryShaderProgramFileName.c_str(), ELL_WARNING);
		}
	}

	if (pixelShaderProgramFileName.size() )
	{
		psfile = FileSystem->createAndOpenFile(pixelShaderProgramFileName);
		if (!psfile)
		{
			os::Printer::log("Could not open pixel shader program file",
				pixelShaderProgramFileName.c_str(), ELL_WARNING);
		}
	}

	int32_t result = addHighLevelShaderMaterialFromFiles(
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

int32_t CNullDriver::addHighLevelShaderMaterialFromFiles(
    io::IReadFile* vertexShaderProgram,
    io::IReadFile* controlShaderProgram,
    io::IReadFile* evaluationShaderProgram,
    io::IReadFile* geometryShaderProgram,
    io::IReadFile* pixelShaderProgram,
    uint32_t patchVertices,
    E_MATERIAL_TYPE baseMaterial,
    IShaderConstantSetCallBack* callback,
    const char** xformFeedbackOutputs,
    const uint32_t& xformFeedbackOutputCount,
    int32_t userData,
    const char* vertexShaderEntryPointName,
    const char* controlShaderEntryPointName,
    const char* evaluationShaderEntryPointName,
    const char* geometryShaderEntryPointName,
    const char* pixelShaderEntryPointName)
{
	char* vs = 0;
	char* cts = 0;
	char* ets = 0;
	char* gs = 0;
	char* ps = 0;

	if (vertexShaderProgram)
	{
		const long size = vertexShaderProgram->getSize();
		if (size)
		{
			vs = new char[size+1];
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
			ps = new char[size+1];
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
			gs = new char[size+1];
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
			cts = new char[size+1];
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
			ets = new char[size+1];
			evaluationShaderProgram->read(ets, size);
			ets[size] = 0;
		}
	}


	int32_t result = this->addHighLevelShaderMaterial(
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
    if (gs)
        delete [] gs;
    if (cts)
        delete [] cts;
    if (ets)
        delete [] ets;

	return result;
}

void CNullDriver::addMultisampleTexture(IMultisampleTexture* tex)
{
	MultisampleTextures.push_back(tex);
	std::sort(MultisampleTextures.begin(),MultisampleTextures.end());
}

void CNullDriver::addTextureBufferObject(ITextureBufferObject* tbo)
{
	TextureBufferObjects.push_back(tbo);
	std::sort(TextureBufferObjects.begin(),TextureBufferObjects.end());
}


void CNullDriver::blitRenderTargets(IFrameBuffer* in, IFrameBuffer* out, bool copyDepth, bool copyStencil,
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

void CNullDriver::beginTransformFeedback(ITransformFeedback* xformFeedback, const E_MATERIAL_TYPE& xformFeedbackShader, const asset::E_PRIMITIVE_TYPE& primType)
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
IVideoDriver* createNullDriver(IrrlichtDevice* dev, io::IFileSystem* io, const core::dimension2d<uint32_t>& screenSize)
{
	CNullDriver* nullDriver = new CNullDriver(dev, io, screenSize);

	// create empty material renderers
	for(uint32_t i=0; sBuiltInMaterialTypeNames[i]; ++i)
	{
		IMaterialRenderer* imr = new IMaterialRenderer();
		nullDriver->addMaterialRenderer(imr);
		imr->drop();
	}

	return nullDriver;
}



//! Enable/disable a clipping plane.
void CNullDriver::enableClipPlane(uint32_t index, bool enable)
{
	// not necessary
}

//! Get the 2d override material for altering its values
SGPUMaterial& CNullDriver::getMaterial2D()
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

} // end namespace
} // end namespace
