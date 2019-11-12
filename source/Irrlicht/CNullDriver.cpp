// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "CNullDriver.h"
#include "os.h"
#include "IAnimatedMeshSceneNode.h"
#include "irr/asset/CMeshManipulator.h"

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
			: IVideoDriver(dev), FileSystem(io), ViewPort(0,0,0,0), ScreenSize(screenSize), PrimitivesDrawn(0)
{
	#ifdef _IRR_DEBUG
	setDebugName("CNullDriver");
	#endif

    for (size_t i = 0; i < EQOT_COUNT; i++)
        currentQuery[i] = nullptr;

	ViewPort = core::rect<int32_t>(core::position2d<int32_t>(0,0), core::dimension2di(screenSize));


	if (FileSystem)
		FileSystem->grab();

    MaxTextureSizes[IGPUImageView::ET_1D][0] = 0x80u;
    MaxTextureSizes[IGPUImageView::ET_1D][1] = 0x1u;
    MaxTextureSizes[IGPUImageView::ET_1D][2] = 0x1u;

    MaxTextureSizes[IGPUImageView::ET_2D][0] = 0x80u;
    MaxTextureSizes[IGPUImageView::ET_2D][1] = 0x80u;
    MaxTextureSizes[IGPUImageView::ET_2D][2] = 0x1u;

    MaxTextureSizes[IGPUImageView::ET_3D][0] = 0x80u;
    MaxTextureSizes[IGPUImageView::ET_3D][1] = 0x80u;
    MaxTextureSizes[IGPUImageView::ET_3D][2] = 0x80u;

    MaxTextureSizes[IGPUImageView::ET_1D_ARRAY][0] = 0x80u;
    MaxTextureSizes[IGPUImageView::ET_1D_ARRAY][1] = 0x1u;
    MaxTextureSizes[IGPUImageView::ET_1D_ARRAY][2] = 0x800u;

    MaxTextureSizes[IGPUImageView::ET_2D_ARRAY][0] = 0x80u;
    MaxTextureSizes[IGPUImageView::ET_2D_ARRAY][1] = 0x80u;
    MaxTextureSizes[IGPUImageView::ET_2D_ARRAY][2] = 0x800u;

    MaxTextureSizes[IGPUImageView::ET_CUBE_MAP][0] = 0x80u;
    MaxTextureSizes[IGPUImageView::ET_CUBE_MAP][1] = 0x80u;
    MaxTextureSizes[IGPUImageView::ET_CUBE_MAP][2] = 0x6u;

    MaxTextureSizes[IGPUImageView::ET_CUBE_MAP_ARRAY][0] = 0x80u;
    MaxTextureSizes[IGPUImageView::ET_CUBE_MAP_ARRAY][1] = 0x80u;
    MaxTextureSizes[IGPUImageView::ET_CUBE_MAP_ARRAY][2] = 0x800u*6;


	// set ExposedData to 0
	memset(&ExposedData, 0, sizeof(ExposedData));
}


//! destructor
CNullDriver::~CNullDriver()
{
	if (FileSystem)
		FileSystem->drop();
}

//! applications must call this method before performing any rendering. returns false if failed.
bool CNullDriver::beginScene(bool backBuffer, bool zBuffer, SColor color,
		const SExposedVideoData& videoData, core::rect<int32_t>* sourceRect)
{
	PrimitivesDrawn = 0;
	return true;
}


//! applications must call this method after performing any rendering. returns false if failed.
bool CNullDriver::endScene()
{
	FPSCounter.registerFrame(std::chrono::high_resolution_clock::now(), PrimitivesDrawn);

	return true;
}

void CNullDriver::removeFrameBuffer(IFrameBuffer* framebuf)
{
}

void CNullDriver::removeAllFrameBuffers()
{
}

void CNullDriver::bindDescriptorSets_generic(const IGPUPipelineLayout* _newLayout, uint32_t _first, uint32_t _count, const IGPUDescriptorSet** _descSets, const IGPUPipelineLayout** _destPplnLayouts)
{
    uint32_t compatibilityLimits[IGPUPipelineLayout::DESCRIPTOR_SET_COUNT]{}; //actually more like "compatibility limit + 1" (i.e. 0 mean not comaptible at all)
    for (uint32_t i = 0u; i < IGPUPipelineLayout::DESCRIPTOR_SET_COUNT; ++i)
    {
        const uint32_t lim = _destPplnLayouts[i] ? //if no descriptor set bound at this index
            _destPplnLayouts[i]->isCompatibleForSet(IGPUPipelineLayout::DESCRIPTOR_SET_COUNT - 1u, _newLayout) : 0u;

        compatibilityLimits[i] = (lim == IGPUPipelineLayout::DESCRIPTOR_SET_COUNT) ? 0u : (lim + 1u);
    }

    /*
    https://www.khronos.org/registry/vulkan/specs/1.1-extensions/html/vkspec.html#descriptorsets-compatibility
    When binding a descriptor set (see Descriptor Set Binding) to set number N, if the previously bound descriptor sets for sets zero through N-1 were all bound using compatible pipeline layouts, then performing this binding does not disturb any of the lower numbered sets.
    */
    for (uint32_t i = 0u; i < _first; i++)
        if (compatibilityLimits[i] <= i)
            _destPplnLayouts[i] = nullptr;

    /*
    If, additionally, the previous bound descriptor set for set N was bound using a pipeline layout compatible for set N, then the bindings in sets numbered greater than N are also not disturbed.
    */
    if (compatibilityLimits[_first] <= _first)
        for (uint32_t i = _first + _count; i < IGPUPipelineLayout::DESCRIPTOR_SET_COUNT; i++)
            _destPplnLayouts = nullptr;
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

//!
void CNullDriver::drawMeshBuffer(const IGPUMeshBuffer* mb)
{
	if (!mb)
		return;

    uint32_t increment = mb->getInstanceCount();
    switch (mb->getPipeline()->getPrimitiveAssemblyParams().primitiveType)
    {
        case asset::EPT_POINT_LIST:
            increment *= mb->getIndexCount();
            break;
        case asset::EPT_LINE_STRIP:
            increment *= mb->getIndexCount()-1;
            break;
        case asset::EPT_LINE_LIST:
            increment *= mb->getIndexCount()/2;
            break;
        case asset::EPT_TRIANGLE_STRIP:
            increment *= mb->getIndexCount()-2;
            break;
        case asset::EPT_TRIANGLE_FAN:
            increment *= mb->getIndexCount()-2;
            break;
        case asset::EPT_TRIANGLE_LIST:
            increment *= mb->getIndexCount()/3;
            break;
    }
    PrimitivesDrawn += increment;
}

void CNullDriver::beginQuery(IQueryObject* query)
{
    if (!query)
        return; //error

    if (currentQuery[query->getQueryObjectType()])
        return; //error

    query->grab();
    currentQuery[query->getQueryObjectType()] = query;
}
void CNullDriver::endQuery(IQueryObject* query)
{
    if (!query)
        return; //error
    if (currentQuery[query->getQueryObjectType()]!=query)
        return; //error

    if (currentQuery[query->getQueryObjectType()])
        currentQuery[query->getQueryObjectType()]->drop();
    currentQuery[query->getQueryObjectType()] = NULL;
}


//! Only used by the internal engine. Used to notify the driver that
//! the window was resized.
void CNullDriver::OnResize(const core::dimension2d<uint32_t>& size)
{
	ScreenSize = size;
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


// prints renderer version
void CNullDriver::printVersion()
{
	core::stringw namePrint = L"Using renderer: ";
	namePrint += getName();
	os::Printer::log(namePrint.c_str(), ELL_INFORMATION);
}

bool CNullDriver::validateImageCreationParams(const asset::IImage::SCreationParams& _params) const
{
    // some most common validation done
    if (!_params.extent.width || !_params.extent.height || !_params.extent.depth)
        return false;
    if (!_params.mipLevels)
        return false;
    if (!_params.arrayLayers)
        return false;
    if (_params.type == asset::IImage::ET_3D && _params.arrayLayers != 1u)
        return false;
    if ((_params.samples == asset::IImage::ESCF_1_BIT) &&
        ((_params.type != asset::IImage::ET_2D) || (_params.flags & asset::IImage::ECF_CUBE_COMPATIBLE_BIT) || (_params.mipLevels != 1u))
    ) {
        return false;
    }
    if ((_params.flags & asset::IImage::ECF_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT) && !asset::isBlockCompressionFormat(_params.format))
        return false;

    return true;
}

bool CNullDriver::validateImageViewCreationParams(const IGPUImageView::SCreationParams & _params) const
{
    auto ci = _params.image;
    if (!ci)
        return false;

    const auto& img_params = ci->getCreationParameters();
    if (img_params.extent.width == 0u)
        return false;
    if (img_params.arrayLayers == 0u)
        return false;
    if (img_params.type == asset::IImage::ET_3D && _params.viewType == asset::ICPUImageView::ET_3D && _params.subresourceRange.baseArrayLayer != 0u)
        return false;
    if (_params.viewType == IGPUImageView::ET_1D || _params.viewType == IGPUImageView::ET_1D_ARRAY)
    {
        if (img_params.type != asset::IImage::ET_1D)
            return false;
        if (img_params.extent.height != 1u)
            return false;
        if (img_params.extent.depth != 1u)
            return false;
        if (img_params.samples != asset::IImage::ESCF_1_BIT)
            return false;
    }
    else if (_params.viewType == IGPUImageView::ET_2D || _params.viewType == IGPUImageView::ET_2D_ARRAY)
    {
        if (img_params.type != asset::IImage::ET_2D && img_params.type != asset::IImage::ET_3D)
            return false;
        if (img_params.extent.height == 0u)
            return false;
        if (img_params.type == asset::IImage::ET_2D)
        {
            if (img_params.extent.depth != 1u)
                return false;
        }
        else //asset::IImage::ET_3D
        {
            if (_params.subresourceRange.levelCount != 1u)
                return false;
            if (img_params.extent.depth == 0u)
                return false;
            if (img_params.arrayLayers != 1u)
                return false;
            if (img_params.samples != asset::IImage::ESCF_1_BIT)
                return false;
            if (!(img_params.flags & asset::IImage::ECF_2D_ARRAY_COMPATIBLE_BIT))
                return false;
            if (img_params.flags & (asset::IImage::ECF_SPARSE_BINDING_BIT | asset::IImage::ECF_SPARSE_RESIDENCY_BIT | asset::IImage::ECF_SPARSE_ALIASED_BIT))
                return false;

            if (_params.viewType == IGPUImageView::ET_2D)
            {
                if (_params.subresourceRange.layerCount != 1u)
                    return false;
            }
            else //ET_2D_ARRAY
            {
                if (_params.subresourceRange.layerCount == 0u)
                    return false;
            }
        }
    }
    else if (_params.viewType == IGPUImageView::ET_CUBE_MAP || _params.viewType == IGPUImageView::ET_CUBE_MAP_ARRAY)
    {
        if (img_params.type != asset::IImage::ET_2D)
            return false;
        if (img_params.extent.height != img_params.extent.width)
            return false;
        if (img_params.extent.depth != 1u)
            return false;
        if (img_params.samples != asset::IImage::ESCF_1_BIT)
            return false;
        if (img_params.arrayLayers < 6u)
            return false;
        if (!(img_params.flags & asset::IImage::ECF_CUBE_COMPATIBLE_BIT))
            return false;
    }
    else if (_params.viewType == IGPUImageView::ET_3D)
    {
        if (img_params.type != asset::IImage::ET_3D)
            return false;
        if (img_params.extent.height == 0u)
            return false;
        if (img_params.extent.depth == 0u)
            return false;
        if (img_params.arrayLayers != 1u)
            return false;
        if (img_params.samples != asset::IImage::ESCF_1_BIT)
            return false;
    }

    return true;
}


//! creates a video driver
IVideoDriver* createNullDriver(IrrlichtDevice* dev, io::IFileSystem* io, const core::dimension2d<uint32_t>& screenSize)
{
	CNullDriver* nullDriver = new CNullDriver(dev, io, screenSize);

	return nullDriver;
}



//! Enable/disable a clipping plane.
void CNullDriver::enableClipPlane(uint32_t index, bool enable)
{
	// not necessary
}


const uint32_t* CNullDriver::getMaxTextureSize(IGPUImageView::E_TYPE type) const
{
    return MaxTextureSizes[type];
}

} // end namespace
} // end namespace
