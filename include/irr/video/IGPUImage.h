// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __I_GPU_IMAGE_H_INCLUDED__
#define __I_GPU_IMAGE_H_INCLUDED__

#include "dimension2d.h"
#include "IDriverMemoryBacked.h"

#include "irr/asset/IImage.h"

namespace irr
{
namespace video
{

class IGPUImage : public core::impl::ResolveAlignment<IDriverMemoryBacked,asset::IImage>
{
    public:
        _IRR_RESOLVE_NEW_DELETE_AMBIGUITY(IRenderableVirtualTexture,IDriverMemoryBacked)
    protected:
        _IRR_INTERFACE_CHILD(IGPUImage) {}

        //! constructor
		IGPUImage(	const IDriverMemoryBacked::SDriverMemoryRequirements& reqs,
					E_IMAGE_CREATE_FLAGS _flags,
					E_IMAGE_TYPE _type,
					asset::E_FORMAT _format,
					const asset::VkExtent3D& _extent,
					uint32_t _mipLevels,
					uint32_t _arrayLayers,
					E_SAMPLE_COUNT_FLAGS _samples/*,
					E_IMAGE_TILING _tiling,
					E_IMAGE_USAGE_FLAGS _usage,
					E_SHARING_MODE _sharingMode,
					core::smart_refctd_dynamic_aray<uint32_t>&& _queueFamilyIndices,
					E_IMAGE_LAYOUT _initialLayout*/)
        {
            cachedMemoryReqs = reqs;

			flags = _flags;
			type = _type;
			format = _format;
			extent = _extent;
			mipLevels = _mipLevels;
			arrayLayers = _arrayLayers;
			samples = _samples;
        }
};


} // end namespace video
} // end namespace irr

#endif

