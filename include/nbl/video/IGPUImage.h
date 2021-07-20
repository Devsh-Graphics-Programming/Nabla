// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_GPU_IMAGE_H_INCLUDED__
#define __NBL_VIDEO_GPU_IMAGE_H_INCLUDED__

#include "dimension2d.h"
#include "IDriverMemoryBacked.h"

#include "nbl/asset/IImage.h"

#include "IGPUBuffer.h"
#include "nbl/video/IBackendObject.h"

namespace nbl
{
namespace video
{

class IGPUImage : public core::impl::ResolveAlignment<IDriverMemoryBacked,asset::IImage>, public IBackendObject
{
    public:
        _NBL_RESOLVE_NEW_DELETE_AMBIGUITY(IDriverMemoryBacked,asset::IImage)
			
		//!
		virtual bool validateCopies(const SBufferCopy* pRegionsBegin, const SBufferCopy* pRegionsEnd, const IGPUBuffer* src) const
		{
			if (!validateCopies_template(pRegionsBegin, pRegionsEnd, src))
				return false;
			
			#ifdef _NBL_DEBUG // TODO: When Vulkan comes
			#endif
			return true;
		}
			
		virtual bool validateCopies(const SImageCopy* pRegionsBegin, const SImageCopy* pRegionsEnd, const IGPUImage* src) const
		{
			if (!validateCopies_template(pRegionsBegin, pRegionsEnd, src))
				return false;

			#ifdef _NBL_DEBUG // TODO: When Vulkan comes
				// image offset and extent must respect granularity requirements
				// buffer has memory bound (with sparse exceptions)
				// check buffer has transfer usage flag
				// format features of dstImage contain transfer dst bit
				// dst image not created subsampled
				// etc.
			#endif
			return true;
		}

    protected:
        _NBL_INTERFACE_CHILD(IGPUImage) {}

        //! constructor
		IGPUImage(ILogicalDevice* dev, SCreationParams&& _params) : IBackendObject(dev)
        {
			params = std::move(_params);
        }
};


} // end namespace video
} // end namespace nbl

#endif

