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
        _IRR_RESOLVE_NEW_DELETE_AMBIGUITY(IDriverMemoryBacked,asset::IImage)
			
		//!
		virtual bool validateCopies(const SBufferCopy* pRegionsBegin, const SBufferCopy* pRegionsEnd, const IGPUBuffer* src)
		{
			if (!validateCopies_template(pRegionsBegin, pRegionsEnd, src))
				return false;
			
			#ifdef _IRR_DEBUG // TODO: When Vulkan comes
			#endif
			return true;
		}
			
		virtual bool validateCopies(const SImageCopy* pRegionsBegin, const SImageCopy* pRegionsEnd, const IGPUImage* src)
		{
			if (!validateCopies_template(pRegionsBegin, pRegionsEnd, src))
				return false;

			#ifdef _IRR_DEBUG // TODO: When Vulkan comes
				// image offset and extent must respect granularity requirements
				// buffer has memory bound (with sparse exceptions)
				// check buffer has transfer usage flag
				// format features of dstImage contain transfer dst bit
				// dst image not created subsampled
				// etc.
			#endif
			return true;
		}

        //depr
        virtual void generateMipmaps() = 0;
    protected:
        _IRR_INTERFACE_CHILD(IGPUImage) {}

        //! constructor
		IGPUImage(SCreationParams&& _params)
        {
			params = std::move(_params);
        }
};


} // end namespace video
} // end namespace irr

#endif

