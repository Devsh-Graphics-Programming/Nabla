// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_GPU_IMAGE_H_INCLUDED__
#define __NBL_VIDEO_GPU_IMAGE_H_INCLUDED__


#include "dimension2d.h"
#include "IDriverMemoryBacked.h"

#include "nbl/asset/IImage.h"

#include "nbl/video/IGPUBuffer.h"
#include "nbl/video/decl/IBackendObject.h"


namespace nbl::video
{

class IGPUImage : public core::impl::ResolveAlignment<IDriverMemoryBacked,asset::IImage>, public IBackendObject
{
	private:
		using base_t = core::impl::ResolveAlignment<IDriverMemoryBacked, asset::IImage>;

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

		inline const SDriverMemoryRequirements& getMemoryReqs() const { return base_t::getMemoryReqs(); }

    protected:
        _NBL_INTERFACE_CHILD(IGPUImage) {}

        //! constructor
		IGPUImage(core::smart_refctd_ptr<const ILogicalDevice>&& dev,
			SCreationParams&& _params,
			const IDriverMemoryBacked::SDriverMemoryRequirements reqs = IDriverMemoryBacked::SDriverMemoryRequirements())
			: base_t(reqs), IBackendObject(std::move(dev))
        {
			params = std::move(_params);
        }
};


} // end namespace nbl::video

#endif

