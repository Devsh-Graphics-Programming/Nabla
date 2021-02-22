// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_I_GPU_ANIMATION_LIBRARY_H_INCLUDED__
#define __NBL_VIDEO_I_GPU_ANIMATION_LIBRARY_H_INCLUDED__

#include "nbl/asset/asset.h"
#include "IGPUBuffer.h"

namespace nbl
{
namespace video
{

class IGPUAnimationLibrary final : public asset::IAnimationLibrary<IGPUBuffer>
{
        using base_t = asset::IAnimationLibrary<IGPUBuffer>;

    public:
		template<class OtherBufferType>
		inline IGPUAnimationLibrary(
			asset::SBufferBinding<IGPUBuffer>&& _keyframeStorageBinding,
			asset::SBufferBinding<IGPUBuffer>&& _timestampStorageBinding,
			asset::SBufferRange<IGPUBuffer>&& _animationStorageRange,
			const asset::IAnimationLibrary<OtherBufferType>* animationLibraryToCopyNamedRanges) :
				base_t(
					std::move(_keyframeStorageBinding),
					std::move(_timestampStorageBinding),
					animationLibraryToCopyNamedRanges->getAnimationCapacity(),
					std::move(_animationStorageRange)
				)
		{
			base_t::setAnimationNames<OtherBufferType>(animationLibraryToCopyNamedRanges);
		}

		template<typename... Args>
		inline IGPUAnimationLibrary(Args&&... args) : base_t(std::forward<Args>(args)...) {}
};

} // end namespace video
} // end namespace nbl



#endif


