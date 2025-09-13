// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_I_GPU_PIPELINE_CACHE_H_INCLUDED__
#define __NBL_VIDEO_I_GPU_PIPELINE_CACHE_H_INCLUDED__


#include "nbl/asset/ICPUPipelineCache.h"

#include "nbl/video/decl/IBackendObject.h"


namespace nbl::video
{

class IGPUPipelineCache : public IBackendObject
{
	public:
		inline bool merge(const std::span<const IGPUPipelineCache* const> _srcCaches)
		{
			if (_srcCaches.empty())
				return false;
			for (auto cache : _srcCaches)
			if (!cache->isCompatibleDevicewise(this))
				return false;
			return merge_impl(_srcCaches);
		}

		virtual core::smart_refctd_ptr<asset::ICPUPipelineCache> convertToCPUCache() const = 0;
		
	protected:
		explicit IGPUPipelineCache(core::smart_refctd_ptr<const ILogicalDevice>&& dev) : IBackendObject(std::move(dev)) {}
		virtual ~IGPUPipelineCache() = default;

		virtual bool merge_impl(const std::span<const IGPUPipelineCache* const> _srcCaches) = 0;
};

}

#endif