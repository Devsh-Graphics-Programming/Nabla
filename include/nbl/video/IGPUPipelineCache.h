// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_I_GPU_PIPELINE_CACHE_H_INCLUDED__
#define __NBL_VIDEO_I_GPU_PIPELINE_CACHE_H_INCLUDED__


#include "nbl/asset/ICPUPipelineCache.h"

#include "nbl/video/decl/IBackendObject.h"


namespace nbl::video
{

class NBL_API IGPUPipelineCache : public core::IReferenceCounted, public IBackendObject
{
	protected:
		virtual ~IGPUPipelineCache() = default;

	public:
		explicit IGPUPipelineCache(core::smart_refctd_ptr<const ILogicalDevice>&& dev) : IBackendObject(std::move(dev)) {}

		virtual void merge(uint32_t _count, const IGPUPipelineCache** _srcCaches) = 0;

		virtual core::smart_refctd_ptr<asset::ICPUPipelineCache> convertToCPUCache() const = 0;
};

}

#endif