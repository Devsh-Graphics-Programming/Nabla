// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __IRR_I_GPU_PIPELINE_CACHE_H_INCLUDED__
#define __IRR_I_GPU_PIPELINE_CACHE_H_INCLUDED__

#include "irr/core/IReferenceCounted.h"
#include "irr/asset/ICPUPipelineCache.h"

namespace irr { namespace video
{

class IGPUPipelineCache : public core::IReferenceCounted
{
protected:
	virtual ~IGPUPipelineCache() = default;

public:
	virtual void merge(uint32_t _count, const IGPUPipelineCache** _srcCaches) = 0;

	virtual core::smart_refctd_ptr<asset::ICPUPipelineCache> convertToCPUCache() const = 0;
};

}}

#endif//__IRR_I_GPU_PIPELINE_CACHE_H_INCLUDED__