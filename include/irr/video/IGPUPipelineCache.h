#ifndef __IRR_I_GPU_PIPELINE_CACHE_H_INCLUDED__
#define __IRR_I_GPU_PIPELINE_CACHE_H_INCLUDED__

#include "irr/core/IReferenceCounted.h"

namespace irr { namespace video
{

class IGPUPipelineCache : public core::IReferenceCounted
{
protected:
	virtual ~IGPUPipelineCache() = default;

public:
	virtual void merge(uint32_t _count, const IGPUPipelineCache** _srcCaches) = 0;
};

}}

#endif//__IRR_I_GPU_PIPELINE_CACHE_H_INCLUDED__