// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_I_GPU_RENDERPASS_INDEPENDENT_PIPELINE_H_INCLUDED_
#define _NBL_VIDEO_I_GPU_RENDERPASS_INDEPENDENT_PIPELINE_H_INCLUDED_


#include "nbl/asset/IRenderpassIndependentPipeline.h"

#include "nbl/video/IGPUShader.h"
#include "nbl/video/IPipeline.h"


namespace nbl::video
{

//! GPU Version of Renderpass Independent Pipeline
/*
	@see IRenderpassIndependentPipeline
*/
class IGPURenderpassIndependentPipeline : public IPipeline<IGPURenderpassIndependentPipeline>, public asset::IRenderpassIndependentPipeline<IGPUShader>
{
		using pipeline_t = IPipeline<IGPURenderpassIndependentPipeline>;
		using base_t = asset::IRenderpassIndependentPipeline<IGPUShader>;

	public:
		struct SCreationParams final : pipeline_t::SCreationParams, base_t::SCreationParams
		{
            #define base_flag(F) static_cast<uint64_t>(pipeline_t::SCreationParams::FLAGS::F)
            enum class FLAGS : uint64_t
            {
                NONE = base_flag(NONE),
                DISABLE_OPTIMIZATIONS = base_flag(DISABLE_OPTIMIZATIONS),
                ALLOW_DERIVATIVES = base_flag(ALLOW_DERIVATIVES),
                VIEW_INDEX_FROM_DEVICE_INDEX = 1<<3,
                CAPTURE_STATISTICS = base_flag(CAPTURE_STATISTICS),
                CAPTURE_INTERNAL_REPRESENTATIONS = base_flag(CAPTURE_INTERNAL_REPRESENTATIONS),
                FAIL_ON_PIPELINE_COMPILE_REQUIRED = base_flag(FAIL_ON_PIPELINE_COMPILE_REQUIRED),
                EARLY_RETURN_ON_FAILURE = base_flag(EARLY_RETURN_ON_FAILURE),
                LINK_TIME_OPTIMIZATION = base_flag(LINK_TIME_OPTIMIZATION),
                RETAIN_LINK_TIME_OPTIMIZATION_INFO = base_flag(RETAIN_LINK_TIME_OPTIMIZATION_INFO)
            };
            #undef base_flag
		};

	protected:
		IGPURenderpassIndependentPipeline(core::smart_refctd_ptr<const ILogicalDevice>&& dev, const base_t::SCreationParams& params) : pipeline_t(std::move(dev)), base_t(params) {}
		virtual ~IGPURenderpassIndependentPipeline() = default;
};

}

#endif