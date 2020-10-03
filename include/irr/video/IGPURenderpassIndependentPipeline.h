// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_I_GPU_RENDERPASS_INDEPENDENT_PIPELINE_H_INCLUDED__
#define __NBL_I_GPU_RENDERPASS_INDEPENDENT_PIPELINE_H_INCLUDED__

#include "irr/asset/IRenderpassIndependentPipeline.h"
#include "irr/video/IGPUSpecializedShader.h"
#include "irr/video/IGPUPipelineLayout.h"

namespace irr
{
namespace video
{

class IGPURenderpassIndependentPipeline : public asset::IRenderpassIndependentPipeline<IGPUSpecializedShader, IGPUPipelineLayout>
{
		using base_t = asset::IRenderpassIndependentPipeline<IGPUSpecializedShader, IGPUPipelineLayout>;

	public:
		using base_t::base_t;

	protected:
		virtual ~IGPURenderpassIndependentPipeline() = default;
};

}
}

#endif