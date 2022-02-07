// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_I_GPU_RENDERPASS_INDEPENDENT_PIPELINE_H_INCLUDED__
#define __NBL_VIDEO_I_GPU_RENDERPASS_INDEPENDENT_PIPELINE_H_INCLUDED__

#include "nbl/asset/IRenderpassIndependentPipeline.h"
#include "nbl/video/IGPUSpecializedShader.h"
#include "nbl/video/IGPUPipelineLayout.h"

namespace nbl
{
namespace video
{
//! GPU Version of Renderpass Independent Pipeline
/*
	@see IRenderpassIndependentPipeline
*/

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