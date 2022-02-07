// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_I_GPU_COMPUTE_PIPELINE_H_INCLUDED__
#define __NBL_VIDEO_I_GPU_COMPUTE_PIPELINE_H_INCLUDED__

#include "nbl/asset/IComputePipeline.h"
#include "nbl/video/IGPUSpecializedShader.h"
#include "nbl/video/IGPUPipelineLayout.h"

namespace nbl
{
namespace video
{
class IGPUComputePipeline : public asset::IComputePipeline<IGPUSpecializedShader, IGPUPipelineLayout>
{
    using base_t = asset::IComputePipeline<IGPUSpecializedShader, IGPUPipelineLayout>;

public:
    using base_t::base_t;

protected:
    virtual ~IGPUComputePipeline() = default;

    bool m_allowDispatchBase = false;
};

}
}

#endif