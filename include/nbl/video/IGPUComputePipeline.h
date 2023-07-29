// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_I_GPU_COMPUTE_PIPELINE_H_INCLUDED_
#define _NBL_VIDEO_I_GPU_COMPUTE_PIPELINE_H_INCLUDED_


#include "nbl/asset/IComputePipeline.h"

#include "nbl/video/IGPUSpecializedShader.h"
#include "nbl/video/IGPUPipelineLayout.h"


namespace nbl::video
{

class IGPUComputePipeline : public IBackendObject, public asset::IComputePipeline<IGPUSpecializedShader,IGPUPipelineLayout>
{
        using base_t = asset::IComputePipeline<IGPUSpecializedShader, IGPUPipelineLayout>;

    public:
        IGPUComputePipeline(
            core::smart_refctd_ptr<const ILogicalDevice>&& dev,
            core::smart_refctd_ptr<IGPUPipelineLayout>&& _layout,
            core::smart_refctd_ptr<IGPUSpecializedShader>&& _cs
        ) : IBackendObject(std::move(dev)), base_t(std::move(_layout),std::move(_cs)) {}

        struct SCreationParams
        {
            IPipeline::E_PIPELINE_CREATION flags;
            core::smart_refctd_ptr<IGPUPipelineLayout> layout;
            core::smart_refctd_ptr<IGPUSpecializedShader> shader;
            core::smart_refctd_ptr<IGPUComputePipeline> basePipeline;
            int32_t basePipelineIndex;
        };

    protected:
        virtual ~IGPUComputePipeline() = default;

        bool m_allowDispatchBase = false;
};

}

#endif