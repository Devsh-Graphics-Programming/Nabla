// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_I_GPU_PIPELINE_LAYOUT_H_INCLUDED_
#define _NBL_VIDEO_I_GPU_PIPELINE_LAYOUT_H_INCLUDED_


#include "nbl/core/IReferenceCounted.h"

#include "nbl/asset/IPipelineLayout.h"

#include "nbl/video/IGPUDescriptorSetLayout.h"


namespace nbl::video
{

//! GPU Version of Pipeline Layout
/*
    @see IPipelineLayout
*/

class IGPUPipelineLayout : public IBackendObject, public asset::IPipelineLayout<IGPUDescriptorSetLayout>
{
        using base_t = asset::IPipelineLayout<IGPUDescriptorSetLayout>;

    public:
        IGPUPipelineLayout(
            core::smart_refctd_ptr<const ILogicalDevice>&& dev,
            const asset::SPushConstantRange* const _pcRangesBegin = nullptr, const asset::SPushConstantRange* const _pcRangesEnd = nullptr,
            core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout0 = nullptr, core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout1 = nullptr,
            core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout2 = nullptr, core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout3 = nullptr
        ) : IBackendObject(std::move(dev)), base_t(_pcRangesBegin,_pcRangesEnd,std::move(_layout0),std::move(_layout1),std::move(_layout2),std::move(_layout3)) {}

        using asset_t = asset::ICPUPipelineLayout;
        //using patchable_params_t = TODO none;

    protected:
        virtual ~IGPUPipelineLayout() = default;
};

}

#endif