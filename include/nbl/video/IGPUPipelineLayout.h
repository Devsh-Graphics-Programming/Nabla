// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_I_GPU_PIPELINE_LAYOUT_H_INCLUDED__
#define __NBL_VIDEO_I_GPU_PIPELINE_LAYOUT_H_INCLUDED__


#include "nbl/core/IReferenceCounted.h"

#include "nbl/asset/IPipelineLayout.h"

#include "nbl/video/IGPUDescriptorSetLayout.h"


namespace nbl::video
{

//! GPU Version of Pipeline Layout
/*
    @see IPipelineLayout
*/

class IGPUPipelineLayout : public core::IReferenceCounted, public asset::IPipelineLayout<IGPUDescriptorSetLayout>, public IBackendObject
{
        using base_t = asset::IPipelineLayout<IGPUDescriptorSetLayout>;

    public:
        IGPUPipelineLayout(
            core::smart_refctd_ptr<const ILogicalDevice>&& dev,
            const asset::SPushConstantRange* const _pcRangesBegin = nullptr, const asset::SPushConstantRange* const _pcRangesEnd = nullptr,
            core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout0 = nullptr, core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout1 = nullptr,
            core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout2 = nullptr, core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout3 = nullptr
        ) : base_t(_pcRangesBegin, _pcRangesEnd, std::move(_layout0), std::move(_layout1), std::move(_layout2), std::move(_layout3)), IBackendObject(std::move(dev))
        {

        }

    protected:
        virtual ~IGPUPipelineLayout() = default;
};

}

#endif