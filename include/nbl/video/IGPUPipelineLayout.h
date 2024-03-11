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

class IGPUPipelineLayout : public IBackendObject, public asset::IPipelineLayout<const IGPUDescriptorSetLayout>
{
        using base_t = asset::IPipelineLayout<const IGPUDescriptorSetLayout>;

    public:
        IGPUPipelineLayout(
            core::smart_refctd_ptr<const ILogicalDevice>&& dev, const std::span<const asset::SPushConstantRange> _pcRanges,
            core::smart_refctd_ptr<const IGPUDescriptorSetLayout>&& _layout0, core::smart_refctd_ptr<const IGPUDescriptorSetLayout>&& _layout1,
            core::smart_refctd_ptr<const IGPUDescriptorSetLayout>&& _layout2, core::smart_refctd_ptr<const IGPUDescriptorSetLayout>&& _layout3
        ) : IBackendObject(std::move(dev)), base_t(_pcRanges,std::move(_layout0),std::move(_layout1),std::move(_layout2),std::move(_layout3)) {}

    protected:
        virtual ~IGPUPipelineLayout() = default;
};
}

#endif