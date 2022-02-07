// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_I_GPU_PIPELINE_LAYOUT_H_INCLUDED__
#define __NBL_VIDEO_I_GPU_PIPELINE_LAYOUT_H_INCLUDED__

#include "nbl/core/IReferenceCounted.h"
#include "nbl/asset/IPipelineLayout.h"
#include "nbl/video/IGPUDescriptorSetLayout.h"

namespace nbl
{
namespace video
{
//! GPU Version of Pipeline Layout
/*
    @see IPipelineLayout
*/

class IGPUPipelineLayout : public core::IReferenceCounted, public asset::IPipelineLayout<IGPUDescriptorSetLayout>
{
public:
    using asset::IPipelineLayout<IGPUDescriptorSetLayout>::IPipelineLayout;

protected:
    virtual ~IGPUPipelineLayout() = default;
};

}
}

#endif