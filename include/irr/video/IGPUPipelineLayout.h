// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_I_GPU_PIPELINE_LAYOUT_H_INCLUDED__
#define __NBL_I_GPU_PIPELINE_LAYOUT_H_INCLUDED__

#include "irr/core/IReferenceCounted.h"
#include "irr/asset/IPipelineLayout.h"
#include "irr/video/IGPUDescriptorSetLayout.h"

namespace irr {
namespace video
{

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