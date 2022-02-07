// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_I_GPU_DESCRIPTOR_SET_LAYOUT_H_INCLUDED__
#define __NBL_VIDEO_I_GPU_DESCRIPTOR_SET_LAYOUT_H_INCLUDED__

#include "nbl/asset/IDescriptorSetLayout.h"

#include "nbl/video/IGPUSampler.h"

namespace nbl
{
namespace video
{
//! GPU Version of Descriptor Set Layout
/*
    @see IDescriptorSetLayout
*/

class IGPUDescriptorSetLayout : public asset::IDescriptorSetLayout<IGPUSampler>
{
public:
    using IDescriptorSetLayout<IGPUSampler>::IDescriptorSetLayout;

protected:
    virtual ~IGPUDescriptorSetLayout() = default;

    bool m_isPushDescLayout = false;
    bool m_canUpdateAfterBind = false;
};

}
}

#endif