// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_I_GPU_DESCRIPTOR_SET_LAYOUT_H_INCLUDED__
#define __NBL_VIDEO_I_GPU_DESCRIPTOR_SET_LAYOUT_H_INCLUDED__


#include "nbl/asset/IDescriptorSetLayout.h"

#include "nbl/video/decl/IBackendObject.h"
#include "nbl/video/IGPUSampler.h"


namespace nbl::video
{

//! GPU Version of Descriptor Set Layout
/*
    @see IDescriptorSetLayout
*/

class NBL_API IGPUDescriptorSetLayout : public asset::IDescriptorSetLayout<IGPUSampler>, public IBackendObject
{
        using base_t = asset::IDescriptorSetLayout<IGPUSampler>;

    public:
        IGPUDescriptorSetLayout(core::smart_refctd_ptr<const ILogicalDevice>&& dev, const SBinding* const _begin, const SBinding* const _end)
            : base_t(_begin, _end), IBackendObject(std::move(dev))
        {}

        // TODO(achal): This can be moved to IDescriptorSetLayout, if ICPUDescriptorSetLayout also want this.
        inline uint32_t getDescriptorOffsetForBinding(const asset::E_DESCRIPTOR_TYPE type, const uint32_t binding) const
        {
            return m_redirects[type][binding];
        }

    protected:
        virtual ~IGPUDescriptorSetLayout() = default;

        bool m_isPushDescLayout = false;
        bool m_canUpdateAfterBind = false;
};

}

#endif