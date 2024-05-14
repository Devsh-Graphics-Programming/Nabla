// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_I_GPU_DESCRIPTOR_SET_LAYOUT_H_INCLUDED_
#define _NBL_VIDEO_I_GPU_DESCRIPTOR_SET_LAYOUT_H_INCLUDED_


#include "nbl/asset/IDescriptorSetLayout.h"

#include "nbl/video/decl/IBackendObject.h"
#include "nbl/video/IGPUSampler.h"


namespace nbl::video
{

//! GPU Version of Descriptor Set Layout
/*
    @see IDescriptorSetLayout
*/

class IGPUDescriptorSetLayout : public asset::IDescriptorSetLayout<IGPUSampler>, public IBackendObject
{
        using base_t = asset::IDescriptorSetLayout<IGPUSampler>;

    public:
        inline IGPUDescriptorSetLayout(core::smart_refctd_ptr<const ILogicalDevice>&& dev, const std::span<const SBinding> _bindings) : base_t(_bindings), IBackendObject(std::move(dev))
        {
            for (const auto& binding : _bindings)
            {
                if (binding.createFlags.hasFlags(SBinding::E_CREATE_FLAGS::ECF_UPDATE_AFTER_BIND_BIT) || binding.createFlags.hasFlags(SBinding::E_CREATE_FLAGS::ECF_UPDATE_UNUSED_WHILE_PENDING_BIT))
                {
                    m_canUpdateAfterBind = true;
                    break;
                }
            }
        }

        inline bool canUpdateAfterBind() const { return m_canUpdateAfterBind; }

    protected:
        virtual ~IGPUDescriptorSetLayout() = default;

        bool m_isPushDescLayout = false;
        bool m_canUpdateAfterBind = false;
};

}

#endif