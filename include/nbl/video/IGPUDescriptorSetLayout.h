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
        IGPUDescriptorSetLayout(core::smart_refctd_ptr<const ILogicalDevice>&& dev, const SBinding* const _begin, const SBinding* const _end) : base_t(_begin, _end), IBackendObject(std::move(dev))
        {
            uint32_t localDescriptorOffset = 0u;
            for (auto b = _begin; b != _end; ++b)
            {
                m_bindingToDescriptorOffsetMap.insert({ b->binding, localDescriptorOffset });
                localDescriptorOffset += b->count;
            }
        }

        inline uint32_t getDescriptorOffsetForBinding(const uint32_t binding) const
        {
            auto found = m_bindingToDescriptorOffsetMap.find(binding);
            if (found != m_bindingToDescriptorOffsetMap.end())
                return found->second;

            return ~0u;
        }

    protected:
        virtual ~IGPUDescriptorSetLayout() = default;

        bool m_isPushDescLayout = false;
        bool m_canUpdateAfterBind = false;

    private:
        // This maps the descriptor set layout's binding number to the LOCAL offset in the array of descriptors where the descriptors of this binding number start.
        // Therefore, a given binding number (say, b) having n array elements, will have descriptors stored at the local offset of [m_bindingToDescriptorOffsetMap[b], m_bindingToDescriptorOffsetMap[b]+n).
        // 
        // TODO(achal): Can I use a flat array here? It can be done if we enforce any upper limit on VkDescriptorSetLayoutBinding::binding.
        core::unordered_map<uint32_t, uint32_t> m_bindingToDescriptorOffsetMap;

};

}

#endif