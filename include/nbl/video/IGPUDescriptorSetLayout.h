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

        struct SBindingRedirect
        {
            SBindingRedirect() : bindings(nullptr), offsets(nullptr), count(0ull) {}

            SBindingRedirect(const size_t _count) : count(_count)
            {
                bindings = std::make_unique<uint32_t[]>(count << 1);
                offsets = bindings.get() + count;
            }

            inline uint32_t operator[](const uint32_t binding) const
            {
                assert(bindings && offsets && (count != 0ull));

                constexpr auto Invalid = ~0u;

                auto found = std::lower_bound(bindings.get(), bindings.get() + count, binding);
                if (found < bindings.get() + count)
                {
                    if (*found != binding)
                        return Invalid;

                    const uint32_t foundIdx = found - bindings.get();
                    assert(foundIdx < count);
                    return offsets[foundIdx];
                }

                return Invalid;
            }

            std::unique_ptr<uint32_t[]> bindings;
            uint32_t* offsets;
            size_t count;
        };

        // Maps a binding number to a local (to descriptor set layout) offset, for a given descriptor type.
        SBindingRedirect m_redirects[asset::EDT_COUNT];
        // TODO(achal): One for samplers

    public:
        IGPUDescriptorSetLayout(core::smart_refctd_ptr<const ILogicalDevice>&& dev, const SBinding* const _begin, const SBinding* const _end) : base_t(_begin, _end), IBackendObject(std::move(dev))
        {
            // TODO(achal): Move this common stuff to IDescriptorSetLayout
            struct SBindingRedirectBuildInfo
            {
                uint32_t binding;
                uint32_t count;

                inline bool operator< (const SBindingRedirectBuildInfo& other) const { return binding < other.binding; }
            };

            core::vector<SBindingRedirectBuildInfo> buildInfo[asset::EDT_COUNT];
            // TODO(achal): One for samplers

            for (auto b = _begin; b != _end; ++b)
            {
                buildInfo[b->type].emplace_back(b->binding, b->count);
                // TODO(achal): One for samplers
            }

            for (auto type = 0u; type < asset::EDT_COUNT; ++type)
            {
                m_redirects[type] = SBindingRedirect(buildInfo[type].size());
            }

            auto buildRedirect = [](SBindingRedirect& redirect, core::vector<SBindingRedirectBuildInfo>& info)
            {
                std::sort(info.begin(), info.end());

                for (size_t i = 0u; i < info.size(); ++i)
                {
                    redirect.bindings[i] = info[i].binding;
                    redirect.offsets[i] = info[i].count;
                }

                std::exclusive_scan(redirect.offsets, redirect.offsets + info.size(), redirect.offsets, 0u);
            };

            for (auto type = 0u; type < asset::EDT_COUNT; ++type)
                buildRedirect(m_redirects[type], buildInfo[type]);
        }

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