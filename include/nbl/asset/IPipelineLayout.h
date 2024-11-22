// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_PIPELINE_LAYOUT_H_INCLUDED_
#define _NBL_ASSET_I_PIPELINE_LAYOUT_H_INCLUDED_

#include "nbl/macros.h"
#include "nbl/core/declarations.h"

#include <algorithm>
#include <array>

#include "nbl/asset/IDescriptorSetLayout.h"
#include "nbl/builtin/hlsl/binding_info.hlsl"


namespace nbl::asset
{

//! Push Constant Ranges
/*
    Push Constants serve a similar purpose to a Uniform Buffer Object,
    however they serve as a fast path with regard to data upload from the
    CPU and data access from the GPU. 
    
    Note that Nabla limits push constant size to 128 bytes.

    Push Constants are an alternative to an UBO where it performs really poorly,
    mostly very small and very frequent updates. Examples of which are:

    - Global Object ID
    - Material/Mesh flags implemented as bits
    - Unique per DrawCall indices or bit-flags
*/

struct SPushConstantRange
{
	IShader::E_SHADER_STAGE stageFlags;
    uint32_t offset;
    uint32_t size;

    inline bool operator<(const SPushConstantRange& _rhs) const
    {
        if (stageFlags==_rhs.stageFlags)
        {
            if (offset==_rhs.offset)
            {
                return size<_rhs.size;
            }
            return offset<_rhs.offset;
        }
        return stageFlags<_rhs.stageFlags;
    }
    inline bool operator==(const SPushConstantRange& _rhs) const
    {
        if (stageFlags != _rhs.stageFlags)
            return false;
        if (offset != _rhs.offset)
            return false;
        return (size == _rhs.size);
    }
    inline bool operator!=(const SPushConstantRange& _rhs) const
    {
        return !((*this)==_rhs);
    }

    inline bool overlap(const SPushConstantRange& _other) const
    {
        const int32_t end1 = offset + size;
        const int32_t end2 = _other.offset + _other.size;

        return (std::min<int32_t>(end1, end2) - std::max<int32_t>(offset, _other.offset)) > 0;
    }
};

//! Interface class for pipeline layouts
/*
    Pipeline layout stores all the state like bindings and set numbers 
    of descriptors as well as the descriptor types common to multiple
    draw calls (meshes) as an aggregate. It exists because the same
    object exists in the Vulkan API. 
    
    Pipeline Layout specifies all 4 templates of resource types 
    ( a null descriptor layout is an empty template) that will be 
    used by all the shaders used in the draw or compute dispatch.
*/

template<typename DescLayoutType>
class IPipelineLayout
{
    public:
        static inline constexpr uint32_t DESCRIPTOR_SET_COUNT = 4u;

        std::span<const DescLayoutType* const,DESCRIPTOR_SET_COUNT> getDescriptorSetLayouts() const
        {
            return std::span<const DescLayoutType* const,DESCRIPTOR_SET_COUNT>(&m_descSetLayouts[0].get(),DESCRIPTOR_SET_COUNT);
        }
        [[deprecated]] const DescLayoutType* getDescriptorSetLayout(uint32_t _set) const { return getDescriptorSetLayouts()[_set]; }

        core::SRange<const SPushConstantRange> getPushConstantRanges() const 
        {
            if (m_pushConstantRanges)
                return {m_pushConstantRanges->data(), m_pushConstantRanges->data()+m_pushConstantRanges->size()};
            else 
                return {nullptr, nullptr};
        }

        bool isCompatibleForPushConstants(const IPipelineLayout<DescLayoutType>* _other) const
        {
            if (getPushConstantRanges().size() != _other->getPushConstantRanges().size())
                return false;

            const size_t cnt = getPushConstantRanges().size();
            const SPushConstantRange* lhs = getPushConstantRanges().begin();
            const SPushConstantRange* rhs = _other->getPushConstantRanges().begin();
            for (size_t i = 0ull; i < cnt; ++i)
                if (lhs[i] != rhs[i])
                    return false;

            return true;
        }

        //! Checks if `this` and `_other` are compatible for set `_setNum`. See https://www.khronos.org/registry/vulkan/specs/1.1-extensions/html/vkspec.html#descriptorsets-compatibility for compatiblity rules.
        /**
        @returns Max value of `_setNum` for which the two pipeline layouts are compatible or -1 if they're not compatible at all.
        */
        int32_t isCompatibleUpToSet(const uint32_t _setNum, const IPipelineLayout<DescLayoutType>* _other) const
        {
            if (!_setNum || (_setNum >= DESCRIPTOR_SET_COUNT)) //vulkan would also care about push constant ranges compatibility here
                return -1;

		    uint32_t i = 0u;
            for (; i <=_setNum; i++)
            {
                const DescLayoutType* lhs = m_descSetLayouts[i].get();
                const DescLayoutType* rhs = _other->getDescriptorSetLayout(i);

                const bool compatible = (lhs == rhs) || (lhs && lhs->isIdenticallyDefined(rhs));
			    if (!compatible)
				    break;
            }
            return static_cast<int32_t>(i)-1;
        }

        // utility function, if you compile shaders for specific layouts, not create layouts given shaders
        struct SBindingKey
        {
            using type_bitset_t = std::bitset<static_cast<size_t>(IDescriptor::E_TYPE::ET_COUNT)>;

            hlsl::SBindingInfo binding = {};
            core::bitflag<IShader::E_SHADER_STAGE> requiredStages = IShader::E_SHADER_STAGE::ESS_UNKNOWN;
            // could have just initialized with `~type_bitset_t()` in C++23
            type_bitset_t allowedTypes = type_bitset_t((0x1u<<static_cast<size_t>(IDescriptor::E_TYPE::ET_COUNT))-1);
        };
        // TODO: add constraints for stage and creation flags, or just return the storage index & redirect?
        core::string getBindingInfoForHLSL(const SBindingKey& key) const
        {
            if (key.binding.set>=DESCRIPTOR_SET_COUNT)
                return "#error \"IPipelineLayout::SBindingKey::binding::set out of range!\"";
            const auto* layout = m_descSetLayouts[key.binding.set].get();
            if (!layout)
                return "#error \"IPipelineLayout::SBindingKey::binding::set layout is nullptr!\"";
            //
            using redirect_t = IDescriptorSetLayoutBase::CBindingRedirect;
            using storage_range_index_t = redirect_t::storage_range_index_t;
            const redirect_t* redirect;
            storage_range_index_t found;
            {
                const redirect_t::binding_number_t binding(key.binding.binding);
                for (auto t=0u; t<static_cast<size_t>(IDescriptor::E_TYPE::ET_COUNT); t++)
                if (key.allowedTypes.test(t))
                {
                    redirect = &layout->getDescriptorRedirect(static_cast<IDescriptor::E_TYPE>(t));
                    found = redirect->findBindingStorageIndex(binding);
                    if (found)
                        break;
                }
                if (!found && key.allowedTypes.test(static_cast<size_t>(IDescriptor::E_TYPE::ET_SAMPLER)))
                {
                    redirect = &layout->getImmutableSamplerRedirect();
                    found = redirect->findBindingStorageIndex(binding);
                }
                if (!found)
                    return "#error \"Could not find `IPipelineLayout::SBindingKey::binding::binding` in `IPipelineLayout::SBindingKey::binding::set`'s layout!\"";
            }
            if (!redirect->getStageFlags(found).hasFlags(key.requiredStages))
                return "#error \"Binding found in the layout doesn't have all the `IPipelineLayout::SBindingKey::binding::requiredStages` flags!\"";
            const auto count = redirect->getCount(found);
            assert(count); // this layout should have never passed validation
            return "::nbl::hlsl::ConstevalBindingInfo<"+std::to_string(key.binding.set)+","+std::to_string(key.binding.binding)+","+std::to_string(count)+">";
        }

    protected:
        IPipelineLayout(
            const std::span<const asset::SPushConstantRange> _pcRanges,
            core::smart_refctd_ptr<DescLayoutType>&& _layout0, core::smart_refctd_ptr<DescLayoutType>&& _layout1,
            core::smart_refctd_ptr<DescLayoutType>&& _layout2, core::smart_refctd_ptr<DescLayoutType>&& _layout3
        ) : m_descSetLayouts{{std::move(_layout0), std::move(_layout1), std::move(_layout2), std::move(_layout3)}}
        {
            if (!_pcRanges.empty())
                m_pushConstantRanges = core::make_refctd_dynamic_array<decltype(m_pushConstantRanges)>(_pcRanges);
        }
        virtual ~IPipelineLayout() = default;

        std::array<core::smart_refctd_ptr<DescLayoutType>,DESCRIPTOR_SET_COUNT> m_descSetLayouts;
        core::smart_refctd_dynamic_array<SPushConstantRange> m_pushConstantRanges;
};

}
#endif