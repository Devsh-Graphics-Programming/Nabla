// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_PIPELINE_LAYOUT_H_INCLUDED_
#define _NBL_ASSET_I_PIPELINE_LAYOUT_H_INCLUDED_


#include <algorithm>
#include <array>

#include "nbl/macros.h"
#include "nbl/core/declarations.h"


namespace nbl::asset
{

//! Push Constant Ranges
/*
    Push Constants serve a similar purpose to a Uniform Buffer Object,
    however they serve as a fast path with regard to data upload from the
    CPU and data access from the GPU. 
    
    Note that IrrlichtBaW limits push constant size to 128 bytes.

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

    const DescLayoutType* getDescriptorSetLayout(uint32_t _set) const { return m_descSetLayouts[_set].get(); }
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