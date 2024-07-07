// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_CPU_PIPELINE_LAYOUT_H_INCLUDED_
#define _NBL_ASSET_I_CPU_PIPELINE_LAYOUT_H_INCLUDED_


#include "nbl/asset/IAsset.h"
#include "nbl/asset/ICPUDescriptorSetLayout.h"
#include "nbl/asset/IPipelineLayout.h"


namespace nbl::asset
{

//! CPU Version of Pipeline Layout
/*
    @see IPipelineLayout
*/

class ICPUPipelineLayout : public IAsset, public IPipelineLayout<ICPUDescriptorSetLayout>
{
	public:
        static inline constexpr uint32_t DESC_SET_LAYOUT_HIERARCHYLEVELS_BELOW = 1u;
        static inline constexpr uint32_t IMMUTABLE_SAMPLER_HIERARCHYLEVELS_BELOW = 1u+ICPUDescriptorSetLayout::IMMUTABLE_SAMPLER_HIERARCHYLEVELS_BELOW;

        inline ICPUPipelineLayout(
            const std::span<const asset::SPushConstantRange> _pcRanges,
            core::smart_refctd_ptr<ICPUDescriptorSetLayout>&& _layout0, core::smart_refctd_ptr<ICPUDescriptorSetLayout>&& _layout1,
            core::smart_refctd_ptr<ICPUDescriptorSetLayout>&& _layout2, core::smart_refctd_ptr<ICPUDescriptorSetLayout>&& _layout3
        ) : IPipelineLayout<ICPUDescriptorSetLayout>(_pcRanges,std::move(_layout0),std::move(_layout1),std::move(_layout2),std::move(_layout3)) {}

		ICPUDescriptorSetLayout* getDescriptorSetLayout(uint32_t _set) 
        {
            assert(isMutable());
            return m_descSetLayouts[_set].get(); 
        }
		const ICPUDescriptorSetLayout* getDescriptorSetLayout(uint32_t _set) const { return m_descSetLayouts[_set].get(); }

        void setDescriptorSetLayout(uint32_t _set, core::smart_refctd_ptr<ICPUDescriptorSetLayout>&& _dslayout) 
        { 
            assert(isMutable());
            assert(_set < DESCRIPTOR_SET_COUNT);
            m_descSetLayouts[_set] = std::move(_dslayout);
        }

        void setPushConstantRanges(core::smart_refctd_dynamic_array<SPushConstantRange>&& _ranges)
        {
            assert(isMutable());
            m_pushConstantRanges = std::move(_ranges);
        }

        core::smart_refctd_ptr<IAsset> clone(uint32_t _depth = ~0u) const override
        {
            std::array<core::smart_refctd_ptr<ICPUDescriptorSetLayout>,DESCRIPTOR_SET_COUNT> dsLayouts;
            for (size_t i = 0ull; i < dsLayouts.size(); ++i)
                dsLayouts[i] = (m_descSetLayouts[i] && _depth > 0u) ? core::smart_refctd_ptr_static_cast<ICPUDescriptorSetLayout>(m_descSetLayouts[i]->clone(_depth-1u)) : m_descSetLayouts[i];

            return core::make_smart_refctd_ptr<ICPUPipelineLayout>(
                std::span<const asset::SPushConstantRange>{m_pushConstantRanges->begin(),m_pushConstantRanges->end()},
                std::move(dsLayouts[0]),std::move(dsLayouts[1]),std::move(dsLayouts[2]),std::move(dsLayouts[3])
            );
        }

        static inline constexpr bool HasDependents = true;

        static inline constexpr auto AssetType = ET_PIPELINE_LAYOUT;
        inline E_TYPE getAssetType() const override { return AssetType; }

    protected:
		virtual ~ICPUPipelineLayout() = default;
};

}
#endif