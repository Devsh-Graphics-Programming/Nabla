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
            assert(!isImmutable_debug());
            return m_descSetLayouts[_set].get(); 
        }
		const ICPUDescriptorSetLayout* getDescriptorSetLayout(uint32_t _set) const { return m_descSetLayouts[_set].get(); }

        void setDescriptorSetLayout(uint32_t _set, core::smart_refctd_ptr<ICPUDescriptorSetLayout>&& _dslayout) 
        { 
            assert(!isImmutable_debug());
            assert(_set < DESCRIPTOR_SET_COUNT);
            m_descSetLayouts[_set] = std::move(_dslayout);
        }

        void setPushConstantRanges(core::smart_refctd_dynamic_array<SPushConstantRange>&& _ranges)
        {
            assert(!isImmutable_debug());
            m_pushConstantRanges = std::move(_ranges);
        }

        core::smart_refctd_ptr<IAsset> clone(uint32_t _depth = ~0u) const override
        {
            std::array<core::smart_refctd_ptr<ICPUDescriptorSetLayout>,DESCRIPTOR_SET_COUNT> dsLayouts;
            for (size_t i = 0ull; i < dsLayouts.size(); ++i)
                dsLayouts[i] = (m_descSetLayouts[i] && _depth > 0u) ? core::smart_refctd_ptr_static_cast<ICPUDescriptorSetLayout>(m_descSetLayouts[i]->clone(_depth-1u)) : m_descSetLayouts[i];

            auto cp = core::make_smart_refctd_ptr<ICPUPipelineLayout>(
                std::span<const asset::SPushConstantRange>{m_pushConstantRanges->begin(),m_pushConstantRanges->end()},
                std::move(dsLayouts[0]),std::move(dsLayouts[1]),std::move(dsLayouts[2]),std::move(dsLayouts[3])
            );
            clone_common(cp.get());

            return cp;
        }

		size_t conservativeSizeEstimate() const override { return m_descSetLayouts.size()*sizeof(void*)+m_pushConstantRanges->size()*sizeof(SPushConstantRange); }
		void convertToDummyObject(uint32_t referenceLevelsBelowToConvert=0u) override
		{
            convertToDummyObject_common(referenceLevelsBelowToConvert);

			if (referenceLevelsBelowToConvert)
			    for (auto it=m_descSetLayouts.begin(); it!=m_descSetLayouts.end(); it++)
			        if (it->get())
				        it->get()->convertToDummyObject(referenceLevelsBelowToConvert-1u);

            if (canBeConvertedToDummy())
			    m_pushConstantRanges = nullptr;
		}

        static inline constexpr auto AssetType = ET_PIPELINE_LAYOUT;
        inline E_TYPE getAssetType() const override { return AssetType; }

        bool canBeRestoredFrom(const IAsset* _other) const override
        {
            auto* other = static_cast<const ICPUPipelineLayout*>(_other);

            if ((!m_pushConstantRanges) != (!other->m_pushConstantRanges))
                return false;
            if (m_pushConstantRanges && m_pushConstantRanges->size() != other->m_pushConstantRanges->size())
                return false;
            for (uint32_t i = 0u; i < m_pushConstantRanges->size(); ++i)
                if ((*m_pushConstantRanges)[i] != (*other->m_pushConstantRanges)[i])
                    return false;

            for (uint32_t i = 0u; i < DESCRIPTOR_SET_COUNT; ++i)
            {
                if ((!m_descSetLayouts[i]) != (!other->m_descSetLayouts[i]))
                    return false;
                if (!m_descSetLayouts[i]->canBeRestoredFrom(other->m_descSetLayouts[i].get()))
                    return false;
            }
            return true;
        }

    protected:
        void restoreFromDummy_impl(IAsset* _other, uint32_t _levelsBelow) override
        {
            auto* other = static_cast<ICPUPipelineLayout*>(_other);

            const bool restorable = willBeRestoredFrom(_other);

            if (restorable)
                std::swap(m_pushConstantRanges, other->m_pushConstantRanges);
            if (_levelsBelow)
            {
                --_levelsBelow;
                for (uint32_t i = 0u; i < m_descSetLayouts.size(); ++i)
                    restoreFromDummy_impl_call(m_descSetLayouts[i].get(), other->m_descSetLayouts[i].get(), _levelsBelow);
            }
        }

        bool isAnyDependencyDummy_impl(uint32_t _levelsBelow) const override
        {
            --_levelsBelow;
            for (auto& dsl : m_descSetLayouts)
                if (dsl && dsl->isAnyDependencyDummy(_levelsBelow))
                    return true;
            return false;
        }

		virtual ~ICPUPipelineLayout() = default;
};

}
#endif