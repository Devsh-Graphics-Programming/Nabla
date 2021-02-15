// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_CPU_PIPELINE_LAYOUT_H_INCLUDED__
#define __NBL_ASSET_I_CPU_PIPELINE_LAYOUT_H_INCLUDED__

#include "nbl/asset/IAsset.h"
#include "nbl/asset/ICPUDescriptorSetLayout.h"
#include "nbl/asset/IPipelineLayout.h"

namespace nbl
{
namespace asset
{

//! CPU Version of Pipeline Layout
/*
    @see IPipelineLayout
*/

class ICPUPipelineLayout : public IAsset, public IPipelineLayout<ICPUDescriptorSetLayout>
{
	public:
		_NBL_STATIC_INLINE_CONSTEXPR uint32_t DESC_SET_LAYOUT_HIERARCHYLEVELS_BELOW = 1u;
		_NBL_STATIC_INLINE_CONSTEXPR uint32_t IMMUTABLE_SAMPLER_HIERARCHYLEVELS_BELOW = 1u+ICPUDescriptorSetLayout::IMMUTABLE_SAMPLER_HIERARCHYLEVELS_BELOW;

		using IPipelineLayout<ICPUDescriptorSetLayout>::IPipelineLayout;

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
            std::array<core::smart_refctd_ptr<ICPUDescriptorSetLayout>, DESCRIPTOR_SET_COUNT> dsLayouts;
            for (size_t i = 0ull; i < dsLayouts.size(); ++i)
                dsLayouts[i] = (m_descSetLayouts[i] && _depth > 0u) ? core::smart_refctd_ptr_static_cast<ICPUDescriptorSetLayout>(m_descSetLayouts[i]->clone(_depth-1u)) : m_descSetLayouts[i];

            auto cp = core::make_smart_refctd_ptr<ICPUPipelineLayout>(
                nullptr, nullptr, 
                std::move(dsLayouts[0]), std::move(dsLayouts[1]), std::move(dsLayouts[2]), std::move(dsLayouts[3])
            );
            clone_common(cp.get());
            cp->m_pushConstantRanges = m_pushConstantRanges;

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

        _NBL_STATIC_INLINE_CONSTEXPR auto AssetType = ET_PIPELINE_LAYOUT;
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
}

#endif