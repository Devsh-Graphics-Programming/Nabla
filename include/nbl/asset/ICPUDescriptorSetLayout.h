// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_CPU_DESCRIPTOR_SET_LAYOUT_H_INCLUDED_
#define _NBL_ASSET_I_CPU_DESCRIPTOR_SET_LAYOUT_H_INCLUDED_

#include "nbl/asset/IDescriptorSetLayout.h"
#include "nbl/asset/IAsset.h"
#include "nbl/asset/ICPUSampler.h"

namespace nbl::asset
{

//! CPU Version of Descriptor Set Layout
/*
    @see IDescriptorSetLayout
    @see IAsset
*/

class ICPUDescriptorSetLayout : public IDescriptorSetLayout<ICPUSampler>, public IAsset
{
    using base_t = asset::IDescriptorSetLayout<ICPUSampler>;

	public:
		_NBL_STATIC_INLINE_CONSTEXPR uint32_t IMMUTABLE_SAMPLER_HIERARCHYLEVELS_BELOW = 1u;

        ICPUDescriptorSetLayout(const SBinding* const _begin, const SBinding* const _end) : base_t({_begin,_end}) {}

        core::smart_refctd_ptr<IAsset> clone(uint32_t _depth = ~0u) const override
        {
            auto cp = core::make_smart_refctd_ptr<ICPUDescriptorSetLayout>(nullptr, nullptr);
            clone_common(cp.get());

            for (uint32_t t = 0; t < static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT); ++t)
                cp->m_descriptorRedirects[t] = m_descriptorRedirects[t].clone();
            cp->m_immutableSamplerRedirect = m_immutableSamplerRedirect.clone();
            cp->m_mutableCombinedSamplerRedirect = m_mutableCombinedSamplerRedirect.clone();

            if (m_immutableSamplers)
            {
                cp->m_immutableSamplers = core::make_refctd_dynamic_array<decltype(m_immutableSamplers)>(m_immutableSamplers->size());

                if (_depth > 0u)
                {
                    for (size_t i = 0ull; i < m_immutableSamplers->size(); ++i)
                        (*cp->m_immutableSamplers)[i] = core::smart_refctd_ptr_static_cast<ICPUSampler>((*m_immutableSamplers)[i]->clone(_depth - 1u));
                }
                else
                {
                    std::copy(m_immutableSamplers->begin(), m_immutableSamplers->end(), cp->m_immutableSamplers->begin());
                }
            }

            return cp;
        }

		size_t conservativeSizeEstimate() const override
        {
            size_t result = 0ull;
            for (uint32_t t = 0; t < static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT); ++t)
                result += m_descriptorRedirects[t].conservativeSizeEstimate();
            result += m_immutableSamplerRedirect.conservativeSizeEstimate();
            result += m_mutableCombinedSamplerRedirect.conservativeSizeEstimate();

            result += m_immutableSamplers->size() * sizeof(void*);

            return result;
        }

		void convertToDummyObject(uint32_t referenceLevelsBelowToConvert=0u) override
		{
            convertToDummyObject_common(referenceLevelsBelowToConvert);

			if (referenceLevelsBelowToConvert)
			{
                --referenceLevelsBelowToConvert;
                if (m_immutableSamplers)
                {
				    for (auto it=m_immutableSamplers->begin(); it!=m_immutableSamplers->end(); it++)
					    it->get()->convertToDummyObject(referenceLevelsBelowToConvert);
                }
			}
		}

        _NBL_STATIC_INLINE_CONSTEXPR auto AssetType = ET_DESCRIPTOR_SET_LAYOUT;
        inline E_TYPE getAssetType() const override { return AssetType; }

        bool canBeRestoredFrom(const IAsset* _other) const override
        {
            auto* other = static_cast<const ICPUDescriptorSetLayout*>(_other);
            if (getTotalBindingCount() != other->getTotalBindingCount())
                return false;
            if ((!m_immutableSamplers) != (!other->m_immutableSamplers))
                return false;
            if (m_immutableSamplers && m_immutableSamplers->size() != other->m_immutableSamplers->size())
                return false;
            if (m_immutableSamplers)
            {
                for (uint32_t i = 0u; i < m_immutableSamplers->size(); ++i)
                {
                    if (!(*m_immutableSamplers)[i]->canBeRestoredFrom((*other->m_immutableSamplers)[i].get()))
                        return false;
                }
            }

            return true;
        }

	protected:
        void restoreFromDummy_impl(IAsset* _other, uint32_t _levelsBelow) override
        {
            auto* other = static_cast<ICPUDescriptorSetLayout*>(_other);

            if (!_levelsBelow)
                return;

            --_levelsBelow;
            if (m_immutableSamplers)
            {
                for (uint32_t i = 0u; i < m_immutableSamplers->size(); ++i)
                    restoreFromDummy_impl_call((*m_immutableSamplers)[i].get(), (*other->m_immutableSamplers)[i].get(), _levelsBelow);
            }
        }

        bool isAnyDependencyDummy_impl(uint32_t _levelsBelow) const override
        {
            --_levelsBelow;
            if (m_immutableSamplers)
            {
                for (uint32_t i = 0u; i < m_immutableSamplers->size(); ++i)
                {
                    if ((*m_immutableSamplers)[i]->isAnyDependencyDummy(_levelsBelow))
                        return true;
                }
            }
            return false;
        }

		virtual ~ICPUDescriptorSetLayout() = default;
};

}
#endif