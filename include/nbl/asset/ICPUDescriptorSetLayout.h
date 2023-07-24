// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_CPU_DESCRIPTOR_SET_LAYOUT_H_INCLUDED__
#define __NBL_ASSET_I_CPU_DESCRIPTOR_SET_LAYOUT_H_INCLUDED__

#include "nbl/asset/IDescriptorSetLayout.h"
#include "nbl/asset/IAsset.h"
#include "nbl/asset/ICPUSampler.h"

namespace nbl
{
namespace asset
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

        ICPUDescriptorSetLayout(const SBinding* const _begin, const SBinding* const _end) : base_t(_begin, _end) {}

        core::smart_refctd_ptr<IAsset> clone(uint32_t _depth = ~0u) const override
        {
            auto cp = core::make_smart_refctd_ptr<ICPUDescriptorSetLayout>(nullptr, nullptr);
            clone_common(cp.get());

            for (uint32_t t = 0; t < static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT); ++t)
                cp->m_descriptorRedirects[t] = m_descriptorRedirects[t].clone();
            cp->m_immutableSamplerRedirect = m_immutableSamplerRedirect.clone();
            cp->m_mutableSamplerRedirect = m_mutableSamplerRedirect.clone();

            if (m_samplers)
            {
                cp->m_samplers = core::make_refctd_dynamic_array<decltype(m_samplers)>(m_samplers->size());

                if (_depth > 0u)
                {
                    for (size_t i = 0ull; i < m_samplers->size(); ++i)
                        (*cp->m_samplers)[i] = core::smart_refctd_ptr_static_cast<ICPUSampler>((*m_samplers)[i]->clone(_depth - 1u));
                }
                else
                {
                    std::copy(m_samplers->begin(), m_samplers->end(), cp->m_samplers->begin());
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
            result += m_mutableSamplerRedirect.conservativeSizeEstimate();

            result += m_samplers->size() * sizeof(void*);

            return result;
        }

        _NBL_STATIC_INLINE_CONSTEXPR auto AssetType = ET_DESCRIPTOR_SET_LAYOUT;
        inline E_TYPE getAssetType() const override { return AssetType; }

	protected:

        bool compatible(const IAsset* _other) const override
        {
            auto* other = static_cast<const ICPUDescriptorSetLayout*>(_other);
            if (getTotalBindingCount() != other->getTotalBindingCount())
                return false;
            if ((!m_samplers) != (!other->m_samplers))
                return false;
            if (m_samplers && m_samplers->size() != other->m_samplers->size())
                return false;
            return true;
        }

        nbl::core::vector<core::smart_refctd_ptr<IAsset>>getMembersToRecurse() const override
        {
            nbl::core::vector<core::smart_refctd_ptr<IAsset>> assets = {};
   
            if (m_samplers)
            {
                for (uint32_t i = 0u; i < m_samplers->size(); ++i)
                    assets.push_back((*m_samplers)[i]);
            }
            return assets;
        }

		virtual ~ICPUDescriptorSetLayout() = default;
};

}
}

#endif