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

        [[deprecated("Use contructor with std::span!")]] ICPUDescriptorSetLayout(const SBinding* const _begin, const SBinding* const _end) : base_t({_begin,_end}) {}
        ICPUDescriptorSetLayout(const std::span<const SBinding> bindings) : base_t(bindings) {}

        core::smart_refctd_ptr<IAsset> clone(uint32_t _depth = ~0u) const override
        {
            auto cp = core::make_smart_refctd_ptr<ICPUDescriptorSetLayout>(nullptr,nullptr);

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

        constexpr static inline auto AssetType = ET_DESCRIPTOR_SET_LAYOUT;
        inline E_TYPE getAssetType() const override { return AssetType; }

        core::unordered_set<const IAsset*> computeDependants() const override
        {
            return computeDependantsImpl(this);
        }

        core::unordered_set<IAsset*> computeDependants() override
        {
            return computeDependantsImpl(this);
        }

	protected:
		virtual ~ICPUDescriptorSetLayout() = default;

      
  private:
      template <typename Self>
        requires(std::same_as<std::remove_cv_t<Self>, ICPUDescriptorSetLayout>)
      static auto computeDependantsImpl(Self* self) {
          using asset_ptr_t = std::conditional_t<std::is_const_v<Self>, const IAsset*, IAsset*>;
          core::unordered_set<asset_ptr_t> dependants;
          if (!self->m_immutableSamplers) return dependants;
          for (const auto& sampler: *self->m_immutableSamplers)
          {
              dependants.insert(sampler.get());
          }
          return dependants;
      }

};

}
#endif