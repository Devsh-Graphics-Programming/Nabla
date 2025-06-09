// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_CPU_DESCRIPTOR_SET_H_INCLUDED_
#define _NBL_ASSET_I_CPU_DESCRIPTOR_SET_H_INCLUDED_

#include "nbl/asset/IAsset.h"
#include "nbl/asset/ICPUBufferView.h"
#include "nbl/asset/ICPUImageView.h"
#include "nbl/asset/ICPUSampler.h"
#include "nbl/asset/ICPUDescriptorSetLayout.h"
#include "nbl/asset/IDescriptorSet.h"

namespace nbl::asset
{

//! CPU Version of Descriptor Set
/*
	DescriptorSet itself is a collection of resources conforming to
	the template given by DescriptorSetLayout and it has to have the
	exact same number and type of resources as specified by the Layout.
	Descriptor Sets do not provide the vertex shader inputs, or fragment
	shader outputs (or subpass inputs).
	@see IDescriptorSet
*/

class NBL_API2 ICPUDescriptorSet final : public IDescriptorSet<ICPUDescriptorSetLayout>, public IAsset
{
		using base_t = IDescriptorSet<ICPUDescriptorSetLayout>;

	public:
		//! Contructor preallocating memory for SDescriptorInfos which user can fill later (using non-const getDescriptorInfos()).
		//! @see getDescriptorInfos()
		inline ICPUDescriptorSet(core::smart_refctd_ptr<ICPUDescriptorSetLayout>&& _layout) : base_t(std::move(_layout)), IAsset()
		{
			for (uint32_t t = 0u; t < static_cast<uint32_t>(IDescriptor::E_TYPE::ET_COUNT); ++t)
			{
				const auto type = static_cast<IDescriptor::E_TYPE>(t);
				const uint32_t count = m_layout->getTotalDescriptorCount(type);
				if (count == 0u)
					continue;

				m_descriptorInfos[t] = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ICPUDescriptorSet::SDescriptorInfo>>(count);
			}
		}

		constexpr static inline auto AssetType = ET_DESCRIPTOR_SET;
		inline E_TYPE getAssetType() const override {return AssetType;}

		//
		inline ICPUDescriptorSetLayout* getLayout() 
		{
			assert(isMutable());
			return m_layout.get();
		}
		inline const ICPUDescriptorSetLayout* getLayout() const { return m_layout.get(); }

		inline std::span<SDescriptorInfo> getDescriptorInfoStorage(const IDescriptor::E_TYPE type) const
		{
			// TODO: @Hazardu
			// Cannot do the mutability check here because it requires the function to be non-const, but the function cannot be non-const because it's called
			// from const functions in the asset converter.
			// Relevant comments/conversations:
			// https://github.com/Devsh-Graphics-Programming/Nabla/pull/345#discussion_r1054258384
			// https://github.com/Devsh-Graphics-Programming/Nabla/pull/345#discussion_r1056289599
			// 
			// assert(isMutable());
			if (!m_descriptorInfos[static_cast<uint32_t>(type)])
				return { };
			else
				return { m_descriptorInfos[static_cast<uint32_t>(type)]->begin(), m_descriptorInfos[static_cast<uint32_t>(type)]->end() };
		}

		std::span<SDescriptorInfo> getDescriptorInfos(const ICPUDescriptorSetLayout::CBindingRedirect::binding_number_t binding, IDescriptor::E_TYPE type = IDescriptor::E_TYPE::ET_COUNT);

		std::span<const SDescriptorInfo> getDescriptorInfos(const ICPUDescriptorSetLayout::CBindingRedirect::binding_number_t binding, IDescriptor::E_TYPE type = IDescriptor::E_TYPE::ET_COUNT) const;

		core::smart_refctd_ptr<IAsset> clone(uint32_t _depth = ~0u) const override;

		core::unordered_set<const IAsset*> computeDependants() const override;
		core::unordered_set<IAsset*> computeDependants() override;

	protected:
		virtual ~ICPUDescriptorSet() = default;


	private:

		core::smart_refctd_dynamic_array<ICPUDescriptorSet::SDescriptorInfo> m_descriptorInfos[static_cast<uint32_t>(IDescriptor::E_TYPE::ET_COUNT)];

    template <typename Self>
      requires(std::same_as<std::remove_cv_t<Self>, ICPUDescriptorSet>)
    static auto computeDependantsImpl(Self* self) {
        using asset_ptr_t = std::conditional_t<std::is_const_v<Self>, const IAsset*, IAsset*>;

        using cpu_buffer_ptr_t = std::conditional_t<std::is_const_v<Self>, const ICPUBuffer*, ICPUBuffer*>;
        using cpu_sampler_ptr_t = std::conditional_t<std::is_const_v<Self>, const ICPUSampler*, ICPUSampler*>;
        using cpu_image_view_ptr_t = std::conditional_t<std::is_const_v<Self>, const ICPUImageView*, ICPUImageView*>;
        using cpu_buffer_view_ptr_t = std::conditional_t<std::is_const_v<Self>, const ICPUBufferView*, ICPUBufferView*>;
        using cpu_tlas_ptr_t = std::conditional_t<std::is_const_v<Self>, const ICPUTopLevelAccelerationStructure*, ICPUTopLevelAccelerationStructure*>;

        core::unordered_set<asset_ptr_t> dependants = { self->m_layout.get() };
        for (auto i = 0u; i < static_cast<uint32_t>(IDescriptor::E_TYPE::ET_COUNT); i++)
        {
          if (!self->m_descriptorInfos[i]) continue;
          const auto size = self->m_descriptorInfos[i]->size();
          for (auto desc_i = 0u; desc_i < size; desc_i++)
          {
            auto* desc = self->m_descriptorInfos[i]->operator[](desc_i).desc.get();
            if (!desc) continue;
            switch (IDescriptor::GetTypeCategory(static_cast<IDescriptor::E_TYPE>(i)))
            {
            case IDescriptor::EC_BUFFER:
              dependants.insert(static_cast<cpu_buffer_ptr_t>(desc));
            case IDescriptor::EC_SAMPLER:
              dependants.insert(static_cast<cpu_sampler_ptr_t>(desc));
            case IDescriptor::EC_IMAGE:
              dependants.insert(static_cast<cpu_image_view_ptr_t>(desc));
            case IDescriptor::EC_BUFFER_VIEW:
              dependants.insert(static_cast<cpu_buffer_view_ptr_t>(desc));
            case IDescriptor::EC_ACCELERATION_STRUCTURE:
              dependants.insert(static_cast<cpu_tlas_ptr_t>(desc));
            default:
              break;
            }
          }
        }
        return dependants;
    }
};

}

#endif