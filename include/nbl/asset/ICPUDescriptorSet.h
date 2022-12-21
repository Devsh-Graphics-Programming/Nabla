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

class NBL_API ICPUDescriptorSet final : public IDescriptorSet<ICPUDescriptorSetLayout>, public IAsset
{
	using base_t = IDescriptorSet<ICPUDescriptorSetLayout>;

public:
	//! Contructor preallocating memory for SDescriptorInfos which user can fill later (using non-const getDescriptorInfos()).
	//! @see getDescriptorInfos()
	ICPUDescriptorSet(core::smart_refctd_ptr<ICPUDescriptorSetLayout>&& _layout) : base_t(std::move(_layout)), IAsset()
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

	_NBL_STATIC_INLINE_CONSTEXPR auto AssetType = ET_DESCRIPTOR_SET;
	inline E_TYPE getAssetType() const override { return AssetType; }

	inline ICPUDescriptorSetLayout* getLayout() 
	{
		assert(!isImmutable_debug());
		return m_layout.get();
	}

	inline const ICPUDescriptorSetLayout* getLayout() const { return m_layout.get(); }

	inline bool canBeRestoredFrom(const IAsset* _other) const override
	{
		auto* other = static_cast<const ICPUDescriptorSet*>(_other);
		return m_layout->canBeRestoredFrom(other->m_layout.get());
	}

	inline size_t conservativeSizeEstimate() const override
	{
		assert(!"Invalid code path.");
		return 0xdeadbeefull;
	}

	inline core::SRange<SDescriptorInfo> getDescriptorInfoStorage(const IDescriptor::E_TYPE type) const
	{
		if (!m_descriptorInfos[static_cast<uint32_t>(type)])
			return { nullptr, nullptr };
		else
			return { m_descriptorInfos[static_cast<uint32_t>(type)]->begin(), m_descriptorInfos[static_cast<uint32_t>(type)]->end() };
	}

	core::SRange<SDescriptorInfo> getDescriptorInfos(const ICPUDescriptorSetLayout::CBindingRedirect::binding_number_t binding, IDescriptor::E_TYPE type = IDescriptor::E_TYPE::ET_COUNT);

	core::smart_refctd_ptr<IAsset> clone(uint32_t _depth = ~0u) const override;

	void convertToDummyObject(uint32_t referenceLevelsBelowToConvert = 0u) override;

protected:
	void restoreFromDummy_impl(IAsset* _other, uint32_t _levelsBelow) override;

	bool isAnyDependencyDummy_impl(uint32_t _levelsBelow) const override;

	virtual ~ICPUDescriptorSet() = default;

private:
	static inline IDescriptor::E_CATEGORY getCategoryFromType(const IDescriptor::E_TYPE type)
	{
		auto category = IDescriptor::E_CATEGORY::EC_COUNT;
		switch (type)
		{
		case IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER: [[fallthrough]];
		case IDescriptor::E_TYPE::ET_STORAGE_IMAGE: [[fallthrough]];
		case IDescriptor::E_TYPE::ET_INPUT_ATTACHMENT:
			category = IDescriptor::E_CATEGORY::EC_IMAGE;
			break;

		case IDescriptor::E_TYPE::ET_UNIFORM_BUFFER: [[fallthrough]];
		case IDescriptor::E_TYPE::ET_UNIFORM_BUFFER_DYNAMIC: [[fallthrough]];
		case IDescriptor::E_TYPE::ET_STORAGE_BUFFER: [[fallthrough]];
		case IDescriptor::E_TYPE::ET_STORAGE_BUFFER_DYNAMIC:
			category = IDescriptor::E_CATEGORY::EC_BUFFER;
			break;

		case IDescriptor::E_TYPE::ET_UNIFORM_TEXEL_BUFFER:
		case IDescriptor::E_TYPE::ET_STORAGE_TEXEL_BUFFER:
			category = IDescriptor::E_CATEGORY::EC_BUFFER_VIEW;
			break;

		case IDescriptor::E_TYPE::ET_ACCELERATION_STRUCTURE:
			category = IDescriptor::E_CATEGORY::EC_ACCELERATION_STRUCTURE;
			break;

		default:
			assert(!"Invalid code path.");
		}
		return category;
	}

	// TODO(achal): Remove.
	void allocateDescriptors() override { assert(!"Invalid code path."); }

	core::smart_refctd_dynamic_array<ICPUDescriptorSet::SDescriptorInfo> m_descriptorInfos[static_cast<uint32_t>(IDescriptor::E_TYPE::ET_COUNT)];
};

}

#endif