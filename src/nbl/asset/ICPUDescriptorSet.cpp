#include "nbl/asset/ICPUDescriptorSet.h"

namespace nbl::asset
{

std::span<ICPUDescriptorSet::SDescriptorInfo> ICPUDescriptorSet::getDescriptorInfos(const ICPUDescriptorSetLayout::CBindingRedirect::binding_number_t binding, IDescriptor::E_TYPE type)
{
	assert(isMutable());
	auto immutableResult = const_cast<const ICPUDescriptorSet*>(this)->getDescriptorInfos(binding, type);
	return {const_cast<ICPUDescriptorSet::SDescriptorInfo*>(immutableResult.data()), immutableResult.size()};
}

std::span<const ICPUDescriptorSet::SDescriptorInfo> ICPUDescriptorSet::getDescriptorInfos(const ICPUDescriptorSetLayout::CBindingRedirect::binding_number_t binding, IDescriptor::E_TYPE type) const
{
	if (type == IDescriptor::E_TYPE::ET_COUNT)
	{
		for (uint32_t t = 0; t < static_cast<uint32_t>(IDescriptor::E_TYPE::ET_COUNT); ++t)
		{
			const auto possibleType = static_cast<IDescriptor::E_TYPE>(t);
			const auto& redirect = getLayout()->getDescriptorRedirect(possibleType);
			if (redirect.findBindingStorageIndex(binding).data != redirect.Invalid)
			{
				type = possibleType;
				break;
			}
		}

		if (type == IDescriptor::E_TYPE::ET_COUNT)
			return { };
	}

	const auto& redirect = getLayout()->getDescriptorRedirect(type);
	const auto bindingNumberIndex = redirect.findBindingStorageIndex(binding);
	if (bindingNumberIndex.data == redirect.Invalid)
		return { };

	const auto offset = redirect.getStorageOffset(asset::ICPUDescriptorSetLayout::CBindingRedirect::storage_range_index_t{ bindingNumberIndex }).data;
	const auto count = redirect.getCount(asset::ICPUDescriptorSetLayout::CBindingRedirect::storage_range_index_t{ bindingNumberIndex });

	auto infosBegin = m_descriptorInfos[static_cast<uint32_t>(type)]->begin() + offset;

	return { infosBegin, count };
}

core::smart_refctd_ptr<IAsset> ICPUDescriptorSet::clone(uint32_t _depth) const
{
	auto layout = (_depth > 0u && m_layout) ? core::smart_refctd_ptr_static_cast<ICPUDescriptorSetLayout>(m_layout->clone(_depth - 1u)) : m_layout;
	auto cp = core::make_smart_refctd_ptr<ICPUDescriptorSet>(std::move(layout));

	for (uint32_t t = 0u; t < static_cast<uint32_t>(IDescriptor::E_TYPE::ET_COUNT); ++t)
	{
		const auto type = static_cast<IDescriptor::E_TYPE>(t);

		for (uint32_t i = 0u; i < m_descriptorInfos[t]->size(); ++i)
		{
			const auto& srcDescriptorInfo = m_descriptorInfos[t]->begin()[i];
			auto& dstDescriptorInfo = cp->m_descriptorInfos[t]->begin()[i];

			auto category = IDescriptor::GetTypeCategory(type);
			
			if (category == IDescriptor::E_CATEGORY::EC_IMAGE)
				dstDescriptorInfo.info.combinedImageSampler = srcDescriptorInfo.info.combinedImageSampler;
			else
				dstDescriptorInfo.info.buffer = srcDescriptorInfo.info.buffer;

			if (_depth > 0u)
			{
				// Clone the descriptor.
				{
					assert(srcDescriptorInfo.desc);

					IAsset* descriptor = nullptr;
					if (category == IDescriptor::E_CATEGORY::EC_SAMPLER)
						descriptor = static_cast<ICPUSampler*>(srcDescriptorInfo.desc.get());
					else if (category == IDescriptor::E_CATEGORY::EC_IMAGE)
						descriptor = static_cast<ICPUImageView*>(srcDescriptorInfo.desc.get());
					else if (category == IDescriptor::E_CATEGORY::EC_BUFFER_VIEW)
						descriptor = static_cast<ICPUBufferView*>(srcDescriptorInfo.desc.get());
					else
						descriptor = static_cast<ICPUBuffer*>(srcDescriptorInfo.desc.get());

					auto descriptorClone = descriptor->clone(_depth - 1);

					if (category == IDescriptor::E_CATEGORY::EC_SAMPLER)
						dstDescriptorInfo.desc = core::smart_refctd_ptr_static_cast<ICPUSampler>(std::move(descriptorClone));
					else if (category == IDescriptor::E_CATEGORY::EC_IMAGE)
						dstDescriptorInfo.desc = core::smart_refctd_ptr_static_cast<ICPUImageView>(std::move(descriptorClone));
					else if (category == IDescriptor::E_CATEGORY::EC_BUFFER_VIEW)
						dstDescriptorInfo.desc = core::smart_refctd_ptr_static_cast<ICPUBufferView>(std::move(descriptorClone));
					else
						dstDescriptorInfo.desc = core::smart_refctd_ptr_static_cast<ICPUBuffer>(std::move(descriptorClone));

				}

				// Clone the sampler.
				{
					if (category == IDescriptor::E_CATEGORY::EC_IMAGE and srcDescriptorInfo.info.combinedImageSampler.sampler)
						dstDescriptorInfo.info.combinedImageSampler.sampler = core::smart_refctd_ptr_static_cast<ICPUSampler>(srcDescriptorInfo.info.combinedImageSampler.sampler->clone(_depth - 1u));
				}
			}
			else
			{
				dstDescriptorInfo.desc = srcDescriptorInfo.desc;
			}
		}
	}

	return cp;
}

IAsset* ICPUDescriptorSet::getDependant_impl(size_t ix)
{
	for (auto i=0u; i<static_cast<uint32_t>(IDescriptor::E_TYPE::ET_COUNT); i++)
	if (m_descriptorInfos[i])
	{
		const auto size = m_descriptorInfos[i]->size();
		if (ix<size)
		{
			auto* desc = m_descriptorInfos[i]->operator[](ix).desc.get();
			if (desc)
			switch (IDescriptor::GetTypeCategory(static_cast<IDescriptor::E_TYPE>(i)))
			{
				case IDescriptor::EC_BUFFER:
					return static_cast<ICPUBuffer*>(desc);
				case IDescriptor::EC_SAMPLER:
					return static_cast<ICPUSampler*>(desc);
				case IDescriptor::EC_IMAGE:
					return static_cast<ICPUImageView*>(desc);
				case IDescriptor::EC_BUFFER_VIEW:
					return static_cast<ICPUBufferView*>(desc);
				case IDescriptor::EC_ACCELERATION_STRUCTURE:
					return static_cast<ICPUTopLevelAccelerationStructure*>(desc);
				default:
					break;
			}
			return nullptr;
		}
		else
			ix -= size;
	}
	return nullptr;
}
}