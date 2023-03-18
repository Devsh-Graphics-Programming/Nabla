#include "nbl/asset/ICPUDescriptorSet.h"

namespace nbl::asset
{

core::SRange<ICPUDescriptorSet::SDescriptorInfo> ICPUDescriptorSet::getDescriptorInfos(const ICPUDescriptorSetLayout::CBindingRedirect::binding_number_t binding, IDescriptor::E_TYPE type)
{
	assert(!isImmutable_debug());
	auto immutableResult = const_cast<const ICPUDescriptorSet*>(this)->getDescriptorInfos(binding, type);
	return {const_cast<ICPUDescriptorSet::SDescriptorInfo*>(immutableResult.begin()), const_cast<ICPUDescriptorSet::SDescriptorInfo*>(immutableResult.end())};
}

core::SRange<const ICPUDescriptorSet::SDescriptorInfo> ICPUDescriptorSet::getDescriptorInfos(const ICPUDescriptorSetLayout::CBindingRedirect::binding_number_t binding, IDescriptor::E_TYPE type) const
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
			return { nullptr, nullptr };
	}

	const auto& redirect = getLayout()->getDescriptorRedirect(type);
	const auto bindingNumberIndex = redirect.findBindingStorageIndex(binding);
	if (bindingNumberIndex.data == redirect.Invalid)
		return { nullptr, nullptr };

	const auto offset = redirect.getStorageOffset(asset::ICPUDescriptorSetLayout::CBindingRedirect::storage_range_index_t{ bindingNumberIndex }).data;
	const auto count = redirect.getCount(asset::ICPUDescriptorSetLayout::CBindingRedirect::storage_range_index_t{ bindingNumberIndex });

	auto infosBegin = m_descriptorInfos[static_cast<uint32_t>(type)]->begin() + offset;

	return { infosBegin, infosBegin + count };
}

core::smart_refctd_ptr<IAsset> ICPUDescriptorSet::clone(uint32_t _depth) const
{
	auto layout = (_depth > 0u && m_layout) ? core::smart_refctd_ptr_static_cast<ICPUDescriptorSetLayout>(m_layout->clone(_depth - 1u)) : m_layout;
	auto cp = core::make_smart_refctd_ptr<ICPUDescriptorSet>(std::move(layout));
	clone_common(cp.get());

	for (uint32_t t = 0u; t < static_cast<uint32_t>(IDescriptor::E_TYPE::ET_COUNT); ++t)
	{
		const auto type = static_cast<IDescriptor::E_TYPE>(t);

		for (uint32_t i = 0u; i < m_descriptorInfos[t]->size(); ++i)
		{
			const auto& srcDescriptorInfo = m_descriptorInfos[t]->begin()[i];
			auto& dstDescriptorInfo = cp->m_descriptorInfos[t]->begin()[i];

			auto category = getCategoryFromType(type);
			
			if (category == IDescriptor::E_CATEGORY::EC_IMAGE)
				dstDescriptorInfo.info.image = srcDescriptorInfo.info.image;
			else
				dstDescriptorInfo.info.buffer = srcDescriptorInfo.info.buffer;

			if (_depth > 0u)
			{
				// Clone the descriptor.
				{
					assert(srcDescriptorInfo.desc);

					IAsset* descriptor = nullptr;
					if (category == IDescriptor::E_CATEGORY::EC_IMAGE)
						descriptor = static_cast<ICPUImageView*>(srcDescriptorInfo.desc.get());
					else if (category == IDescriptor::E_CATEGORY::EC_BUFFER_VIEW)
						descriptor = static_cast<ICPUBufferView*>(srcDescriptorInfo.desc.get());
					else
						descriptor = static_cast<ICPUBuffer*>(srcDescriptorInfo.desc.get());

					auto descriptorClone = descriptor->clone(_depth - 1);

					if (category == IDescriptor::E_CATEGORY::EC_IMAGE)
						dstDescriptorInfo.desc = core::smart_refctd_ptr_static_cast<ICPUImageView>(std::move(descriptorClone));
					else if (category == IDescriptor::E_CATEGORY::EC_BUFFER_VIEW)
						dstDescriptorInfo.desc = core::smart_refctd_ptr_static_cast<ICPUBufferView>(std::move(descriptorClone));
					else
						dstDescriptorInfo.desc = core::smart_refctd_ptr_static_cast<ICPUBuffer>(std::move(descriptorClone));

				}

				// Clone the sampler.
				{
					if ((category == IDescriptor::E_CATEGORY::EC_IMAGE) && srcDescriptorInfo.info.image.sampler)
						dstDescriptorInfo.info.image.sampler = core::smart_refctd_ptr_static_cast<ICPUSampler>(srcDescriptorInfo.info.image.sampler->clone(_depth - 1u));
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

void ICPUDescriptorSet::convertToDummyObject(uint32_t referenceLevelsBelowToConvert)
{
	convertToDummyObject_common(referenceLevelsBelowToConvert);

	if (referenceLevelsBelowToConvert)
	{
		--referenceLevelsBelowToConvert;

		for (uint32_t t = 0u; t < static_cast<uint32_t>(IDescriptor::E_TYPE::ET_COUNT); ++t)
		{
			const auto type = static_cast<IDescriptor::E_TYPE>(t);
			const auto descriptorCount = m_layout->getTotalDescriptorCount(type);
			if (descriptorCount == 0ull)
				continue;

			auto descriptorInfos = m_descriptorInfos[t]->begin();
			assert(descriptorInfos);

			const auto category = getCategoryFromType(type);
			for (uint32_t i = 0u; i < descriptorCount; ++i)
			{
				switch (category)
				{
				case IDescriptor::E_CATEGORY::EC_BUFFER:
					static_cast<asset::ICPUBuffer*>(descriptorInfos[i].desc.get())->convertToDummyObject(referenceLevelsBelowToConvert);
					break;

				case IDescriptor::E_CATEGORY::EC_IMAGE:
				{
					static_cast<asset::ICPUImageView*>(descriptorInfos[i].desc.get())->convertToDummyObject(referenceLevelsBelowToConvert);
					if (descriptorInfos[i].info.image.sampler)
						descriptorInfos[i].info.image.sampler->convertToDummyObject(referenceLevelsBelowToConvert);
				} break;

				case IDescriptor::EC_BUFFER_VIEW:
					static_cast<asset::ICPUBufferView*>(descriptorInfos[i].desc.get())->convertToDummyObject(referenceLevelsBelowToConvert);
					break;

				default:
					assert(!"Invalid code path.");
				}
			}
		}

		m_layout->convertToDummyObject(referenceLevelsBelowToConvert);
	}
}

void ICPUDescriptorSet::restoreFromDummy_impl(IAsset* _other, uint32_t _levelsBelow)
{
	auto* other = static_cast<ICPUDescriptorSet*>(_other);

	if (_levelsBelow)
	{
		--_levelsBelow;
		restoreFromDummy_impl_call(m_layout.get(), other->getLayout(), _levelsBelow);

		for (uint32_t t = 0u; t < static_cast<uint32_t>(IDescriptor::E_TYPE::ET_COUNT); ++t)
		{
			const auto type = static_cast<IDescriptor::E_TYPE>(t);
			const auto descriptorCount = m_layout->getTotalDescriptorCount(type);
			if (descriptorCount == 0ull)
				continue;

			auto descriptorInfos = m_descriptorInfos[t]->begin();
			assert(descriptorInfos);

			auto otherDescriptorInfos = other->m_descriptorInfos[t]->begin();

			const auto category = getCategoryFromType(type);
			for (uint32_t i = 0u; i < descriptorCount; ++i)
			{
				switch (category)
				{
				case IDescriptor::E_CATEGORY::EC_BUFFER:
					restoreFromDummy_impl_call(static_cast<ICPUBuffer*>(descriptorInfos[i].desc.get()), static_cast<ICPUBuffer*>(otherDescriptorInfos[i].desc.get()), _levelsBelow);
					break;

				case IDescriptor::EC_IMAGE:
				{
					restoreFromDummy_impl_call(static_cast<ICPUImageView*>(descriptorInfos[i].desc.get()), static_cast<ICPUImageView*>(otherDescriptorInfos[i].desc.get()), _levelsBelow);
					if (descriptorInfos[i].info.image.sampler && otherDescriptorInfos[i].info.image.sampler)
						restoreFromDummy_impl_call(descriptorInfos[i].info.image.sampler.get(), otherDescriptorInfos[i].info.image.sampler.get(), _levelsBelow);
				} break;

				case IDescriptor::EC_BUFFER_VIEW:
					restoreFromDummy_impl_call(static_cast<ICPUBufferView*>(descriptorInfos[i].desc.get()), static_cast<ICPUBufferView*>(otherDescriptorInfos[i].desc.get()), _levelsBelow);
					break;

				default:
					assert(!"Invalid code path.");
				}
			}
		}
	}
}

bool ICPUDescriptorSet::isAnyDependencyDummy_impl(uint32_t _levelsBelow) const
{
	--_levelsBelow;
	if (_levelsBelow)
	{
		if (m_layout->isAnyDependencyDummy(_levelsBelow))
			return true;

		for (uint32_t t = 0u; t < static_cast<uint32_t>(IDescriptor::E_TYPE::ET_COUNT); ++t)
		{
			const auto type = static_cast<IDescriptor::E_TYPE>(t);
			const auto descriptorCount = m_layout->getTotalDescriptorCount(type);
			if (descriptorCount == 0ull)
				continue;

			auto descriptorInfos = m_descriptorInfos[t]->begin();
			assert(descriptorInfos);

			const auto category = getCategoryFromType(type);
			for (uint32_t i = 0u; i < descriptorCount; ++i)
			{
				switch (category)
				{
				case IDescriptor::EC_BUFFER:
					if (static_cast<ICPUBuffer*>(descriptorInfos[i].desc.get())->isAnyDependencyDummy(_levelsBelow))
						return true;
					break;

				case IDescriptor::EC_IMAGE:
				{
					if (static_cast<ICPUImageView*>(descriptorInfos[i].desc.get())->isAnyDependencyDummy(_levelsBelow))
						return true;

					if (descriptorInfos[i].info.image.sampler && descriptorInfos[i].info.image.sampler->isAnyDependencyDummy(_levelsBelow))
						return true;
				} break;

				case IDescriptor::EC_BUFFER_VIEW:
					if (static_cast<ICPUBufferView*>(descriptorInfos[i].desc.get())->isAnyDependencyDummy(_levelsBelow))
						return true;
					break;

				default:
					assert(!"Invalid code path.");
				}
			}
		}
	}
	return false;
}

}