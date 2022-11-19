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

class NBL_API ICPUDescriptorSet final : public IDescriptorSet<ICPUDescriptorSetLayout>, public IAsset, public impl::IEmulatedDescriptorSet<ICPUDescriptorSetLayout>
{
		using impl_t = impl::IEmulatedDescriptorSet<ICPUDescriptorSetLayout>;

	public:
		using base_t = IDescriptorSet<ICPUDescriptorSetLayout>;

		//! Contructor preallocating memory for SDescriptorBindings which user can fill later (using non-const getDescriptors()).
		//! @see getDescriptors()
		ICPUDescriptorSet(core::smart_refctd_ptr<ICPUDescriptorSetLayout>&& _layout) : base_t(std::move(_layout)), IAsset(), impl_t(m_layout.get())
		{
			for (uint32_t t = 0u; t < EDT_COUNT; ++t)
			{
				if (m_descriptors[t])
					m_descriptorInfos[t] = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ICPUDescriptorSet::SDescriptorInfo::SBufferImageInfo>>(m_descriptors[t]->size());
			}
		}

		inline size_t conservativeSizeEstimate() const override
		{
			assert(!"Invalid code path.");
			return 0xdeadbeefull;
		}

        core::smart_refctd_ptr<IAsset> clone(uint32_t _depth = ~0u) const override
        {
            auto layout = (_depth > 0u && m_layout) ? core::smart_refctd_ptr_static_cast<ICPUDescriptorSetLayout>(m_layout->clone(_depth - 1u)) : m_layout;
            auto cp = core::make_smart_refctd_ptr<ICPUDescriptorSet>(std::move(layout));
            clone_common(cp.get());

			auto cloneDescriptor = [](const core::smart_refctd_ptr<IDescriptor>& _desc, uint32_t _depth) -> core::smart_refctd_ptr<IDescriptor>
			{
				if (!_desc)
					return nullptr;

				IAsset* asset = nullptr;
				switch (_desc->getTypeCategory())
				{
				case IDescriptor::EC_BUFFER:
					asset = static_cast<ICPUBuffer*>(_desc.get()); break;
				case IDescriptor::EC_BUFFER_VIEW:
					asset = static_cast<ICPUBufferView*>(_desc.get()); break;
				case IDescriptor::EC_IMAGE:
					asset = static_cast<ICPUImageView*>(_desc.get()); break;
				}

				auto cp = asset->clone(_depth);

				switch (_desc->getTypeCategory())
				{
				case IDescriptor::EC_BUFFER:
					return core::smart_refctd_ptr_static_cast<ICPUBuffer>(std::move(cp));
				case IDescriptor::EC_BUFFER_VIEW:
					return core::smart_refctd_ptr_static_cast<ICPUBufferView>(std::move(cp));
				case IDescriptor::EC_IMAGE:
					return core::smart_refctd_ptr_static_cast<ICPUImageView>(std::move(cp));
				}
				return nullptr;
			};

			for (uint32_t t = 0u; t < EDT_COUNT; ++t)
			{
				const auto type = static_cast<E_DESCRIPTOR_TYPE>(t);

				for (uint32_t i = 0u; i < m_descriptors[type]->size(); ++i)
				{
					const auto& srcDescriptor = getDescriptorStorage(type)[i];
					const auto& srcDescriptorInfo = getDescriptorInfoStorage(type)[i];

					auto& dstDescriptor = cp->getDescriptorStorage(type)[i];
					auto& dstDescriptorInfo = cp->getDescriptorInfoStorage(type)[i];

					const auto descriptorCategory = srcDescriptor->getTypeCategory();
					if (descriptorCategory != IDescriptor::EC_IMAGE)
						dstDescriptorInfo.buffer = srcDescriptorInfo.buffer;
					else
						dstDescriptorInfo.image = srcDescriptorInfo.image;

					if (_depth > 0u)
						dstDescriptor = cloneDescriptor(srcDescriptor, _depth - 1u);
				}
			}

			for (uint32_t i = 0u; i < m_layout->getTotalMutableSamplerCount(); ++i)
				cp->getMutableSamplerStorage()[i] = core::smart_refctd_ptr_static_cast<ICPUSampler>(getMutableSamplerStorage()[i]->clone(_depth - 1u));

            return cp;
        }

		inline void convertToDummyObject(uint32_t referenceLevelsBelowToConvert=0u) override
		{
            convertToDummyObject_common(referenceLevelsBelowToConvert);

			if (referenceLevelsBelowToConvert)
			{
                --referenceLevelsBelowToConvert;
				m_layout->convertToDummyObject(referenceLevelsBelowToConvert);

				for (uint32_t t = 0u; t < EDT_COUNT; ++t)
				{
					const auto type = static_cast<E_DESCRIPTOR_TYPE>(t);
					const auto descriptorCount = m_layout->getTotalDescriptorCount(type);
					if (descriptorCount == 0ull)
						continue;

					auto descriptors = m_descriptors[type]->begin();
					assert(descriptors);

					for (uint32_t i = 0u; i < descriptorCount; ++i)
					{
						switch (descriptors[i]->getTypeCategory())
						{
						case IDescriptor::EC_BUFFER:
							static_cast<asset::ICPUBuffer*>(descriptors[i].get())->convertToDummyObject(referenceLevelsBelowToConvert);
							break;

						case IDescriptor::EC_IMAGE:
						{
							static_cast<asset::ICPUImageView*>(descriptors[i].get())->convertToDummyObject(referenceLevelsBelowToConvert);
							const auto mutableSamplerCount = m_layout->getTotalMutableSamplerCount();
							for (uint32_t s = 0u; s < mutableSamplerCount; ++s)
								m_mutableSamplers->begin()[s]->convertToDummyObject(referenceLevelsBelowToConvert);
						} break;

						case IDescriptor::EC_BUFFER_VIEW:
							static_cast<asset::ICPUBufferView*>(descriptors[i].get())->convertToDummyObject(referenceLevelsBelowToConvert);
							break;

						default:
							assert(!"Invalid code path.");
						}
					}
				}
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

		std::pair<core::SRange<core::smart_refctd_ptr<IDescriptor>>, core::SRange<SDescriptorInfo::SBufferImageInfo>> getDescriptors(const uint32_t binding)
		{
			const auto bindingInfo = std::lower_bound(getLayout()->getBindings().begin(), getLayout()->getBindings().end(), ICPUDescriptorSetLayout::SBinding{binding});
			assert(bindingInfo->binding == binding && "binding is not in the descriptor set!");

			const uint32_t descriptorOffset = getLayout()->getDescriptorOffset(bindingInfo->type, binding);
			const uint32_t descriptorCount = bindingInfo->count;

			auto descriptorsBegin = m_descriptors[bindingInfo->type]->begin() + descriptorOffset;
			auto descriptorInfosBegin = m_descriptorInfos[bindingInfo->type]->begin() + descriptorOffset;

			return { {descriptorsBegin, descriptorsBegin+descriptorCount}, {descriptorInfosBegin, descriptorInfosBegin+descriptorCount} };
		}

		core::SRange<core::smart_refctd_ptr<ICPUSampler>> getMutableSamplers(const uint32_t binding) const
		{
			const uint32_t offset = getLayout()->getMutableSamplerOffset(binding);
			if (offset == ~0u)
				return { nullptr, nullptr };

			const auto bindingInfo = std::lower_bound(getLayout()->getBindings().begin(), getLayout()->getBindings().end(), ICPUDescriptorSetLayout::SBinding{ binding });
			assert(bindingInfo->binding == binding && "binding is not in the descriptor set!");

			auto samplersBegin = m_mutableSamplers->begin() + offset;
			return { samplersBegin, samplersBegin + bindingInfo->count };
		}

		inline core::smart_refctd_ptr<IDescriptor>* getDescriptorStorage(const E_DESCRIPTOR_TYPE type) const { return m_descriptors[type]->begin(); }
		inline SDescriptorInfo::SBufferImageInfo* getDescriptorInfoStorage(const E_DESCRIPTOR_TYPE type) const { return m_descriptorInfos[type]->begin(); }
		inline core::smart_refctd_ptr<ICPUSampler>* getMutableSamplerStorage() const { return m_mutableSamplers->begin(); }

		bool canBeRestoredFrom(const IAsset* _other) const override
		{
			auto* other = static_cast<const ICPUDescriptorSet*>(_other);
			return m_layout->canBeRestoredFrom(other->m_layout.get());
		}

	protected:
		void restoreFromDummy_impl(IAsset* _other, uint32_t _levelsBelow) override
		{
			auto* other = static_cast<ICPUDescriptorSet*>(_other);

			if (_levelsBelow)
			{
				--_levelsBelow;
				restoreFromDummy_impl_call(m_layout.get(), other->getLayout(), _levelsBelow);

				for (uint32_t t = 0u; t < EDT_COUNT; ++t)
				{
					const auto type = static_cast<E_DESCRIPTOR_TYPE>(t);
					const auto descriptorCount = m_layout->getTotalDescriptorCount(type);
					if (descriptorCount == 0ull)
						continue;

					auto descriptors = m_descriptors[type]->begin();
					assert(descriptors);

					auto otherDescriptors = other->m_descriptors[type]->begin();

					for (uint32_t i = 0u; i < descriptorCount; ++i)
					{
						switch (descriptors[i]->getTypeCategory())
						{
						case IDescriptor::EC_BUFFER:
							restoreFromDummy_impl_call(static_cast<ICPUBuffer*>(descriptors[i].get()), static_cast<ICPUBuffer*>(otherDescriptors[i].get()), _levelsBelow);
							break;

						case IDescriptor::EC_IMAGE:
							restoreFromDummy_impl_call(static_cast<ICPUImageView*>(descriptors[i].get()), static_cast<ICPUImageView*>(otherDescriptors[i].get()), _levelsBelow);
							break;

						case IDescriptor::EC_BUFFER_VIEW:
							restoreFromDummy_impl_call(static_cast<ICPUBufferView*>(descriptors[i].get()), static_cast<ICPUBufferView*>(otherDescriptors[i].get()), _levelsBelow);
							break;

						default:
							assert(!"Invalid code path.");
						}
					}
				}

				for (uint32_t i = 0u; i < m_layout->getTotalMutableSamplerCount(); ++i)
					restoreFromDummy_impl_call(m_mutableSamplers->begin()[i].get(), other->m_mutableSamplers->begin()[i].get(), _levelsBelow);
			}
		}

		bool isAnyDependencyDummy_impl(uint32_t _levelsBelow) const override
		{
			--_levelsBelow;
			if (m_layout->isAnyDependencyDummy(_levelsBelow))
				return true;

			for (uint32_t t = 0u; t < EDT_COUNT; ++t)
			{
				const auto type = static_cast<E_DESCRIPTOR_TYPE>(t);
				const auto descriptorCount = m_layout->getTotalDescriptorCount(type);
				if (descriptorCount == 0ull)
					continue;

				auto descriptors = m_descriptors[type]->begin();
				assert(descriptors);

				for (uint32_t i = 0u; i < descriptorCount; ++i)
				{
					switch (descriptors[i]->getTypeCategory())
					{
					case IDescriptor::EC_BUFFER:
						if (static_cast<ICPUBuffer*>(descriptors[i].get())->isAnyDependencyDummy(_levelsBelow))
							return true;
						break;

					case IDescriptor::EC_IMAGE:
						if (static_cast<ICPUImageView*>(descriptors[i].get())->isAnyDependencyDummy(_levelsBelow))
							return true;
						break;

					case IDescriptor::EC_BUFFER_VIEW:
						if (static_cast<ICPUBufferView*>(descriptors[i].get())->isAnyDependencyDummy(_levelsBelow))
							return true;
						break;

					default:
						assert(!"Invalid code path.");
					}
				}
			}

			for (uint32_t i = 0u; i < m_layout->getTotalMutableSamplerCount(); ++i)
			{
				if (m_mutableSamplers->begin()[i]->isAnyDependencyDummy(_levelsBelow))
					return true;
			}

			return false;
		}

		virtual ~ICPUDescriptorSet() = default;

		private:
			// Mutable samplers are NOT stored in this array (in SDescriptorInfo::SImageInfo::sampler member), but in IEmulatedDescriptorSet::m_mutableSamplers.
			core::smart_refctd_dynamic_array<ICPUDescriptorSet::SDescriptorInfo::SBufferImageInfo> m_descriptorInfos[EDT_COUNT];
};

}

#endif