// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_I_GPU_DESCRIPTOR_SET_H_INCLUDED__
#define __NBL_VIDEO_I_GPU_DESCRIPTOR_SET_H_INCLUDED__


#include "nbl/asset/IDescriptorSet.h"

#include "nbl/video/IGPUBuffer.h"
#include "nbl/video/IGPUBufferView.h"
#include "nbl/video/IGPUImageView.h"
#include "nbl/video/IGPUSampler.h"
#include "nbl/video/IGPUDescriptorSetLayout.h"


namespace nbl::video
{

//! GPU Version of Descriptor Set
/*
	@see IDescriptorSet
*/

class NBL_API IGPUDescriptorSet : public asset::IDescriptorSet<const IGPUDescriptorSetLayout>, public IBackendObject
{
		using base_t = asset::IDescriptorSet<const IGPUDescriptorSetLayout>;

	public:
		IGPUDescriptorSet(core::smart_refctd_ptr<const ILogicalDevice>&& dev, core::smart_refctd_ptr<const IGPUDescriptorSetLayout>&& _layout, core::smart_refctd_ptr<IDescriptorPool>&& pool, IDescriptorPool::SDescriptorOffsets&& descriptorStorageOffsets)
			: base_t(std::move(_layout)), IBackendObject(std::move(dev)), m_pool(std::move(pool)), m_descriptorStorageOffsets(std::move(descriptorStorageOffsets))
		{}

		uint8_t* getDescriptorMemory(const asset::E_DESCRIPTOR_TYPE type, const uint32_t binding) const;

	protected:
		virtual ~IGPUDescriptorSet()
		{
			for (const auto& b : getLayout()->getBindings())
			{
				assert(m_descriptorStorageOffsets.data[b.type] != ~0u && "Descriptor of this type doesn't exist in the set!");

				auto* descriptorMemory = getDescriptorMemory(b.type, b.binding);
				assert(descriptorMemory);

				auto* descriptors = reinterpret_cast<core::smart_refctd_ptr<const asset::IDescriptor>*>(descriptorMemory);

				for (auto i = 0; i < b.count; ++i)
					descriptors[i].~smart_refctd_ptr();
			}
		}

	private:
		core::smart_refctd_ptr<asset::IDescriptor>* getDescriptorStorage(const asset::E_DESCRIPTOR_TYPE type) const override
		{
            core::smart_refctd_ptr<asset::IDescriptor>* baseAddress;
            switch (type)
            {
            case asset::EDT_COMBINED_IMAGE_SAMPLER:
                // TODO(achal): This is obviously wrong because m_combinedImageSamplerStorage is an array of SCombinedImageSampler. Switch to SoA to fix this shit.
                baseAddress = reinterpret_cast<core::smart_refctd_ptr<asset::IDescriptor>*>(m_pool->m_combinedImageSamplerStorage.get());
                break;
            case asset::EDT_STORAGE_IMAGE:
                baseAddress = reinterpret_cast<core::smart_refctd_ptr<asset::IDescriptor>*>(m_pool->m_storageImageStorage.get());
                break;
            case asset::EDT_UNIFORM_TEXEL_BUFFER:
                baseAddress = reinterpret_cast<core::smart_refctd_ptr<asset::IDescriptor>*>(m_pool->m_UTB_STBStorage.get());
                break;
            case asset::EDT_STORAGE_TEXEL_BUFFER:
                baseAddress = reinterpret_cast<core::smart_refctd_ptr<asset::IDescriptor>*>(m_pool->m_UTB_STBStorage.get()) + m_pool->m_maxDescriptorCount[asset::EDT_UNIFORM_TEXEL_BUFFER] * sizeof(void*);
                break;
            case asset::EDT_UNIFORM_BUFFER:
                baseAddress = reinterpret_cast<core::smart_refctd_ptr<asset::IDescriptor>*>(m_pool->m_UBO_SSBOStorage.get());
                break;
            case asset::EDT_STORAGE_BUFFER:
                baseAddress = reinterpret_cast<core::smart_refctd_ptr<asset::IDescriptor>*>(m_pool->m_UBO_SSBOStorage.get()) + m_pool->m_maxDescriptorCount[asset::EDT_UNIFORM_BUFFER] * sizeof(void*);
                break;
            case asset::EDT_UNIFORM_BUFFER_DYNAMIC:
                baseAddress = reinterpret_cast<core::smart_refctd_ptr<asset::IDescriptor>*>(m_pool->m_UBO_SSBOStorage.get()) + (m_pool->m_maxDescriptorCount[asset::EDT_UNIFORM_BUFFER] + m_pool->m_maxDescriptorCount[asset::EDT_STORAGE_BUFFER]) * sizeof(void*);
                break;
            case asset::EDT_STORAGE_BUFFER_DYNAMIC:
                baseAddress = reinterpret_cast<core::smart_refctd_ptr<asset::IDescriptor>*>(m_pool->m_UBO_SSBOStorage.get()) + (m_pool->m_maxDescriptorCount[asset::EDT_UNIFORM_BUFFER] + m_pool->m_maxDescriptorCount[asset::EDT_STORAGE_BUFFER] + m_pool->m_maxDescriptorCount[asset::EDT_UNIFORM_BUFFER_DYNAMIC]) * sizeof(void*);
                break;
            case asset::EDT_INPUT_ATTACHMENT:
                baseAddress = reinterpret_cast<core::smart_refctd_ptr<asset::IDescriptor>*>(m_pool->m_storageImageStorage.get()) + m_pool->m_maxDescriptorCount[asset::EDT_STORAGE_IMAGE] * sizeof(void*);
                break;
            case asset::EDT_ACCELERATION_STRUCTURE:
                baseAddress = reinterpret_cast<core::smart_refctd_ptr<asset::IDescriptor>*>(m_pool->m_accelerationStructureStorage.get());
                break;
            default:
                assert(!"Invalid code path.");
                return nullptr;
            }

            return baseAddress;
		}

		// TODO(achal): We need another method for SamplerStorage here

#if 0
		void allocateDescriptors() override
		{
			// No-OP because allocation is already been done in ILogicalDevice::createDescriptorSet
		}
#endif

		core::smart_refctd_ptr<IDescriptorPool> m_pool;
		IDescriptorPool::SDescriptorOffsets m_descriptorStorageOffsets;
};

}

#endif