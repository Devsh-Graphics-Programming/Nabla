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
		{
            // TODO(achal): Samplers.
            for (auto i = 0u; i < asset::EDT_COUNT; ++i)
            {
                // There is no descriptor of such type in the set.
                if (m_descriptorStorageOffsets.data[i] == ~0u)
                    continue;

                const auto type = static_cast<asset::E_DESCRIPTOR_TYPE>(i);

                // Default-construct the core::smart_refctd_ptr<IDescriptor>s because even if the user didn't update the descriptor set with ILogicalDevice::updateDescriptorSet we
                // won't have uninitialized memory and destruction wouldn't crash in ~IGPUDescriptorSet.
                std::uninitialized_default_construct_n(getDescriptorStorage(type) + m_descriptorStorageOffsets.data[i], m_layout->getTotalDescriptorCount(type));
            }
        }

        // TODO(achal): Remove.
		uint8_t* getDescriptorMemory(const asset::E_DESCRIPTOR_TYPE type, const uint32_t binding) const;

        // This assumes that descriptors of a particular type in a set will always be contiguous in pool's storage memory, regardless of which binding they "belong" to.
        core::smart_refctd_ptr<asset::IDescriptor>* getDescriptors(const asset::E_DESCRIPTOR_TYPE type)
        {
            auto* baseAddress = getDescriptorStorage(type);
            if (baseAddress == nullptr)
                return nullptr;

            const auto offset = m_descriptorStorageOffsets.data[type];
            if (offset == ~0u)
                return nullptr;

            return baseAddress + offset;
        }

        inline uint32_t getDescriptorStorageOffset(const asset::E_DESCRIPTOR_TYPE type) const { return m_descriptorStorageOffsets.data[type]; }

	protected:
		virtual ~IGPUDescriptorSet()
		{
            // TODO(achal): Samplers.
            for (auto i = 0u; i < asset::EDT_COUNT; ++i)
            {
                // There is no descriptor of such type in the set.
                if (m_descriptorStorageOffsets.data[i] == ~0u)
                    continue;

                const auto type = static_cast<asset::E_DESCRIPTOR_TYPE>(i);
                std::destroy_n(getDescriptorStorage(type) + m_descriptorStorageOffsets.data[i], m_layout->getTotalDescriptorCount(type));
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
                baseAddress = reinterpret_cast<core::smart_refctd_ptr<asset::IDescriptor>*>(m_pool->m_UTB_STBStorage.get()) + m_pool->m_maxDescriptorCount[asset::EDT_UNIFORM_TEXEL_BUFFER];
                break;
            case asset::EDT_UNIFORM_BUFFER:
                baseAddress = reinterpret_cast<core::smart_refctd_ptr<asset::IDescriptor>*>(m_pool->m_UBO_SSBOStorage.get());
                break;
            case asset::EDT_STORAGE_BUFFER:
                baseAddress = reinterpret_cast<core::smart_refctd_ptr<asset::IDescriptor>*>(m_pool->m_UBO_SSBOStorage.get()) + m_pool->m_maxDescriptorCount[asset::EDT_UNIFORM_BUFFER];
                break;
            case asset::EDT_UNIFORM_BUFFER_DYNAMIC:
                baseAddress = reinterpret_cast<core::smart_refctd_ptr<asset::IDescriptor>*>(m_pool->m_UBO_SSBOStorage.get()) + (m_pool->m_maxDescriptorCount[asset::EDT_UNIFORM_BUFFER] + m_pool->m_maxDescriptorCount[asset::EDT_STORAGE_BUFFER]);
                break;
            case asset::EDT_STORAGE_BUFFER_DYNAMIC:
                baseAddress = reinterpret_cast<core::smart_refctd_ptr<asset::IDescriptor>*>(m_pool->m_UBO_SSBOStorage.get()) + (m_pool->m_maxDescriptorCount[asset::EDT_UNIFORM_BUFFER] + m_pool->m_maxDescriptorCount[asset::EDT_STORAGE_BUFFER] + m_pool->m_maxDescriptorCount[asset::EDT_UNIFORM_BUFFER_DYNAMIC]);
                break;
            case asset::EDT_INPUT_ATTACHMENT:
                baseAddress = reinterpret_cast<core::smart_refctd_ptr<asset::IDescriptor>*>(m_pool->m_storageImageStorage.get()) + m_pool->m_maxDescriptorCount[asset::EDT_STORAGE_IMAGE];
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