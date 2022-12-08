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
		IGPUDescriptorSet(core::smart_refctd_ptr<const ILogicalDevice>&& dev, core::smart_refctd_ptr<const IGPUDescriptorSetLayout>&& _layout, core::smart_refctd_ptr<IDescriptorPool>&& pool)
			: base_t(std::move(_layout)), IBackendObject(std::move(dev)), m_version(0ull), m_pool(std::move(pool))
		{
            allocateDescriptors();

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

            const auto mutableSamplerCount = m_layout->getTotalMutableSamplerCount();
            if (mutableSamplerCount > 0)
                std::uninitialized_default_construct_n(getMutableSamplerStorage() + m_descriptorStorageOffsets.data[asset::EDT_COUNT], mutableSamplerCount);
        }

        inline uint64_t getVersion() const { return m_version.load(); }

        inline void incrementVersion() { m_version.fetch_add(1ull); }

        inline core::smart_refctd_ptr<asset::IDescriptor>* getAllDescriptors(const asset::E_DESCRIPTOR_TYPE type) const
        {
            auto* baseAddress = getDescriptorStorage(type);
            if (baseAddress == nullptr)
                return nullptr;

            const auto offset = m_descriptorStorageOffsets.data[type];
            if (offset == ~0u)
                return nullptr;

            return baseAddress + offset;
        }

        inline core::smart_refctd_ptr<IGPUSampler>* getAllMutableSamplers() const
        {
            auto* baseAddress = getMutableSamplerStorage();
            if (baseAddress == nullptr)
                return nullptr;

            const auto poolOffset = m_descriptorStorageOffsets.data[asset::EDT_COUNT];
            if (poolOffset == ~0u)
                return nullptr;

            return baseAddress + poolOffset;
        }

        // This assumes that descriptors of a particular type in the set will always be contiguous in pool's storage memory, regardless of which binding in the set they belong to.
        inline core::smart_refctd_ptr<asset::IDescriptor>* getDescriptors(const asset::E_DESCRIPTOR_TYPE type, const uint32_t binding) const
        {
            const auto localOffset = getLayout()->getDescriptorOffset(type, binding);
            if (localOffset == ~0)
                return nullptr;

            auto* descriptors = getAllDescriptors(type);
            if (!descriptors)
                return nullptr;

            return descriptors + localOffset;
        }

        inline core::smart_refctd_ptr<IGPUSampler>* getMutableSamplers(const uint32_t binding) const
        {
            const auto localOffset = getLayout()->getMutableSamplerOffset(binding);
            if (localOffset == ~0u)
                return nullptr;

            auto* samplers = getAllMutableSamplers();
            if (!samplers)
                return nullptr;

            return samplers + localOffset;
        }

        inline uint32_t getDescriptorStorageOffset(const asset::E_DESCRIPTOR_TYPE type) const { return m_descriptorStorageOffsets.data[type]; }
        inline uint32_t getMutableSamplerStorageOffset() const { return m_descriptorStorageOffsets.data[asset::EDT_COUNT]; }

	protected:
		virtual ~IGPUDescriptorSet()
		{
            for (auto i = 0u; i < asset::EDT_COUNT; ++i)
            {
                // There is no descriptor of such type in the set.
                if (m_descriptorStorageOffsets.data[i] == ~0u)
                    continue;

                const auto type = static_cast<asset::E_DESCRIPTOR_TYPE>(i);
                std::destroy_n(getDescriptorStorage(type) + m_descriptorStorageOffsets.data[i], m_layout->getTotalDescriptorCount(type));
            }

            const auto mutableSamplerCount = m_layout->getTotalMutableSamplerCount();
            if (mutableSamplerCount > 0)
                std::destroy_n(getMutableSamplerStorage() + getMutableSamplerStorageOffset(), mutableSamplerCount);
		}

	private:
        struct SDescriptorOffsets
        {
            SDescriptorOffsets()
            {
                // The default constructor should initiailze all the offsets to an invalid value (~0u) because ~IGPUDescriptorSet relies on it to
                // know which descriptors are present in the set and hence should be destroyed.
                std::fill_n(data, asset::EDT_COUNT + 1, ~0u);
            }

            uint32_t data[asset::EDT_COUNT + 1];
        };

        // Returns the offset into the pool's descriptor storage. These offsets will be combined
        // later with base memory addresses to get the actual memory address where we put the core::smart_refctd_ptr<const IDescriptor>.
        void allocateDescriptors() override;

		inline core::smart_refctd_ptr<asset::IDescriptor>* getDescriptorStorage(const asset::E_DESCRIPTOR_TYPE type) const
		{
            core::smart_refctd_ptr<asset::IDescriptor>* baseAddress;
            switch (type)
            {
            case asset::EDT_COMBINED_IMAGE_SAMPLER:
                baseAddress = reinterpret_cast<core::smart_refctd_ptr<asset::IDescriptor>*>(m_pool->m_textureStorage.get());
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

        inline core::smart_refctd_ptr<IGPUSampler>* getMutableSamplerStorage() const { return reinterpret_cast<core::smart_refctd_ptr<IGPUSampler>*>(m_pool->m_mutableSamplerStorage.get()); }

        std::atomic_uint64_t m_version;
		core::smart_refctd_ptr<IDescriptorPool> m_pool;
		SDescriptorOffsets m_descriptorStorageOffsets;
};

}

#endif