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

#include "nbl/video/IDescriptorPool.h"

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
        IGPUDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>&& _layout, core::smart_refctd_ptr<IDescriptorPool>&& pool, IDescriptorPool::SDescriptorOffsets&& offsets);

        inline uint64_t getVersion() const { return m_version.load(); }

        inline IDescriptorPool* getPool() const { return m_pool.get(); }

        inline bool isZombie() const
        {
            if (m_pool->m_version.load() > m_parentPoolVersion)
                return true;
            else
                return false;
        }

        inline core::smart_refctd_ptr<asset::IDescriptor>* getAllDescriptors(const asset::IDescriptor::E_TYPE type) const
        {
            auto* baseAddress = m_pool->getDescriptorStorage(type);
            if (baseAddress == nullptr)
                return nullptr;

            const auto offset = m_descriptorStorageOffsets.data[static_cast<uint32_t>(type)];
            if (offset == ~0u)
                return nullptr;

            return baseAddress + offset;
        }

        inline core::smart_refctd_ptr<IGPUSampler>* getAllMutableSamplers() const
        {
            auto* baseAddress = m_pool->getMutableSamplerStorage();
            if (baseAddress == nullptr)
                return nullptr;

            const auto poolOffset = m_descriptorStorageOffsets.data[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT)];
            if (poolOffset == ~0u)
                return nullptr;

            return baseAddress + poolOffset;
        }

        inline uint32_t getDescriptorStorageOffset(const asset::IDescriptor::E_TYPE type) const { return m_descriptorStorageOffsets.data[static_cast<uint32_t>(type)]; }
        inline uint32_t getMutableSamplerStorageOffset() const { return m_descriptorStorageOffsets.data[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT)]; }

	protected:
		virtual ~IGPUDescriptorSet()
		{
            if (!isZombie())
            {
                const bool allowsFreeing = m_pool->m_flags & IDescriptorPool::ECF_FREE_DESCRIPTOR_SET_BIT;
                for (auto i = 0u; i < static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT); ++i)
                {
                    // There is no descriptor of such type in the set.
                    if (m_descriptorStorageOffsets.data[i] == ~0u)
                        continue;

                    const auto type = static_cast<asset::IDescriptor::E_TYPE>(i);

                    const uint32_t allocatedOffset = getDescriptorStorageOffset(type);
                    assert(allocatedOffset != ~0u);

                    const uint32_t count = m_layout->getTotalDescriptorCount(type);
                    assert(count != 0u);

                    std::destroy_n(m_pool->getDescriptorStorage(type) + allocatedOffset, count);

                    if (allowsFreeing)
                        m_pool->m_descriptorAllocators[i]->free(allocatedOffset, count);
                }

                const auto mutableSamplerCount = m_layout->getTotalMutableSamplerCount();
                if (mutableSamplerCount > 0)
                {
                    const uint32_t allocatedOffset = getMutableSamplerStorageOffset();
                    assert(allocatedOffset != ~0u);
                        
                    std::destroy_n(m_pool->getMutableSamplerStorage() + allocatedOffset, mutableSamplerCount);

                    if (allowsFreeing)
                        m_pool->m_descriptorAllocators[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT)]->free(allocatedOffset, mutableSamplerCount);
                }
            }
		}

	private:
        friend class ILogicalDevice;

        // This assumes that descriptors of a particular type in the set will always be contiguous in pool's storage memory, regardless of which binding in the set they belong to.
        inline core::smart_refctd_ptr<asset::IDescriptor>* getDescriptors(const asset::IDescriptor::E_TYPE type, const uint32_t binding) const
        {
            const auto localOffset = getLayout()->getDescriptorRedirect(type).getStorageOffset(IGPUDescriptorSetLayout::CBindingRedirect::binding_number_t{ binding }).data;
            if (localOffset == ~0)
                return nullptr;

            auto* descriptors = getAllDescriptors(type);
            if (!descriptors)
                return nullptr;

            return descriptors + localOffset;
        }

        inline core::smart_refctd_ptr<IGPUSampler>* getMutableSamplers(const uint32_t binding) const
        {
            const auto localOffset = getLayout()->getMutableSamplerRedirect().getStorageOffset(IGPUDescriptorSetLayout::CBindingRedirect::binding_number_t{ binding }).data;
            if (localOffset == getLayout()->getMutableSamplerRedirect().Invalid)
                return nullptr;

            auto* samplers = getAllMutableSamplers();
            if (!samplers)
                return nullptr;

            return samplers + localOffset;
        }

        inline void incrementVersion() { m_version.fetch_add(1ull); }

        std::atomic_uint64_t m_version;
        const uint32_t m_parentPoolVersion;
        core::smart_refctd_ptr<IDescriptorPool> m_pool;
        IDescriptorPool::SDescriptorOffsets m_descriptorStorageOffsets;
};

}

#endif