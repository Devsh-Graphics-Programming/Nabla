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

class IGPUDescriptorSet : public asset::IDescriptorSet<const IGPUDescriptorSetLayout>, public IBackendObject
{
		using base_t = asset::IDescriptorSet<const IGPUDescriptorSetLayout>;

	public:
        struct SWriteDescriptorSet
        {
            //smart pointer not needed here
            IGPUDescriptorSet* dstSet;
            uint32_t binding;
            uint32_t arrayElement;
            uint32_t count;
            asset::IDescriptor::E_TYPE descriptorType;
            SDescriptorInfo* info;
        };

        struct SCopyDescriptorSet
        {
            //smart pointer not needed here
            IGPUDescriptorSet* dstSet;
            const IGPUDescriptorSet* srcSet;
            uint32_t srcBinding;
            uint32_t srcArrayElement;
            uint32_t dstBinding;
            uint32_t dstArrayElement;
            uint32_t count;
        };

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

	protected:
        virtual ~IGPUDescriptorSet();

	private:
        friend class ILogicalDevice;

        inline void incrementVersion() { m_version.fetch_add(1ull); }

        // TODO(achal): Don't know yet if we want to keep these.
        inline void processWrite(const IGPUDescriptorSet::SWriteDescriptorSet& write)
        {
            assert(write.dstSet == this);

            auto* descriptors = getDescriptors(write.descriptorType, write.binding);
            auto* samplers = getMutableSamplers(write.binding);
            for (auto j = 0; j < write.count; ++j)
            {
                descriptors[j] = write.info[j].desc;

                if (samplers)
                    samplers[j] = write.info[j].info.image.sampler;
            }
        }

#if 0
        inline void processCopy(const IGPUDescriptorSet::SCopyDescriptorSet& copy)
        {
            assert(copy.dstSet == this);

            for (uint32_t t = 0; t < static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT); ++t)
            {
                const auto type = static_cast<asset::IDescriptor::E_TYPE>(t);

                auto* srcDescriptors = srcDS->getDescriptors(type, pDescriptorCopies[i].srcBinding);
                auto* srcSamplers = srcDS->getMutableSamplers(pDescriptorCopies[i].srcBinding);

                auto* dstDescriptors = dstDS->getDescriptors(type, pDescriptorCopies[i].dstBinding);
                auto* dstSamplers = dstDS->getMutableSamplers(pDescriptorCopies[i].dstBinding);

                if (srcDescriptors && dstDescriptors)
                    std::copy_n(srcDescriptors, pDescriptorCopies[i].count, dstDescriptors);

                if (srcSamplers && dstSamplers)
                    std::copy_n(srcSamplers, pDescriptorCopies[i].count, dstSamplers);
            }
        }
#endif

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

        std::atomic_uint64_t m_version;
        const uint32_t m_parentPoolVersion;
        core::smart_refctd_ptr<IDescriptorPool> m_pool;
        IDescriptorPool::SDescriptorOffsets m_descriptorStorageOffsets;
};

}

#endif