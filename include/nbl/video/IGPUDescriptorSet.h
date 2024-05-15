// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_I_GPU_DESCRIPTOR_SET_H_INCLUDED_
#define _NBL_VIDEO_I_GPU_DESCRIPTOR_SET_H_INCLUDED_


#include "nbl/asset/IDescriptorSet.h"

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
            const SDescriptorInfo* info;
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

        struct SDropDescriptorSet
        {
            IGPUDescriptorSet* dstSet;
            uint32_t binding;
            uint32_t arrayElement;
            uint32_t count;
        };

        inline uint64_t getVersion() const { return m_version.load(); }
        inline IDescriptorPool* getPool() const { return m_pool.get(); }
        inline bool isZombie() const { return (m_pool.get() == nullptr); }

        // small utility
        inline asset::IDescriptor::E_TYPE getBindingType(const uint32_t binding) const
        {
            for (auto t=0u; t<static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT); t++)
            {
                const auto type = static_cast<asset::IDescriptor::E_TYPE>(t);
                const auto& bindingRedirect = getLayout()->getDescriptorRedirect(type);
                if (bindingRedirect.getStorageOffset(redirect_t::binding_number_t{binding}).data!=redirect_t::Invalid)
                    return type;
            }
            return asset::IDescriptor::E_TYPE::ET_COUNT;
        }

	protected:
        IGPUDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>&& _layout, core::smart_refctd_ptr<IDescriptorPool>&& pool, IDescriptorPool::SStorageOffsets&& offsets);
        virtual ~IGPUDescriptorSet();

	private:
        inline void incrementVersion() { m_version.fetch_add(1ull); }

        friend class ILogicalDevice;
        asset::IDescriptor::E_TYPE validateWrite(const IGPUDescriptorSet::SWriteDescriptorSet& write) const;
        void processWrite(const IGPUDescriptorSet::SWriteDescriptorSet& write, const asset::IDescriptor::E_TYPE type);
        bool validateCopy(const IGPUDescriptorSet::SCopyDescriptorSet& copy) const;
        void processCopy(const IGPUDescriptorSet::SCopyDescriptorSet& copy);
        void dropDescriptors(const IGPUDescriptorSet::SDropDescriptorSet& drop);

        using redirect_t = IGPUDescriptorSetLayout::CBindingRedirect;
        // This assumes that descriptors of a particular type in the set will always be contiguous in pool's storage memory, regardless of which binding in the set they belong to.
        inline core::smart_refctd_ptr<asset::IDescriptor>* getDescriptors(const asset::IDescriptor::E_TYPE type, const uint32_t binding) const
        {
            const auto localOffset = getLayout()->getDescriptorRedirect(type).getStorageOffset(redirect_t::binding_number_t{ binding }).data;
            if (localOffset == ~0)
                return nullptr;

            auto* descriptors = getAllDescriptors(type);
            if (!descriptors)
                return nullptr;

            return descriptors+localOffset;
        }

        inline core::smart_refctd_ptr<IGPUSampler>* getMutableSamplers(const uint32_t binding) const
        {
            const auto localOffset = getLayout()->getMutableSamplerRedirect().getStorageOffset(redirect_t::binding_number_t{ binding }).data;
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

            const auto offset = m_storageOffsets.getDescriptorOffset(type);
            if (offset == ~0u)
                return nullptr;

            return baseAddress + offset;
        }

        inline core::smart_refctd_ptr<IGPUSampler>* getAllMutableSamplers() const
        {
            auto* baseAddress = m_pool->getMutableSamplerStorage();
            if (baseAddress == nullptr)
                return nullptr;

            const auto offset = m_storageOffsets.getMutableSamplerOffset();
            if (offset == ~0u)
                return nullptr;

            return baseAddress + offset;
        }

        std::atomic_uint64_t m_version;
        friend class IDescriptorPool;
        core::smart_refctd_ptr<IDescriptorPool> m_pool;
        const IDescriptorPool::SStorageOffsets m_storageOffsets;
};

}

#endif