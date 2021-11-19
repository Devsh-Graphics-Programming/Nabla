// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef _NBL_SCENE_C_SKIN_INSTANCE_CACHE_H_INCLUDED_
#define _NBL_SCENE_C_SKIN_INSTANCE_CACHE_H_INCLUDED_

#include "nbl/scene/ISkinInstanceCache.h"

namespace nbl::scene
{

template<template<class...> class allocator = core::allocator>
class CSkinInstanceCache final : public ISkinInstanceCache
{
		using this_t = CSkinInstanceCache<allocator>;

	public:
        // easy dont care creation
        static inline core::smart_refctd_ptr<this_t> create(ImplicitBufferCreationParameters&& params, allocator<uint8_t>&& alloc=allocator<uint8_t>())
        {
            if (!params.device || params.minAllocSizeInJoints==0u || params.jointCapacity<params.minAllocSizeInJoints)
                return nullptr;

            const auto& limits = params.device->getPhysicalDevice()->getLimits();
            
            ExplicitBufferCreationParameters explicit_params;
            static_cast<CreationParametersBase&>(explicit_params) = std::move(params);
            {
                video::IGPUBuffer::SCreationParams creationParams = {};
                creationParams.usage = asset::IBuffer::EUF_STORAGE_BUFFER_BIT;
            
                explicit_params.skinningMatrixBuffer.offset = 0ull;
                explicit_params.skinningMatrixBuffer.size = core::roundUp<size_t>(params.jointCapacity*sizeof(skinning_matrix_t),limits.SSBOAlignment);
                explicit_params.skinningMatrixBuffer.buffer = params.device->createDeviceLocalGPUBufferOnDedMem(creationParams,explicit_params.skinningMatrixBuffer.size);
                explicit_params.jointNodeBuffer.offset = 0ull;
                explicit_params.jointNodeBuffer.size = core::roundUp<size_t>(params.jointCapacity*sizeof(joint_t),limits.SSBOAlignment);
                explicit_params.jointNodeBuffer.buffer = params.device->createDeviceLocalGPUBufferOnDedMem(creationParams,explicit_params.jointNodeBuffer.size);
                explicit_params.recomputedTimestampBuffer.offset = 0ull;
                explicit_params.recomputedTimestampBuffer.size = core::roundUp<size_t>(params.jointCapacity*sizeof(recomputed_stamp_t),limits.SSBOAlignment);
                explicit_params.recomputedTimestampBuffer.buffer = params.device->createDeviceLocalGPUBufferOnDedMem(creationParams,explicit_params.recomputedTimestampBuffer.size);
            }
            return create(std::move(explicit_params),std::move(alloc));
        }
        // you can either construct the allocator with capacity deduced from the memory blocks you pass
		static inline core::smart_refctd_ptr<this_t> create(ExplicitBufferCreationParameters&& params, allocator<uint8_t>&& alloc=allocator<uint8_t>())
		{
            if (!params.device || params.minAllocSizeInJoints==0u || !params.skinningMatrixBuffer.isValid() || !params.jointNodeBuffer.isValid() || !params.recomputedTimestampBuffer.isValid())
                return nullptr;
            
            size_t jointCapacity = invalid;
            jointCapacity = core::min(params.skinningMatrixBuffer.size/sizeof(skinning_matrix_t),jointCapacity);
            jointCapacity = core::min(params.jointNodeBuffer.size/sizeof(joint_t),jointCapacity);
            jointCapacity = core::min(params.recomputedTimestampBuffer.size/sizeof(recomputed_stamp_t),jointCapacity);
            if (jointCapacity==0u)
                return nullptr;
            
			video::IDescriptorPool::SDescriptorPoolSize size = {asset::E_DESCRIPTOR_TYPE::EDT_STORAGE_BUFFER,BufferCount};
			auto dsp = params.device->createDescriptorPool(video::IDescriptorPool::ECF_NONE,1u,1u,&size);
			if (!dsp)
				return nullptr;

            auto layout = ISkinInstanceCache::createDescriptorSetLayout(params.device);
            if (!layout)
                return nullptr;

            auto descriptorSet = params.device->createGPUDescriptorSet(dsp.get(),std::move(layout));
			if (!descriptorSet)
				return nullptr;

            video::IGPUDescriptorSet::SDescriptorInfo infos[BufferCount];
            infos[0] = params.skinningMatrixBuffer;
            infos[1] = params.jointNodeBuffer;
            infos[2] = params.recomputedTimestampBuffer;
            video::IGPUDescriptorSet::SWriteDescriptorSet writes[BufferCount];
			for (auto i=0u; i<BufferCount; i++)
			{
                writes[i].binding = i;
                writes[i].descriptorType = asset::E_DESCRIPTOR_TYPE::EDT_STORAGE_BUFFER;
                writes[i].count = 1u;
				writes[i].dstSet = descriptorSet.get();
				writes[i].arrayElement = 0u;
				writes[i].info = infos+i;
			}
            params.device->updateDescriptorSets(BufferCount,writes,0u,nullptr);

			const auto skinAllocatorReservedSize = computeReservedSize(jointCapacity,params.minAllocSizeInJoints);
			auto skinAllocatorReserved = std::allocator_traits<allocator<uint8_t>>::allocate(alloc,skinAllocatorReservedSize);
			if (!skinAllocatorReserved)
				return nullptr;

			auto* retval = new CSkinInstanceCache(
                params.minAllocSizeInJoints,jointCapacity,skinAllocatorReserved,
                std::move(params.skinningMatrixBuffer),std::move(params.jointNodeBuffer),std::move(params.recomputedTimestampBuffer),
                std::move(descriptorSet),std::move(alloc)
            );
			if (!retval) // TODO: redo this, allocate the memory for the object, if fail, then dealloc, we cannot free from a moved allocator
				std::allocator_traits<allocator<uint8_t>>::deallocate(alloc,skinAllocatorReserved,skinAllocatorReservedSize);

            return core::smart_refctd_ptr<CSkinInstanceCache>(retval,core::dont_grab);
        }

    protected:
        CSkinInstanceCache(
			uint8_t minAllocSizeInJoints, const uint32_t capacity, void* _skinAllocatorReserved,
			asset::SBufferRange<video::IGPUBuffer>&& _skinningMatrixBuffer,
			asset::SBufferRange<video::IGPUBuffer>&& _jointNodeBuffer,
			asset::SBufferRange<video::IGPUBuffer>&& _recomputedTimestampBuffer,
			core::smart_refctd_ptr<video::IGPUDescriptorSet>&& _descriptorSet, allocator<uint8_t>&& _alloc
        ) : ISkinInstanceCache(
                minAllocSizeInJoints,capacity,_skinAllocatorReserved,
                std::move(_skinningMatrixBuffer),std::move(_jointNodeBuffer),std::move(_recomputedTimestampBuffer),
                std::move(_descriptorSet)
            ), m_alloc(std::move(_alloc))
        {
        }
        ~CSkinInstanceCache()
        {
            size_t jointCapacity = invalid;
            jointCapacity = core::min(m_skinningMatrixBlock.size/sizeof(skinning_matrix_t),jointCapacity);
            jointCapacity = core::min(m_jointNodeBlock.size/sizeof(joint_t),jointCapacity);
            jointCapacity = core::min(m_recomputedTimestampBlock.size/sizeof(recomputed_stamp_t),jointCapacity);
            std::allocator_traits<allocator<uint8_t>>::deallocate(
                m_alloc,reinterpret_cast<uint8_t*>(m_skinAllocatorReserved),computeReservedSize(jointCapacity,m_skinAllocator.min_size())
            );
        }

        static inline size_t computeReservedSize(const uint32_t jointCapacity, const uint32_t minAllocSizeInJoints)
        {
            return AddressAllocator::reserved_size(1u,jointCapacity,minAllocSizeInJoints);
        }

        allocator<uint8_t> m_alloc;
};

} // end namespace nbl::scene

#endif

