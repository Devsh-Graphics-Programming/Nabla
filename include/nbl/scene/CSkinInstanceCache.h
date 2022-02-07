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
    static inline core::smart_refctd_ptr<this_t> create(ImplicitCreationParameters&& params, allocator<uint8_t>&& alloc = allocator<uint8_t>())
    {
        if(!params.isValid())
            return nullptr;

        const auto& limits = params.device->getPhysicalDevice()->getLimits();

        ExplicitCreationParameters explicit_params;
        static_cast<CreationParametersBase&>(explicit_params) = std::move(params);
        {
            video::IGPUBuffer::SCreationParams creationParams = {};
            creationParams.usage = asset::IBuffer::EUF_STORAGE_BUFFER_BIT;

            explicit_params.jointNodeBuffer.offset = 0ull;
            explicit_params.jointNodeBuffer.size = core::roundUp<size_t>(params.jointCapacity * sizeof(joint_t), limits.SSBOAlignment);
            explicit_params.jointNodeBuffer.buffer = params.device->createDeviceLocalGPUBufferOnDedMem(creationParams, explicit_params.jointNodeBuffer.size);
            explicit_params.skinningMatrixBuffer.offset = 0ull;
            explicit_params.skinningMatrixBuffer.size = core::roundUp<size_t>(params.jointCapacity * sizeof(skinning_matrix_t), limits.SSBOAlignment);
            explicit_params.skinningMatrixBuffer.buffer = params.device->createDeviceLocalGPUBufferOnDedMem(creationParams, explicit_params.skinningMatrixBuffer.size);
            explicit_params.recomputedTimestampBuffer.offset = 0ull;
            explicit_params.recomputedTimestampBuffer.size = core::roundUp<size_t>(params.jointCapacity * sizeof(recomputed_stamp_t), limits.SSBOAlignment);
            explicit_params.recomputedTimestampBuffer.buffer = params.device->createDeviceLocalGPUBufferOnDedMem(creationParams, explicit_params.recomputedTimestampBuffer.size);
            explicit_params.inverseBindPoseOffsetBuffer.offset = 0ull;
            explicit_params.inverseBindPoseOffsetBuffer.size = core::roundUp<size_t>(params.jointCapacity * sizeof(inverse_bind_pose_offset_t), limits.SSBOAlignment);
            explicit_params.inverseBindPoseOffsetBuffer.buffer = params.device->createDeviceLocalGPUBufferOnDedMem(creationParams, explicit_params.inverseBindPoseOffsetBuffer.size);
        }
        explicit_params.inverseBindPosePool = video::CPropertyPool<core::allocator, inverse_bind_pose_t>::create(params.device, params.inverseBindPoseCapacity);
        return create(std::move(explicit_params), std::move(alloc));
    }
    // you can either construct the allocator with capacity deduced from the memory blocks you pass
    static inline core::smart_refctd_ptr<this_t> create(ExplicitCreationParameters&& params, allocator<uint8_t>&& alloc = allocator<uint8_t>())
    {
        if(!params.isValid())
            return nullptr;

        size_t jointCapacity = invalid_instance;
        jointCapacity = core::min(params.skinningMatrixBuffer.size / sizeof(skinning_matrix_t), jointCapacity);
        jointCapacity = core::min(params.jointNodeBuffer.size / sizeof(joint_t), jointCapacity);
        jointCapacity = core::min(params.recomputedTimestampBuffer.size / sizeof(recomputed_stamp_t), jointCapacity);
        jointCapacity = core::min(params.inverseBindPoseOffsetBuffer.size / sizeof(recomputed_stamp_t), jointCapacity);
        if(jointCapacity == 0u)
            return nullptr;

        const auto skinAllocatorReservedSize = computeReservedSize(jointCapacity, params.minAllocSizeInJoints);
        auto skinAllocatorReserved = std::allocator_traits<allocator<uint8_t>>::allocate(alloc, skinAllocatorReservedSize);
        if(!skinAllocatorReserved)
            return nullptr;

        const auto* transformTree = params.associatedTransformTree.get();
        assert(transformTree->getRenderDescriptorSetBindingCount() <= ITransformTreeWithNormalMatrices::RenderDescriptorSetBindingCount);
        const auto poolSizeCount = CacheDescriptorSetBindingCount + ITransformTreeWithNormalMatrices::RenderDescriptorSetBindingCount + 1u;
        video::IDescriptorPool::SDescriptorPoolSize size = {asset::E_DESCRIPTOR_TYPE::EDT_STORAGE_BUFFER, poolSizeCount};
        auto dsp = params.device->createDescriptorPool(video::IDescriptorPool::ECF_NONE, 2u, 1u, &size);
        if(!dsp)
            return nullptr;

        video::IGPUDescriptorSet::SWriteDescriptorSet writes[CacheDescriptorSetBindingCount];
        static_assert(ITransformTreeWithNormalMatrices::RenderDescriptorSetBindingCount < CacheDescriptorSetBindingCount);
        for(auto i = 0u; i < CacheDescriptorSetBindingCount; i++)
        {
            writes[i].binding = i;
            writes[i].descriptorType = asset::E_DESCRIPTOR_TYPE::EDT_STORAGE_BUFFER;
            writes[i].count = 1u;
        }
        auto cacheUpdateLayout = createCacheDescriptorSetLayout(params.device);
        core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> renderLayout;
        if(transformTree->hasNormalMatrices())
            renderLayout = createRenderDescriptorSetLayout<ITransformTreeWithNormalMatrices>(params.device);
        else
            renderLayout = createRenderDescriptorSetLayout<ITransformTreeWithoutNormalMatrices>(params.device);
        if(!cacheUpdateLayout || !renderLayout)
            return false;

        auto cacheUpdateDescriptorSet = params.device->createGPUDescriptorSet(dsp.get(), std::move(cacheUpdateLayout));
        auto renderDescriptorSet = params.device->createGPUDescriptorSet(dsp.get(), std::move(renderLayout));
        if(!cacheUpdateDescriptorSet || !renderDescriptorSet)
            return nullptr;

        const video::IPropertyPool* nodePP = transformTree->getNodePropertyPool();

        video::IGPUDescriptorSet::SDescriptorInfo infos[CacheDescriptorSetBindingCount];
        infos[0] = params.jointNodeBuffer;
        infos[1] = params.skinningMatrixBuffer;
        infos[2] = params.recomputedTimestampBuffer;
        static_assert(ITransformTree::global_transform_prop_ix == 3u);
        infos[3] = nodePP->getPropertyMemoryBlock(ITransformTree::global_transform_prop_ix);
        static_assert(ITransformTree::recomputed_stamp_prop_ix == 4u);
        infos[4] = nodePP->getPropertyMemoryBlock(ITransformTree::recomputed_stamp_prop_ix);
        infos[5] = params.inverseBindPoseOffsetBuffer;
        infos[6] = params.inverseBindPosePool->getPropertyMemoryBlock(inverse_bind_pose_prop_ix);
        for(auto i = 0u; i < CacheDescriptorSetBindingCount; i++)
        {
            writes[i].dstSet = cacheUpdateDescriptorSet.get();
            writes[i].arrayElement = 0u;
            writes[i].info = infos + i;
        }
        params.device->updateDescriptorSets(CacheDescriptorSetBindingCount, writes, 0u, nullptr);
        infos[0] = nodePP->getPropertyMemoryBlock(ITransformTree::global_transform_prop_ix);
        if(transformTree->hasNormalMatrices())
            infos[2] = nodePP->getPropertyMemoryBlock(ITransformTreeWithNormalMatrices::normal_matrix_prop_ix);
        const auto renderDescriptorSetBindingCount = transformTree->getRenderDescriptorSetBindingCount() + 1u;
        for(auto i = 0u; i < renderDescriptorSetBindingCount; i++)
            writes[i].dstSet = renderDescriptorSet.get();
        params.device->updateDescriptorSets(renderDescriptorSetBindingCount, writes, 0u, nullptr);

        auto* retval = new CSkinInstanceCache(
            params.minAllocSizeInJoints, jointCapacity, skinAllocatorReserved,
            std::move(params.jointNodeBuffer), std::move(params.skinningMatrixBuffer), std::move(params.recomputedTimestampBuffer),
            std::move(params.associatedTransformTree), std::move(params.inverseBindPoseOffsetBuffer), std::move(params.inverseBindPosePool),
            std::move(cacheUpdateDescriptorSet), std::move(renderDescriptorSet), std::move(alloc));
        if(!retval)  // TODO: redo this, allocate the memory for the object, if fail, then dealloc, we cannot free from a moved allocator
            std::allocator_traits<allocator<uint8_t>>::deallocate(alloc, skinAllocatorReserved, skinAllocatorReservedSize);

        return core::smart_refctd_ptr<CSkinInstanceCache>(retval, core::dont_grab);
    }

protected:
    CSkinInstanceCache(
        uint8_t minAllocSizeInJoints, const uint32_t capacity, void* _skinAllocatorReserved,
        asset::SBufferRange<video::IGPUBuffer>&& _jointNodeBuffer,
        asset::SBufferRange<video::IGPUBuffer>&& _skinningMatrixBuffer,
        asset::SBufferRange<video::IGPUBuffer>&& _recomputedTimestampBuffer,
        core::smart_refctd_ptr<ITransformTree>&& _tt,
        asset::SBufferRange<video::IGPUBuffer>&& _inverseBindPoseOffsetBuffer,
        core::smart_refctd_ptr<video::IPropertyPool>&& _inverseBindPosePool,
        core::smart_refctd_ptr<video::IGPUDescriptorSet>&& _cacheDS,
        core::smart_refctd_ptr<video::IGPUDescriptorSet>&& _renderDS,
        allocator<uint8_t>&& _alloc)
        : ISkinInstanceCache(
              minAllocSizeInJoints, capacity, _skinAllocatorReserved,
              std::move(_jointNodeBuffer), std::move(_skinningMatrixBuffer), std::move(_recomputedTimestampBuffer),
              std::move(_tt), std::move(_inverseBindPoseOffsetBuffer), std::move(_inverseBindPosePool),
              std::move(_cacheDS), std::move(_renderDS)),
          m_alloc(std::move(_alloc))
    {
    }
    ~CSkinInstanceCache()
    {
        size_t jointCapacity = invalid_instance;
        jointCapacity = core::min(m_skinningMatrixBlock.size / sizeof(skinning_matrix_t), jointCapacity);
        jointCapacity = core::min(m_jointNodeBlock.size / sizeof(joint_t), jointCapacity);
        jointCapacity = core::min(m_recomputedTimestampBlock.size / sizeof(recomputed_stamp_t), jointCapacity);
        jointCapacity = core::min(m_inverseBindPoseOffsetBlock.size / sizeof(recomputed_stamp_t), jointCapacity);
        std::allocator_traits<allocator<uint8_t>>::deallocate(
            m_alloc, reinterpret_cast<uint8_t*>(m_skinAllocatorReserved), computeReservedSize(jointCapacity, m_skinAllocator.min_size()));
    }

    static inline size_t computeReservedSize(const uint32_t jointCapacity, const uint32_t minAllocSizeInJoints)
    {
        return AddressAllocator::reserved_size(1u, jointCapacity, minAllocSizeInJoints);
    }

    allocator<uint8_t> m_alloc;
};

}  // end namespace nbl::scene

#endif
