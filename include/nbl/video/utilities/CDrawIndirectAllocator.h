// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_VIDEO_C_DRAW_INDIRECT_ALLOCATOR_H_INCLUDED_
#define _NBL_VIDEO_C_DRAW_INDIRECT_ALLOCATOR_H_INCLUDED_


#include "nbl/video/utilities/IDrawIndirectAllocator.h"


namespace nbl::video
{

template<template<class...> class allocator=core::allocator>
class CDrawIndirectAllocator final : public IDrawIndirectAllocator
{
        using this_t = CDrawIndirectAllocator<allocator>;

    public:

        static void enableRequiredFeautres(SPhysicalDeviceFeatures& featuresToEnable)
        {
        }

        static void enablePreferredFeatures(const SPhysicalDeviceFeatures& availableFeatures, SPhysicalDeviceFeatures& featuresToEnable)
        {
        }

        // easy dont care creation
        static inline core::smart_refctd_ptr<this_t> create(ImplicitBufferCreationParameters&& params, allocator<uint8_t>&& alloc=allocator<uint8_t>())
        {
            if (!params.device || params.drawCommandCapacity==0u)
                return nullptr;

            const auto& limits = params.device->getPhysicalDevice()->getLimits();
            if (!limits.drawIndirectCount)
                params.drawCountCapacity = 0;
            
            ExplicitBufferCreationParameters explicit_params;

            video::IGPUBuffer::SCreationParams creationParams = {};
            creationParams.usage = asset::IBuffer::EUF_STORAGE_BUFFER_BIT;
            creationParams.usage |= asset::IBuffer::EUF_INDIRECT_BUFFER_BIT;

            static_cast<CreationParametersBase&>(explicit_params) = std::move(params);
            explicit_params.drawCommandBuffer.offset = 0ull;
            // need to add a little padding, because generalpurpose allocator doesnt allow for allocations that would leave freeblocks smaller than the minimum allocation size
            explicit_params.drawCommandBuffer.size = core::roundUp<size_t>(params.drawCommandCapacity*params.maxDrawCommandStride+params.maxDrawCommandStride,limits.minSSBOAlignment);

            creationParams.size = explicit_params.drawCommandBuffer.size;
            explicit_params.drawCommandBuffer.buffer = params.device->createBuffer(std::move(creationParams));
            auto mreqsDrawCmdBuf = explicit_params.drawCommandBuffer.buffer->getMemoryReqs();
            mreqsDrawCmdBuf.memoryTypeBits &= params.device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
            auto gpubufMem = params.device->allocate(mreqsDrawCmdBuf, explicit_params.drawCommandBuffer.buffer.get());

            explicit_params.drawCountBuffer.offset = 0ull;
            explicit_params.drawCountBuffer.size = core::roundUp<size_t>(params.drawCountCapacity*sizeof(uint32_t),limits.minSSBOAlignment);
            if (explicit_params.drawCountBuffer.size)
            {
                creationParams.size = explicit_params.drawCountBuffer.size;
                explicit_params.drawCountBuffer.buffer = params.device->createBuffer(std::move(creationParams));
                auto mreqsDrawCountBuf = explicit_params.drawCountBuffer.buffer->getMemoryReqs();
                mreqsDrawCountBuf.memoryTypeBits &= params.device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
                auto gpubufMem = params.device->allocate(mreqsDrawCountBuf, explicit_params.drawCountBuffer.buffer.get());
            }
            else
                explicit_params.drawCountBuffer.buffer = nullptr;
            return create(std::move(explicit_params),std::move(alloc));
        }
        // you can either construct the allocator with capacity deduced from the memory blocks you pass
		static inline core::smart_refctd_ptr<this_t> create(ExplicitBufferCreationParameters&& params, allocator<uint8_t>&& alloc = allocator<uint8_t>())
		{
            const auto drawCapacity = params.drawCommandBuffer.size/params.maxDrawCommandStride;
            if (!params.device || !params.drawCommandBuffer.isValid() || drawCapacity==0u)
                return nullptr;
            
            auto drawCountPool = draw_count_pool_t::create(params.device,&params.drawCountBuffer);
            if (params.drawCountBuffer.size!=0u && !drawCountPool)
                return nullptr;

			const auto drawAllocatorReservedSize = computeReservedSize(params.drawCommandBuffer.size,params.maxDrawCommandStride);
			auto drawAllocatorReserved = std::allocator_traits<allocator<uint8_t>>::allocate(alloc,drawAllocatorReservedSize);
			if (!drawAllocatorReserved)
				return nullptr;

			auto* retval = new CDrawIndirectAllocator(std::move(drawCountPool),params.maxDrawCommandStride,std::move(params.drawCommandBuffer),drawAllocatorReserved,std::move(alloc));
			if (!retval) // TODO: redo this, allocate the memory for the object, if fail, then dealloc, we cannot free from a moved allocator
				std::allocator_traits<allocator<uint8_t>>::deallocate(alloc,drawAllocatorReserved,drawAllocatorReservedSize);

            return core::smart_refctd_ptr<CDrawIndirectAllocator>(retval,core::dont_grab);
        }

    protected:
        CDrawIndirectAllocator(core::smart_refctd_ptr<draw_count_pool_t>&& _drawCountPool, const uint16_t _maxDrawCommandStride, asset::SBufferRange<IGPUBuffer>&& _drawCommandBlock, void* _drawAllocatorReserved, allocator<uint8_t>&& _alloc)
            : IDrawIndirectAllocator(std::move(_drawCountPool),_maxDrawCommandStride,std::move(_drawCommandBlock),_drawAllocatorReserved), m_alloc(std::move(_alloc))
        {
        }
        ~CDrawIndirectAllocator()
        {
            std::allocator_traits<allocator<uint8_t>>::deallocate(
                m_alloc,reinterpret_cast<uint8_t*>(m_drawAllocatorReserved),computeReservedSize(m_drawCommandBlock.size,m_maxDrawCommandStride)
            );
        }

        static inline size_t computeReservedSize(const uint64_t bufferSize, const uint32_t maxStride)
        {
            return DrawIndirectAddressAllocator::reserved_size(core::roundUpToPoT(maxStride),bufferSize,maxStride);
        }

        allocator<uint8_t> m_alloc;
};

}

#endif