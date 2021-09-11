// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_VIDEO_C_DRAW_INDIRECT_ALLOCATOR_H_INCLUDED_
#define _NBL_VIDEO_C_DRAW_INDIRECT_ALLOCATOR_H_INCLUDED_

#include "nbl/video/utilities/CPropertyPool.h"


namespace nbl::video
{

    
template<template<class...> class allocator=core::allocator>
class CDrawIndirectAllocator : public core::IReferenceCounted
{
        using this_t = CDrawIndirectAllocator<allocator>;
        using draw_count_pool_t = CPropertyPool<uint32_t>;

        static inline constexpr auto MinAllocationBlock = 512u;

    public:
        static inline constexpr auto invalid_draw_count_ix = IPropertyPool::invalid;

        using DrawIndirectAddressAllocator = core::GeneralPurposeAddressAllocatorST<uint32_t>;
        static inline constexpr auto invalid_draw_range_begin = DrawIndirectAddressAllocator::invalid_address;


        struct CreationParametersBase
        {
            allocator<uint8_t> alloc = allocator<uint8_t>();
            const ILogicalDevice* device;
            uint16_t maxDrawCommandStride;
        };
        struct ImplicitBufferCreationParameters : CreationParametersBase
        {
            uint32_t drawCommandCapacity;
            uint32_t drawCountCapacity;
        };
        struct ExplicitBufferCreationParameters : CreationParametersBase
        {
            asset::SBufferRange<video::IGPUBuffer> drawCommandBuffer;
            asset::SBufferRange<video::IGPUBuffer> drawCountBuffer;
        };

        // easy dont care creation
        static inline core::smart_refctd_ptr<this_t> create(ImplicitBufferCreationParameters&& params)
        {
            if (!params.device || params.drawCommandCapacity==0u)
                return nullptr;

            const auto& limits = params.device->getPhysicalDevice()->getLimits();
            if (limits.drawIndirectCount)
                params.drawCountCapacity = 0;
            
            ExplicitBufferCreationParameters explicit_params;
            static_cast<CreationParametersBase&>(explicit_params) = std::move(params);
            explicit_params.drawCommandBuffer.offset = 0ull;
            explicit_params.drawCommandBuffer.size = core::roundUp(params.drawCommandCapacity*params.maxDrawCommandStride,limits.SSBOAlignment);
            explicit_params.drawCommandBuffer.buffer = device->createDeviceLocalGPUBufferOnDedMem(explicit_params.drawCommandBuffer.size);
            explicit_params.drawCountBuffer.offset = 0ull;
            explicit_params.drawCountBuffer.size = core::roundUp(params.drawCountCapacity*sizeof(uint32_t),limits.SSBOAlignment);
            if (explicit_params.drawCountBuffer.size)
                explicit_params.drawCountBuffer.buffer = device->createDeviceLocalGPUBufferOnDedMem(explicit_params.drawCountBuffer.size);
            else
                explicit_params.drawCountBuffer.buffer = nullptr;
            return create(std::move(explicit_params));
        }
        // you can either construct the allocator with capacity deduced from the memory blocks you pass
		static inline core::smart_refctd_ptr<this_t> create(ExplicitBufferCreationParameters&& params)
		{
            const auto drawCapacity = params.drawCommandBuffer.size/params.maxDrawCommandStride;
            if (!params.device || !params.drawCommandBuffer.buffer || drawCapacity==0u)
                return nullptr;
            
            auto drawCountPool = draw_count_pool_t::create(device,&params.drawCountBuffer);
            if (params.drawCountBuffer.size!=0u && !drawCountPool)
                return nullptr;

			const auto drawAllocatorReservedSize = DrawIndirectAddressAllocator::reserved_size(params.maxDrawCommandStride,params.drawCommandBuffer.size,MinAllocationBlock);
			auto drawAllocatorReserved = std::allocator_traits<allocator<uint8_t>>::allocate(alloc,drawAllocatorReservedSize);
			if (!drawAllocatorReserved)
				return nullptr;

			auto* retval = new CDrawIndirectAllocator(std::move(drawCountPool),maxDrawCommandStride,std::move(params.drawCommandBuffer),drawAllocatorReserved,std::move(alloc));
			if (!retval)
				std::allocator_traits<allocator<uint8_t>>::deallocate(alloc,reserved,reservedSize);

            return core::smart_refctd_ptr<CDrawIndirectAllocator>(retval,core::dont_grab);
        }
        

        //
        inline bool supportsMultiDrawIndirectCount() const
        {
            return m_drawCountPool;
        }

        //
        inline uint32_t getAllocatedCommandBytes() const
        {
            return m_drawAllocator.get_allocated_size();
        }
        inline uint32_t getFreeCommandBytes() const
        {
            return m_drawAllocator.get_free_size();
        }
        inline uint32_t getCapacityCommandBytes() const
        {
            // special case allows us to use `get_total_size`, because the pool allocator has no added offsets
            return m_drawAllocator.get_total_size();
        }
        //
        inline uint32_t getAllocatedDrawCounts() const
        {
            if (m_drawCountPool)
                return m_drawCountPool->getAllocated();
            return 0u;
        }
        inline uint32_t getFreeDrawCounts() const
        {
            if (m_drawCountPool)
                return m_drawCountPool->getFree();
            return 0u;
        }
        inline uint32_t getCapacityDrawCounts() const
        {
            if (m_drawCountPool)
                return m_drawCountPool->getCapacity();
            return 0u;
        }

        //
        const asset::SBufferRange<IGPUBuffer>& getDrawCommandMemoryBlock() const override { return m_drawCommandBlock; }
        const asset::SBufferRange<IGPUBuffer>* getDrawCountMemoryBlock() const override
        {
            if (m_drawCountPool)
                return &m_drawCountPool->getPropertyMemoryBlock(0u);
            return nullptr;
        }

        //
        struct Allocation
        {
                uint32_t count = 0u;
                // must point to an array initialized with `invalid_draw_range_begin`
                uint32_t* multiDrawCommandRangeByteOffsets;
                const uint32_t* multiDrawCommandMaxCounts;
                // optional, will not be written if `CDrawIndirectAllocator::supportsMultiDrawIndirectCount()` is false
                // if set then must be initialized to `invalid_draw_count_ix`
                uint32_t* multiDrawCommandCounts = nullptr;

                //
                inline uint32_t getCommandStructSize(const uint32_t i) const
                {
                    assert(i<count);
                    if (cmdStride<(0x1ull<<16ull))
                        return ptrdiff_t(cmdStride);
                    return cmdStride[i];
                }
                //
                inline setCommandStructSizes(const uint16_t* cmdStructSizes) {cmdStride = cmdStructSizes;}
                inline setAllCommandStructSizesConstant(const uint16_t cmdStructSizeForAll)
                {
                    reinterpret_cast<ptrdiff_t&>(cmdStride) = cmdStructSizeForAll;
                }
            private:
                // default is for indexed draws
                const uint16_t* cmdStride = (const uint16_t*)sizeof(asset::DrawElementsIndirectCommand_t);
        };
        inline bool allocateMultiDraws(Allocation& params)
        {
            for (auto i=0u; i<params.count; i++)
            {
                auto& drawRange = params.multiDrawCommandRangeByteOffsets[i];
                if (drawRange!=invalid_draw_range_begin)
                    continue;

                const uint32_t struct_size = params.getCommandStructSize(i);
                assert(!(size>m_maxDrawCommandStride));
                drawRange = m_drawAllocator.alloc_addr(params.count*struct_size,struct_size);
                if (drawRange!=invalid_draw_range_begin)
                    return false;
            }
            if (m_drawCountPool)
                return m_drawCountPool->allocateProperties(params.multiDrawCommandCounts,params.multiDrawCommandCounts+params.count);
            return true;
        }

        //
        inline void freeMultiDraws(const Allocation& params)
        {
            if (m_drawCountPool)
                m_drawCountPool->freeProperties(params.multiDrawCommandCounts,params.multiDrawCommandCounts+params.count);
            for (auto i=0u; i<params.count; i++)
            {
                auto& drawRange = params.multiDrawCommandRangeByteOffsets[i];
                if (drawRange==invalid_draw_range_begin)
                    continue;
                
                const uint32_t struct_size = params.getCommandStructSize(i);
                assert(!(size>m_maxDrawCommandStride));
                m_drawAllocator.free_addr(params.count*struct_size,struct_size);
            }
        }

        //
        inline void clear()
        {
            m_drawAllocator.reset();
            if (m_drawCountPool)
                m_drawCountPool->freeAllProperties();
        }

    protected:
        CDrawIndirectAllocator(core::smart_refctd_ptr<draw_count_pool_t>&& _drawCountPool, const uint16_t _maxDrawCommandStride, asset::SBufferRange<IGPUBuffer>&& _drawCommandBlock, void* _drawAllocatorReserved, allocator<uint8_t>&& _alloc)
            : m_drawCountPool(std::move(_drawCountPool)), m_drawAllocator(_drawAllocatorReserved,0u,0u,_maxDrawCommandStride,_drawCommandBlock.size,MinAllocationBlock),
            m_alloc(std::move(_alloc)), m_drawCommandBlock(std::move(_drawCommandBlock)), m_drawAllocatorReserved(_drawAllocatorReserved), m_maxDrawCommandStride(_maxDrawCommandStride)
        {
        }
        ~CDrawIndirectAllocator()
        {
            std::allocator_traits<allocator<uint8_t>>::deallocate(
                m_alloc,reinterpret_cast<uint8_t*>(m_drawAllocatorReserved),
                DrawIndirectAddressAllocator::reserved_size(sizeof(LargestIndirectCommand_t),m_drawCommandBlock.size,MinAllocationBlock)
            );
        }

        core::smart_refctd_ptr<draw_count_pool_t> m_drawCountPool;
        DrawIndirectAddressAllocator m_drawAllocator;
        allocator<uint8_t> m_alloc;
        asset::SBufferRange<IGPUBuffer> m_drawCommandBlock;
        void* m_drawAllocatorReserved;
        uint16_t m_maxDrawCommandStride;
};


}

#endif