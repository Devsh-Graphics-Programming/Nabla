// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_I_DRAW_INDIRECT_ALLOCATOR_H_INCLUDED_
#define _NBL_VIDEO_I_DRAW_INDIRECT_ALLOCATOR_H_INCLUDED_

#include "nbl/core/alloc/GeneralpurposeAddressAllocator.h"

#include "nbl/video/utilities/CPropertyPool.h"
#include "nbl/video/utilities/CPropertyPoolHandler.h"


namespace nbl::video
{

class IDrawIndirectAllocator : public core::IReferenceCounted
{
    public:
        static inline constexpr auto invalid_draw_count_ix = IPropertyPool::invalid;

        using DrawIndirectAddressAllocator = core::GeneralpurposeAddressAllocator<uint32_t>;
        static inline constexpr auto invalid_draw_range_begin = DrawIndirectAddressAllocator::invalid_address;


        struct CreationParametersBase
        {
            ILogicalDevice* device;
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
        

        //
        inline bool supportsMultiDrawIndirectCount() const
        {
            return !!m_drawCountPool;
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
        const asset::SBufferRange<IGPUBuffer>& getDrawCommandMemoryBlock() const { return m_drawCommandBlock; }
        const asset::SBufferRange<IGPUBuffer>* getDrawCountMemoryBlock() const
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
                // optional, will not be written if `IDrawIndirectAllocator::supportsMultiDrawIndirectCount()` is false
                // if set then must be initialized to `invalid_draw_count_ix`
                uint32_t* multiDrawCommandCountOffsets = nullptr;

                //
                inline uint32_t getCommandStructSize(const uint32_t i) const
                {
                    assert(i<count);
                    if (ptrdiff_t(cmdStride)<(0x1ull<<16ull))
                        return ptrdiff_t(cmdStride);
                    return cmdStride[i];
                }
                //
                inline void setCommandStructSizes(const uint16_t* cmdStructSizes) {cmdStride = cmdStructSizes;}
                inline void setAllCommandStructSizesConstant(const uint16_t cmdStructSizeForAll)
                {
                    reinterpret_cast<ptrdiff_t&>(cmdStride) = cmdStructSizeForAll;
                }
            private:
                // default is for indexed draws
                const uint16_t* cmdStride = (const uint16_t*)sizeof(hlsl::DrawElementsIndirectCommand_t);
        };
        inline bool allocateMultiDraws(Allocation& params)
        {
            for (auto i=0u; i<params.count; i++)
            {
                auto& drawRange = params.multiDrawCommandRangeByteOffsets[i];
                if (drawRange!=invalid_draw_range_begin)
                    continue;

                const uint32_t struct_size = params.getCommandStructSize(i);
                assert(!(struct_size>m_maxDrawCommandStride));
                drawRange = m_drawAllocator.alloc_addr(params.multiDrawCommandMaxCounts[i]*struct_size,struct_size);
                if (drawRange==invalid_draw_range_begin)
                    return false;
            }
            if (m_drawCountPool)
                return m_drawCountPool->allocateProperties(params.multiDrawCommandCountOffsets,params.multiDrawCommandCountOffsets+params.count);
            return true;
        }

        //
        inline void freeMultiDraws(const Allocation& params)
        {
            if (m_drawCountPool)
                m_drawCountPool->freeProperties(params.multiDrawCommandCountOffsets,params.multiDrawCommandCountOffsets+params.count);
            for (auto i=0u; i<params.count; i++)
            {
                auto& drawRange = params.multiDrawCommandRangeByteOffsets[i];
                if (drawRange==invalid_draw_range_begin)
                    continue;
                
                const uint32_t struct_size = params.getCommandStructSize(i);
                assert(!(struct_size>m_maxDrawCommandStride));
                m_drawAllocator.free_addr(drawRange,params.multiDrawCommandMaxCounts[i]*struct_size);
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
        using draw_count_pool_t = CPropertyPool<core::allocator,uint32_t>;

        IDrawIndirectAllocator(core::smart_refctd_ptr<draw_count_pool_t>&& _drawCountPool, const uint16_t _maxDrawCommandStride, asset::SBufferRange<IGPUBuffer>&& _drawCommandBlock, void* _drawAllocatorReserved)
            :   m_drawCountPool(std::move(_drawCountPool)), m_drawAllocator(_drawAllocatorReserved,0u,0u,core::roundUpToPoT<uint32_t>(_maxDrawCommandStride),_drawCommandBlock.size,_maxDrawCommandStride),
                m_drawCommandBlock(std::move(_drawCommandBlock)), m_drawAllocatorReserved(_drawAllocatorReserved), m_maxDrawCommandStride(_maxDrawCommandStride)
        {
        }
        virtual ~IDrawIndirectAllocator() {}

        core::smart_refctd_ptr<draw_count_pool_t> m_drawCountPool;
        DrawIndirectAddressAllocator m_drawAllocator;
        asset::SBufferRange<IGPUBuffer> m_drawCommandBlock;
        void* m_drawAllocatorReserved;
        uint16_t m_maxDrawCommandStride;
};


}

#endif
