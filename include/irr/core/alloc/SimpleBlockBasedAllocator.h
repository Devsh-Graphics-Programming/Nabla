// Copyright (C) 2018 Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW Engine"
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_SIMPLE_BLOCK_BASED_ALLOCATOR_H_INCLUDED__
#define __IRR_SIMPLE_BLOCK_BASED_ALLOCATOR_H_INCLUDED__

#include "IrrCompileConfig.h"

#include "irr/core/Types.h"
#include "irr/core/alloc/aligned_allocator.h"
#include "irr/core/alloc/address_allocator_traits.h"
#include "irr/core/alloc/AddressAllocatorConcurrencyAdaptors.h"

#include <memory>

namespace irr
{
namespace core
{

//! Doesn't resize memory arenas, therefore once allocated pointers shall not move
template<class AddressAllocator, template<class> class DataAllocator >
class SimpleBlockBasedAllocator
{
    public:
		using size_type = typename address_allocator_traits<AddressAllocator>::size_type;
		_IRR_STATIC_INLINE_CONSTEXPR size_type meta_alignment = 64u;

        virtual ~SimpleBlockBasedAllocator()
		{
			reset();
			metaAlloc.deallocate(reinterpret_cast<uint8_t*>(blocks),cachedLocalMemSize);
		}

		template<typename... Args>
		SimpleBlockBasedAllocator(size_type _blockSize=4096u*1024u, size_type _maxBlockCount=256u, const Args&... args) :
			blockSize(_blockSize), maxBlockCount(_maxBlockCount), metaAlloc(),
			cachedLocalMemSize(core::alignUp(maxBlockCount*sizeof(uint8_t*),meta_alignment)+address_allocator_traits<AddressAllocator>::reserved_size(meta_alignment,blockSize*maxBlockCount,args...)),
			blocks(reinterpret_cast<uint8_t**>(metaAlloc.allocate(cachedLocalMemSize, meta_alignment))),
			addrAlloc(reinterpret_cast<uint8_t*>(blocks)+core::alignUp(maxBlockCount*sizeof(uint8_t*),meta_alignment), 0u, 0u, meta_alignment, blockSize*maxBlockCount, args...),
			blockAlloc()
		{
			assert(blockSize%meta_alignment==0u);
			assert(address_allocator_traits<AddressAllocator>::get_align_offset(addrAlloc)==0ul);
			assert(address_allocator_traits<AddressAllocator>::get_combined_offset(addrAlloc)==0u);
			std::fill(blocks,blocks+maxBlockCount,nullptr);
		}

		SimpleBlockBasedAllocator& operator=(SimpleBlockBasedAllocator&& other)
        {
			std::swap(blockSize, other.blockSize);
			std::swap(maxBlockCount, other.maxBlockCount);
			std::swap(metaAlloc, other.metaAlloc);
			std::swap(cachedLocalMemSize, other.cachedLocalMemSize);
			std::swap(blocks, other.blocks);
			std::swap(addrAlloc, other.addrAlloc);
            std::swap(blockAlloc,other.blockAlloc);
            return *this;
        }

        inline void		reset()
        {
			addrAlloc.reset();
			for (auto i=0u; i<maxBlockCount; i++)
				deleteBlock(i);
        }



		inline void*	allocate(size_type bytes, size_type alignment) noexcept
		{
			constexpr auto invalid_address = AddressAllocator::invalid_address;

			size_type addr = invalid_address;
			address_allocator_traits<AddressAllocator>::multi_alloc_addr(addrAlloc, 1u, &addr, &bytes, &alignment);
			if (addr==invalid_address)
				return nullptr;

			auto blockID = addr/blockSize;
			auto& block = blocks[blockID];
			if (!block)
				block = createBlock();
			return block+(addr-blockID*blockSize);
		}
		inline void		deallocate(void* p, size_type bytes) noexcept
		{
			for (auto i=0u; i<maxBlockCount; i++)
			{
				if (blocks[i] || p<blocks[i])
					continue;
				size_type addr = p-blocks[i];
				if (addr<blockSize)
				{
					address_allocator_traits<AddressAllocator>::multi_free_addr(addrAlloc, 1u, &addr, &bytes);
					if (address_allocator_traits<AddressAllocator>::get_allocated_size()==size_type(0u))
						deleteBlock(i);
					return;
				}
			}
			assert(false);
		}

		inline bool		operator!=(const SimpleBlockBasedAllocator<AddressAllocator,DataAllocator>& other) const noexcept
		{
			if (blockSize != other.blockSize)
				return true;
			if (maxBlockCount != other.maxBlockCount)
				return true;
			if (metaAlloc != other.metaAlloc)
				return true;
			if (cachedLocalMemSize != other.cachedLocalMemSize)
				return true;
			if (blocks != other.blocks)
				return true;
			if (addrAlloc != other.addrAlloc)
				return true;
			if (blockAlloc != other.blockAlloc)
				return true;
			return false;
		}
		inline bool		operator==(const SimpleBlockBasedAllocator<AddressAllocator,DataAllocator>& other) const noexcept
		{
			return !operator!=(other);
		}
    protected:
		size_type blockSize;
		size_type maxBlockCount;
		DataAllocator<uint8_t> metaAlloc;
		size_t cachedLocalMemSize;
		uint8_t** blocks;
		AddressAllocator addrAlloc;
		DataAllocator<uint8_t> blockAlloc;

		uint8_t* createBlock()
		{
			return blockAlloc.allocate(blockSize, meta_alignment);
		}
		void deleteBlock(uint32_t index)
		{
			blockAlloc.deallocate(blocks[index], blockSize);
			blocks[index] = nullptr;
		}
};


// aliases
template<class AddressAllocator, template<class> class DataAllocator=aligned_allocator>
using SimpleBlockBasedAllocatorST = SimpleBlockBasedAllocator<AddressAllocator, DataAllocator>;

template<class AddressAllocator, template<class> class DataAllocator=aligned_allocator, class RecursiveLockable=std::recursive_mutex>
using SimpleBlockBasedAllocatorMT = AddressAllocatorBasicConcurrencyAdaptor<SimpleBlockBasedAllocator<AddressAllocator, DataAllocator>,RecursiveLockable>;

}
}

#endif // __IRR_CONTIGUOUS_POOL_ADDRESS_ALLOCATOR_H_INCLUDED__


