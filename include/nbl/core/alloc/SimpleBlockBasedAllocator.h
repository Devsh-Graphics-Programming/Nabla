// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_CORE_SIMPLE_BLOCK_BASED_ALLOCATOR_H_INCLUDED_
#define _NBL_CORE_SIMPLE_BLOCK_BASED_ALLOCATOR_H_INCLUDED_


#include "nbl/core/decl/Types.h"
#include "nbl/core/alloc/refctd_memory_resource.h"
#include "nbl/core/alloc/address_allocator_traits.h"
#include "nbl/core/alloc/AddressAllocatorConcurrencyAdaptors.h"

#include <memory>
#include <mutex>


namespace nbl::core
{

//! Does not resize memory arenas, therefore once allocated pointers shall not move
//! Needs an address allocator that takes a Block Size after max alignment parameter
template<class AddressAllocator>
class SimpleBlockBasedAllocator final
{
	public:
		using addr_alloc_traits = address_allocator_traits<AddressAllocator>;
		using size_type = typename addr_alloc_traits::size_type;
		// The blocks will be allocated aligned to at least this value
		constexpr static inline size_type meta_alignment = 64u;

	private:
		using extra_params_t = tuple_transform_t<std::type_identity,addr_alloc_traits::extra_ctor_param_types>;
		// Blocks get allocated/deallocated on demand, inside there's a suballocator.
		// Blocks are always the same size, and obviously can't allocate anything larger than themselves.
		class Block
		{
				AddressAllocator addrAlloc;

			public:
				template<typename... Args>
				inline Block(const size_type blockSize, const Args&... args) :
					addrAlloc(AddressAllocator(data()+blockSize,./*no address offset*/0u,/*no alignment offset needed*/0u,meta_alignment,blockSize,args...))
				{
					assert(addr_alloc_traits::get_align_offset(addrAlloc) == 0ul);
					assert(addr_alloc_traits::get_combined_offset(addrAlloc) == 0u);
				}

				template<typename... Args>
				static size_type size_of(size_type blockSize, const Args&... args)
				{
					return core::alignUp(sizeof(AddressAllocator),meta_alignment)+blockSize+addr_alloc_traits::reserved_size(meta_alignment,blockSize,args...);
				}

				uint8_t* data() { return reinterpret_cast<uint8_t*>(this)+core::alignUp(sizeof(AddressAllocator),meta_alignment); }
				const uint8_t* data() const
				{
					return const_cast<Block*>(this)->data();
				}
				
				const AddressAllocator& getAllocator() const { return addrAlloc; }

				size_type alloc(size_type bytes, size_type alignment)
				{
					size_type addr = AddressAllocator::invalid_address;
					addr_alloc_traits::multi_alloc_addr(addrAlloc, 1u, &addr, &bytes, &alignment);
					return addr;
				}

				void free(size_type addr, size_type bytes)
				{
					addr_alloc_traits::multi_free_addr(addrAlloc, 1u, &addr, &bytes);
				}
		};

    public:
        virtual inline ~SimpleBlockBasedAllocator()
		{
			reset();
			metaAlloc.deallocate(blocks,maxBlockCount);
		}

		template<typename... Args>
		inline SimpleBlockBasedAllocator(smart_refctd_ptr<refctd_memory_resource>&& _mem_resource, const size_type _blockSize, const size_type _initBlockCount, Args&&... args) :
			m_blockCreationArgs(std::forward<Args>(args)...), m_mem_resource(std::move(_mem_resource)), m_blockSize(_blockSize), m_effectiveBlockSize(Block::size_of(m_blockSize,m_blockCreationArgs))
		{
// TODO: block init
			m_initBlockCount = std::max<size_type>(_initBlockCount,1));
			blocks = metaAlloc.allocate(maxBlockCount,meta_alignment))
			std::fill(blocks,blocks+maxBlockCount,nullptr);
		}

		inline SimpleBlockBasedAllocator& operator=(SimpleBlockBasedAllocator&& other)
        {
			std::swap(m_blockCreationArgs,other.m_blockCreationArgs);
			std::swap(m_mem_resource,other.m_mem_resource);
			std::swap(m_blockSize,other.m_blockSize);
			std::swap(m_effectiveBlockSize,other.m_effectiveBlockSize);
// TODO: block swap
            return *this;
        }
		inline SimpleBlockBasedAllocator(SimpleBlockBasedAllocator<AddressAllocator>&& other)
		{
			operator=(std::move(other));
		}

        inline void		reset()
        {
// TODO: reset
			for (auto i=minBlockCount; i<maxBlockCount; i++)
				deleteBlock(i);
        }


		inline void*	allocate(const size_type bytes, const size_type alignment) noexcept
		{
// TODO: rewrite
			constexpr auto invalid_address = AddressAllocator::invalid_address;
			for (size_type i=0u; i<maxBlockCount; i++)
			{
				auto& block = blocks[i];

				bool die = i==(maxBlockCount-1u);
				if (!block)
				{
					block = createBlock();
					die = true;
				}

				size_type addr = block->alloc(bytes, alignment);
				if (addr == invalid_address)
				{
					if (die)
						break;
					else
						continue;
				}

				return block->data()+addr;
			}
			return nullptr;
		}
		inline void		deallocate(void* p, const size_type bytes) noexcept
		{
// TODO: rewrite
			for (size_type i=0u; i<maxBlockCount; i++)
			{
				auto& block = blocks[i];
				if (!block)
					continue;
                    
				size_type addr = reinterpret_cast<uint8_t*>(p)-block->data();
				if (addr<blockSize)
				{
					block->free(addr,bytes);
					if (i>=minBlockCount && addr_alloc_traits::get_allocated_size(block->getAllocator())==size_type(0u))
						deleteBlock(i);
					return;
				}
			}
			assert(false);
		}

		inline bool		operator!=(const SimpleBlockBasedAllocator<AddressAllocator,DataAllocator>& other) const noexcept
		{
// TODO: rewrite
			if (blockSize != other.blockSize)
				return true;
			if (effectiveBlockSize != other.effectiveBlockSize)
				return true;
			if (minBlockCount != other.minBlockCount)
				return true;
			if (maxBlockCount != other.maxBlockCount)
				return true;
			if (metaAlloc != other.metaAlloc)
				return true;
			if (blocks != other.blocks)
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
		extra_params_t m_blockCreationArgs;
		smart_refctd_ptr<refctd_memory_resource> m_mem_resource;
		size_type m_blockSize;
		size_type m_effectiveBlockSize;
		// TODO: rewrite this
size_type m_initBlockCount;
Block** blocks;


		template<int ...> struct seq {};
		template<int N, int ...S> struct gens : gens<N - 1, N - 1, S...> { };
		template<int ...S> struct gens<0, S...> { typedef seq<S...> type; };

		template<int ...S>
		void constructBlock(Block* mem,seq<S...>)
		{
			std::construct_at(mem,m_blockSize,std::get<S>(blockCreationArgs)...);
		}
		Block* createBlock()
		{
			auto retval = reinterpret_cast<Block*>(blockAlloc.allocate(effectiveBlockSize, meta_alignment));
			constructBlock(retval,typename gens<sizeof...(Args)>::type());
			return retval;
		}


		void deleteBlock(uint32_t index)
		{
			if (!blocks[index])
				return;

			blocks[index]->~Block();
			blockAlloc.deallocate(reinterpret_cast<uint8_t*>(blocks[index]),effectiveBlockSize);
			blocks[index] = nullptr;
		}
};

template<class AddressAllocator>
using SimpleBlockBasedAllocatorST = SimpleBlockBasedAllocator<AddressAllocator>;


template<class Composed, class RecursiveLockable=std::recursive_mutex> requires std::is_base_of_v<SimpleBlockBasedAllocator<typename Composed::addr_alloc_traits::allocator_type>,Composed>
class SimpleBlockBasedAllocatorMT final
{
		using this_t = SimpleBlockBasedAllocatorMT<Composed,RecursiveLockable>;

	protected:
		Composed m_composed;
        RecursiveLockable m_lock;

	public:
        using size_type = typename Composed::size_type;

		template<typename... Args>
		inline SimpleBlockBasedAllocatorMT(size_type _blockSize, size_type _minBlockCount, size_type _maxBlockCount, Args&&... args) :
			m_composed(_blockSize,_minBlockCount,_maxBlockCount,std::forward<Args>(args)...), m_lock() {}

		inline auto& operator=(this_t&& other)
        {
			// TODO: lock both locks till complete?
			std::swap(m_lock,other.m_lock);
			std::swap(m_composed,other.m_composed);
			return *this;
        }

		inline SimpleBlockBasedAllocatorMT(this_t&& other)
		{
			operator=(std::move(other));
		}

        virtual inline ~SimpleBlockBasedAllocatorMT() {}

		//
        inline void		reset()
        {
			m_lock.lock();
			m_composed.reset();
			m_lock.unlock();
        }

		inline void*	allocate(size_type bytes, size_type alignment) noexcept
		{
			m_lock.lock();
			auto ret = m_composed.allocate(bytes, alignment);
			m_lock.unlock();
			return ret;
		}
		inline void		deallocate(void* p, size_type bytes) noexcept
		{
			m_lock.lock();
			m_composed.deallocate(p, bytes);
			m_lock.unlock();
		}

		//! Extra == Use WITH EXTREME CAUTION
		inline RecursiveLockable&   get_lock() noexcept
		{
			return m_lock;
		}
		
		inline bool		operator!=(const SimpleBlockBasedAllocatorMT<AddressAllocator>& other) const noexcept
		{
			return m_composed!=other.m_composed || other.m_lock!=m_lock;
		}
		inline bool		operator==(const SimpleBlockBasedAllocatorMT<AddressAllocator>& other) const noexcept
		{
			return m_composed==other.m_composed && other.m_lock==m_lock;
		}
};
// no aliases

}
#endif


