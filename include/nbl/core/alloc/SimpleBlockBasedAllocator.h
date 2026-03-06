// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_CORE_SIMPLE_BLOCK_BASED_ALLOCATOR_H_INCLUDED_
#define _NBL_CORE_SIMPLE_BLOCK_BASED_ALLOCATOR_H_INCLUDED_


#include "nbl/core/decl/Types.h"
#include "nbl/core/alloc/refctd_memory_resource.h"
#include "nbl/core/alloc/address_allocator_traits.h"
#include "nbl/core/alloc/AddressAllocatorConcurrencyAdaptors.h"

#include "gtl/btree.hpp"

#include <memory>
#include <mutex>


namespace nbl::core
{

//! Does not resize memory arenas, therefore once allocated pointers shall not move
//! Needs an address allocator that takes a Block Size after max alignment parameter
template<class AddressAllocator>
class SimpleBlockBasedAllocator
{
	public:
		using addr_alloc_traits = address_allocator_traits<AddressAllocator>;
		using extra_params_t = tuple_transform_t<impl::identitiy,typename addr_alloc_traits::extra_ctor_param_types>;
		using size_type = typename addr_alloc_traits::size_type;
		// The blocks will be allocated aligned to at least this value
		constexpr static inline size_type meta_alignment = 64u;

	private:
		using this_t = SimpleBlockBasedAllocator<AddressAllocator>;
		// Blocks get allocated/deallocated on demand, inside there's a suballocator.
		// Blocks are always the same size, and obviously can't allocate anything larger than themselves.
		class alignas(16) Block final
		{
				friend class this_t;

				AddressAllocator addrAlloc;

				/* if std::apply doesn't work vause of deduction kicking in https://stackoverflow.com/questions/79893672/using-stdapply-in-a-class-template
				* 
				* use https://www.fluentcpp.com/2021/03/05/stdindex_sequence-and-its-improvement-in-c20/ instead
				template<int ...> struct seq {};
				template<int N, int ...S> struct gens : gens<N - 1, N - 1, S...> { };
				template<int ...S> struct gens<0, S...> { typedef seq<S...> type; };

				template<int ...S>
				void constructBlock(Block* mem,seq<S...>)
				{
					std::construct_at(mem,m_blockSize,std::get<S>(blockCreationArgs)...);
				}
				template<typename... Ts>
				static inline computeReservedSize(const core::tuple<Ts...>& _addrCreationArgs)
				{
					return addr_alloc_traits::reserved_size(meta_alignment,blockSize,std::get<>(_addrCreationArgs)...);
				}
				*/

			public:
				struct SCreationParams
				{
					inline SCreationParams(const size_type _blockSize, const extra_params_t& extraArgs) : addrCreationArgs(extraArgs), blockSize(_blockSize),
						reservedSize(std::apply([_blockSize]<typename... Args>(const Args&... args)->size_type
							{
								return addr_alloc_traits::reserved_size(meta_alignment,_blockSize,args...);
							},addrCreationArgs)
						), totalSize(core::alignUp(sizeof(Block)+reservedSize,meta_alignment)+blockSize) {}

					const extra_params_t addrCreationArgs;
					const size_type blockSize;
					const size_type reservedSize;
					const size_type totalSize;
				};
				//
				inline Block(const SCreationParams& params) : addrAlloc(AddressAllocator())
				{
					std::apply([&]<typename... Args>(const Args&... args)->void
						{
							addrAlloc = AddressAllocator(this+1,/*no address offset*/0u,/*no alignment offset needed*/0u,meta_alignment,params.blockSize,args...);
						},params.addrCreationArgs
					);
					assert(addr_alloc_traits::get_align_offset(addrAlloc) == 0ul);
					assert(addr_alloc_traits::get_combined_offset(addrAlloc) == 0u);
				}

				inline uint8_t* data(const SCreationParams& params) {return reinterpret_cast<uint8_t*>(this)+params.totalSize-params.blockSize;}
				inline const uint8_t* data(const SCreationParams& params) const
				{
					return const_cast<const uint8_t*>(const_cast<Block*>(this)->data(params));
				}
				
				AddressAllocator& getAllocator() {return addrAlloc;}
				const AddressAllocator& getAllocator() const {return addrAlloc;}

				size_type alloc(size_type bytes, size_type alignment)
				{
					size_type addr = AddressAllocator::invalid_address;
					addr_alloc_traits::multi_alloc_addr(addrAlloc,1u,&addr,&bytes,&alignment);
					return addr;
				}

				void free(size_type addr, size_type bytes)
				{
					addr_alloc_traits::multi_free_addr(addrAlloc,1u,&addr,&bytes);
				}
		};

    public:
        virtual inline ~SimpleBlockBasedAllocator()
		{
			reset();
		}

		struct SCreationParams
		{
			smart_refctd_ptr<refctd_memory_resource> mem_resource = nullptr;
			extra_params_t addrAllocCtorExtraParams;
			uint16_t blockSizeKBLog2 : 5 = 7;
			uint16_t initBlockCount : 11 = 1;
		};
		inline SimpleBlockBasedAllocator(SCreationParams&& params) : m_blockCreationParams(1024u<<params.blockSizeKBLog2,std::move(params.addrAllocCtorExtraParams)),
			m_initBlockCount(std::max<size_type>(params.initBlockCount,1)), m_mem_resource(params.mem_resource ? std::move(params.mem_resource):core::getDefaultMemoryResource())
		{
			for (auto i=0u; i<m_initBlockCount; i++)
				m_blocks.insert(createBlock());
		}
		SimpleBlockBasedAllocator(const SimpleBlockBasedAllocator&) = delete;
		SimpleBlockBasedAllocator(SimpleBlockBasedAllocator&&) = delete;

		//
		inline const Block::SCreationParams getBlockCreationParams() const {return m_blockCreationParams;}

		// deallocates everything
        inline void	reset()
        {
			assert(m_blocks.size()>=m_initBlockCount);
			auto it = m_blocks.begin();
			for (auto i=0u; i<m_initBlockCount; i++,it++)
			{
				Block* block = *it;
				if (i<m_initBlockCount)
					block->addrAlloc.reset();
				else
				{
					deleteBlock(block);
					m_blocks.erase(it);
				}
			}
        }

		//
		struct SAllocResult
		{
			explicit inline operator bool() const {return blockData;}
			inline operator void*() const
			{
				if (blockData)
					return blockData+addr;
				return nullptr;
			}

			uint8_t* blockData = nullptr;
			size_type addr : (sizeof(size_type)*8-1);
			size_type newBlock : 1 = false;
		};
		inline SAllocResult allocate(const size_type bytes, const size_type alignment) noexcept
		{
			constexpr auto invalid_address = AddressAllocator::invalid_address;
			// TODO: better allocation strategies like tlsf
			for (auto& entry : m_blocks)
			{
				Block* block = const_cast<Block*>(entry);
				if (const auto addr=block->alloc(bytes,alignment); addr!=invalid_address)
					return {.blockData=block->data(m_blockCreationParams),.addr=addr};
			}
			Block* block = createBlock();
			if (const auto addr=block->alloc(bytes,alignment); addr!=invalid_address)
			{
				m_blocks.insert(block);
				return {.blockData=block->data(m_blockCreationParams),.addr=addr,.newBlock=true};
			}
			else
				deleteBlock(block);
			return {};
		}
		inline void	deallocate(void* p, const size_type bytes) noexcept
		{
			assert(m_blocks.size()>=m_initBlockCount);
			auto found = m_blocks.lower_bound(reinterpret_cast<Block*>(p));
			assert(found!=m_blocks.end());
			auto* block = *found;
			uint8_t* blockData = block->data(m_blockCreationParams);
			assert(blockData<=p && p<blockData+m_blockCreationParams.blockSize);
			const size_type addr = reinterpret_cast<uint8_t*>(p)-blockData;
			block->free(addr,bytes);
			if (m_blocks.size()>m_initBlockCount && addr_alloc_traits::get_allocated_size(block->getAllocator())==size_type(0u))
			{
				deleteBlock(block);
				m_blocks.erase(found);
			}
		}
#if 0 // what do we even need these for
		inline bool	operator!=(const this_t& other) const noexcept
		{
			if (m_blockCreationParams != other.m_blockCreationParams)
				return true;
			if (m_initBlockCount != other.m_initBlockCount)
				return true;
			if (m_mem_resource != other.m_mem_resource)
				return true;
			if (m_blocks != other.m_blocks)
				return true;
			return false;
		}
		inline bool	operator==(const this_t& other) const noexcept
		{
			return !operator!=(other);
		}
#endif
    protected:
		using block_map_t = gtl::btree_set<Block*>;
		
		inline Block* createBlock()
		{
			auto* block = reinterpret_cast<Block*>(m_mem_resource->allocate(m_blockCreationParams.totalSize,meta_alignment));
			std::construct_at(block,m_blockCreationParams);
			m_blocks.insert(block);
			return block;
		}
		inline void deleteBlock(Block* block)
		{
			assert(block);
			std::destroy_at(block);
			m_mem_resource->deallocate(block,m_blockCreationParams.totalSize,meta_alignment);
		}

		const Block::SCreationParams m_blockCreationParams;
		const uint32_t m_initBlockCount;
		smart_refctd_ptr<refctd_memory_resource> m_mem_resource;
		// TODO: better allocation strategies like tlsf
		block_map_t m_blocks;
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

		inline SimpleBlockBasedAllocatorMT(Composed::SCreationParams&& params) : m_composed(std::move(params)), m_lock() {}
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

#if 0 // what do we even need these for
		inline bool		operator!=(const SimpleBlockBasedAllocatorMT<AddressAllocator>& other) const noexcept
		{
			return m_composed!=other.m_composed || other.m_lock!=m_lock;
		}
		inline bool		operator==(const SimpleBlockBasedAllocatorMT<AddressAllocator>& other) const noexcept
		{
			return m_composed==other.m_composed && other.m_lock==m_lock;
		}
#endif
};
// no aliases

}
#endif


