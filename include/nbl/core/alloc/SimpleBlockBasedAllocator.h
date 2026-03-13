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
	
template<typename HandleValue, HandleValue _Invalid> requires (std::is_integral_v<HandleValue>&& std::is_unsigned_v<HandleValue>)
struct ConstHandle
{
	using value_t = HandleValue;
	constexpr static inline value_t Invalid = _Invalid;

	explicit inline operator bool() const {return value!=Invalid;}
	// God, I love C++20
	inline auto operator<=>(const ConstHandle&) const = default;

	// LSB is the offset in the block, MSB is the block index
	value_t value = Invalid;
};
template<typename _ConstHandle> requires std::is_same_v<ConstHandle<typename _ConstHandle::value_t,_ConstHandle::Invalid>,_ConstHandle>
struct Handle : _ConstHandle
{
	using const_type = _ConstHandle;

	inline auto operator<=>(const Handle& other) const {return _ConstHandle::operator<=>(other);}
};

template<typename T, typename _Handle> requires std::is_same_v<Handle<ConstHandle<typename _Handle::value_t,_Handle::Invalid>>,_Handle>
struct TypedHandle final : std::conditional_t<std::is_const_v<T>,typename _Handle::const_type,_Handle>
{
	private:
		using base_t = std::conditional_t<std::is_const_v<T>,typename _Handle::const_type,_Handle>;

	public:
		inline auto operator<=>(const _Handle& other) const {return base_t::operator<=>(other);}
		inline auto operator<=>(const typename _Handle::const_type& other) const {return base_t::operator<=>(other);}

		inline operator TypedHandle<const T,_Handle>() const {return {{.value=base_t::value}};}

		template<typename U> requires ((std::is_void_v<U> || std::is_base_of_v<U,T>) && (!std::is_const_v<T> || std::is_const_v<U>))
		inline operator TypedHandle<U,_Handle>() const
		{
			TypedHandle<U,_Handle> retval;
			retval.value = base_t::value;
			if constexpr (!std::is_void_v<U>)
			if (*this)
			{
				const T* const fake_this = reinterpret_cast<const T*>(sizeof(T));
				retval.value += ptrdiff_t(static_cast<const U*>(fake_this)) - sizeof(T);
			}
			return retval;
		}
};

//! Does not resize memory arenas, therefore once allocations shall not move
//! Needs an address allocator that takes a Block Size after max alignment parameter
template<class AddressAllocator>
class BlockBasedAllocatorBase
{
	public:
		using addr_alloc_traits = address_allocator_traits<AddressAllocator>;
		using extra_params_type = tuple_transform_t<impl::identitiy,typename addr_alloc_traits::extra_ctor_param_types>;
		using size_type = typename addr_alloc_traits::size_type;

		// The blocks will be allocated aligned to at least this value
		constexpr static inline size_type meta_alignment = 64u;
		
		struct SCreationParams final
		{
			smart_refctd_ptr<refctd_memory_resource> mem_resource = nullptr;
			extra_params_type addrAllocCtorExtraParams;
			uint16_t blockSizeKBLog2 : 5 = 7;
			uint16_t initBlockCount : 11 = 1;
		};

	protected:
		// Blocks get allocated/deallocated on demand, inside there's a suballocator.
		// Blocks are always the same size, and obviously can't allocate anything larger than themselves.
		class alignas(16) Block final
		{
				AddressAllocator addrAlloc;

			public:
				struct SCreationParams
				{
					inline SCreationParams(const size_type _blockSize, const extra_params_type& extraArgs) : addrCreationArgs(extraArgs), blockSize(_blockSize),
						reservedSize(std::apply([_blockSize]<typename... Args>(const Args&... args)->size_type
							{
								return addr_alloc_traits::reserved_size(meta_alignment,_blockSize,args...);
							},addrCreationArgs)
						), totalSize(core::alignUp(sizeof(Block)+reservedSize,meta_alignment)+blockSize) {}

					const extra_params_type addrCreationArgs;
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
		
		
		inline BlockBasedAllocatorBase(SCreationParams&& params) : m_blockCreationParams(1024u<<params.blockSizeKBLog2,std::move(params.addrAllocCtorExtraParams)),
			m_initBlockCount(std::max<size_type>(params.initBlockCount,1)), m_mem_resource(params.mem_resource ? std::move(params.mem_resource):core::getDefaultMemoryResource()) {}
		BlockBasedAllocatorBase(const BlockBasedAllocatorBase&) = delete;
		BlockBasedAllocatorBase(BlockBasedAllocatorBase&&) = delete;

		const Block::SCreationParams m_blockCreationParams;
		const uint32_t m_initBlockCount;
		smart_refctd_ptr<refctd_memory_resource> m_mem_resource;

    public:
		//
		inline const Block::SCreationParams getBlockCreationParams() const {return m_blockCreationParams;}

    protected:
		inline Block* createBlock()
		{
			auto* block = reinterpret_cast<Block*>(m_mem_resource->allocate(m_blockCreationParams.totalSize,meta_alignment));
			std::construct_at(block,m_blockCreationParams);
			return block;
		}
		inline void deleteBlock(Block* block)
		{
			assert(block);
			std::destroy_at(block);
			m_mem_resource->deallocate(block,m_blockCreationParams.totalSize,meta_alignment);
		}

};

// forward declare
template<class AddressAllocator, typename HandleValue>
class SimpleBlockBasedAllocator;

//! void* makes it a regular allocator, but one needs to perform a binary search for the block in a map of live blocks when deallocating
//! We could try to allocate the blocks with alignments so massive that we could take the pointer MSB directly, but unsure about OS guarantees
template<class AddressAllocator>
class SimpleBlockBasedAllocator<AddressAllocator,void*> final : protected BlockBasedAllocatorBase<AddressAllocator>
{
		using base_t = BlockBasedAllocatorBase<AddressAllocator>;
		using block_t = typename base_t::Block;

	public:
		using handle_value_type = void*;
		using addr_alloc_traits = typename base_t::addr_alloc_traits;
		using extra_params_type = typename base_t::extra_params_type;
		using size_type = typename base_t::size_type;

		template<typename T>
		using typed_pointer_type = T*;
		
		template<typename T, typename U>
		static inline T* _const_cast(U* p) {return const_cast<T*>(p);}
		template<typename T, typename U>
		static inline T* _reinterpret_cast(U* p) {return reinterpret_cast<T*>(p);}
		template<typename T, typename U>
		static inline T* _static_cast(U* p) {return static_cast<T*>(p);}

		struct SCreationParams final
		{
			base_t::SCreationParams composed;
		};
		inline SimpleBlockBasedAllocator(SCreationParams&& params) : base_t(std::move(params.composed))
		{
			for (auto i=0u; i<base_t::m_initBlockCount; i++)
				m_blocks.insert(base_t::createBlock());
		}
        inline ~SimpleBlockBasedAllocator()
		{
			for (auto& block : m_blocks)
				base_t::deleteBlock(const_cast<block_t*>(block));
		}

		//
		template<typename T> requires (!std::is_const_v<T>)
		inline T* deref(typed_pointer_type<T> p) {return p;}
		template<typename T>
		inline const T* deref(typed_pointer_type<T> p) const {return p;}

		// deallocates everything
        inline void	reset()
        {
			assert(m_blocks.size()>=base_t::m_initBlockCount);
			auto it = m_blocks.begin();
			for (auto i=0u; i<m_blocks.size(); i++,it++)
			{
				block_t* block = *it;
				if (i<base_t::m_initBlockCount)
					block->addrAlloc.reset();
				else
				{
					base_t::deleteBlock(block);
					m_blocks.erase(it);
				}
			}
        }

		//
		inline void* allocate(const size_type bytes, const size_type alignment) noexcept
		{
			constexpr auto invalid_address = AddressAllocator::invalid_address;
			// TODO: better allocation strategies like tlsf
			for (auto& entry : m_blocks)
			{
				block_t* block = const_cast<block_t*>(entry);
				if (const auto addr=block->alloc(bytes,alignment); addr!=invalid_address)
					return block->data(base_t::m_blockCreationParams)+addr;
			}
			block_t* block = base_t::createBlock();
			if (const auto addr=block->alloc(bytes,alignment); addr!=invalid_address)
			{
				m_blocks.insert(block);
				return block->data(base_t::m_blockCreationParams)+addr;
			}
			else
				base_t::deleteBlock(block);
			return nullptr;
		}
		inline void	deallocate(void* p, const size_type bytes) noexcept
		{
			assert(m_blocks.size()>=base_t::m_initBlockCount);
			auto found = m_blocks.lower_bound(reinterpret_cast<block_t*>(p));
			assert(found!=m_blocks.end());
			auto* block = *found;
			uint8_t* blockData = block->data(base_t::m_blockCreationParams);
			assert(blockData<=p && p<blockData+base_t::m_blockCreationParams.blockSize);
			const size_type addr = reinterpret_cast<uint8_t*>(p)-blockData;
			block->free(addr,bytes);
			if (m_blocks.size()>base_t::m_initBlockCount && addr_alloc_traits::get_allocated_size(block->getAllocator())==size_type(0u))
			{
				base_t::deleteBlock(block);
				m_blocks.erase(found);
			}
		}

    private:
		// TODO: better allocation strategies like tlsf
		using block_map_t = gtl::btree_set<block_t*>;
		block_map_t m_blocks;
};

//! This one uses handles instead of pointers, this way they're quasi-serializable and compact
template<class AddressAllocator, typename HandleValue> requires (std::is_integral_v<HandleValue> && std::is_unsigned_v<HandleValue>)
class SimpleBlockBasedAllocator<AddressAllocator,HandleValue> final : protected BlockBasedAllocatorBase<AddressAllocator>
{
		using base_t = BlockBasedAllocatorBase<AddressAllocator>;
		using this_t = SimpleBlockBasedAllocator<AddressAllocator,HandleValue>;
		using block_t = typename base_t::Block;
		//
		using block_id_t = std::conditional_t<sizeof(HandleValue)<=4,uint16_t,uint32_t>;
		using block_id_alloc_t = PoolAddressAllocatorST<block_id_t>;

	public:
		using handle_value_type = typename HandleValue;
		using addr_alloc_traits = typename base_t::addr_alloc_traits;
		using extra_params_type = typename base_t::extra_params_type;
		using size_type = typename base_t::size_type;

		// everything is handed out by index not pointer
		template<typename T>
		using typed_pointer_type = TypedHandle<T,Handle<ConstHandle<HandleValue,~HandleValue(0)> > >;

		//
		template<typename T, typename U> requires std::is_same_v<std::remove_cv_t<T>,std::remove_cv_t<U>>
		static inline typed_pointer_type<T> _const_cast(typed_pointer_type<U> p)
		{
			typed_pointer_type<T> retval;
			retval.value = p.value;
			return retval;
		}
		template<typename T, typename U> requires (!std::is_const_v<U> || std::is_const_v<T>)
		static inline typed_pointer_type<T> _reinterpret_cast(typed_pointer_type<U> p)
		{
			typed_pointer_type<T> retval;
			retval.value = p.value;
			return retval;
		}
		template<typename T, typename U>
		static inline typed_pointer_type<T> _static_cast(typed_pointer_type<U> p)
		{
			typed_pointer_type<T> retval;
			retval.value = p.value;
			if (p)
			{
				const auto begin = std::max(sizeof(T),sizeof(U));
				const U* const fake_p = reinterpret_cast<const U*>(begin);
				retval.value += ptrdiff_t(static_cast<const T*>(fake_p)) - begin;
			}
			return retval;
		}

		struct SCreationParams final
		{
			base_t::SCreationParams composed;
			// TODO: `reserved_size` probably needs to return `size_t` cause otherwise this value can't be bigger than default
			block_id_t maxBlocks = 0x1<<13;
		};
		inline SimpleBlockBasedAllocator(SCreationParams&& params) : base_t(std::move(params.composed)),
			m_blockIndexAlloc(base_t::m_mem_resource->allocate(block_id_alloc_t::reserved_size(1,params.maxBlocks,1),_NBL_SIMD_ALIGNMENT),0,0,1,params.maxBlocks,1),
			m_blockSizeLog2(hlsl::findMSB<size_type>(base_t::getBlockCreationParams().blockSize)), m_loAddrMask((HandleValue(1)<<m_blockSizeLog2)-1)
		{
			assert(base_t::m_initBlockCount<=params.maxBlocks);
			for (auto i=0u; i<base_t::m_initBlockCount; i++)
			{
				const auto id = m_blockIndexAlloc.alloc_addr(1,1);
				assert(id!=block_id_alloc_t::invalid_address);
				m_blocks[id] = base_t::createBlock();
			}
		}
        inline ~SimpleBlockBasedAllocator()
		{
			for (auto& block : m_blocks)
				base_t::deleteBlock(block.second);
			const auto reservedSize = block_id_alloc_t::reserved_size(m_blockIndexAlloc,m_blockIndexAlloc.get_total_size());
			const void* reservedPtr = address_allocator_traits<block_id_alloc_t>::getReservedSpacePtr(m_blockIndexAlloc);
			base_t::m_mem_resource->deallocate(const_cast<void*>(reservedPtr),reservedSize,_NBL_SIMD_ALIGNMENT);
		}

		//
		template<typename T> requires (!std::is_const_v<T>)
		inline T* deref(typed_pointer_type<T> p)
		{
			if (p)
				return reinterpret_cast<T*>(std::launder(getBlock(p)->data(base_t::m_blockCreationParams)+getOffsetInBlock(p)));
			return nullptr;
		}
		template<typename T>
		inline const T* deref(typed_pointer_type<T> p) const
		{
			std::remove_const_t<T>* mut = const_cast<this_t*>(this)->deref<std::remove_const_t<T>>(p);
			return mut;
		}

		// deallocates everything
        inline void	reset()
        {
			block_map_t recycledBlocks;
			recycledBlocks.reserve(base_t::m_initBlockCount);
			assert(m_blocks.size()>=base_t::m_initBlockCount);
			m_blockIndexAlloc.reset();
			for (auto& entry : m_blocks)
			{
				block_t* block = entry.second;
				if (recycledBlocks.size()<base_t::m_initBlockCount)
				{
					block->addrAlloc.reset();
					const auto id = m_blockIndexAlloc.alloc_addr(1,1);
					assert(id!=block_id_alloc_t::invalid_address);
					recycledBlocks[id] = block;
				}
				else
					base_t::deleteBlock(block);
			}
			m_blocks = std::move(recycledBlocks);
        }

		//
		inline typed_pointer_type<void> allocate(const size_type bytes, const size_type alignment) noexcept
		{
			constexpr auto invalid_address = AddressAllocator::invalid_address;
			// TODO: better allocation strategies like tlsf
			for (auto& entry : m_blocks)
			{
				block_t* block = entry.second;
				if (const auto addr=block->alloc(bytes,alignment); addr!=invalid_address)
					return {{{.value=(entry.first<<m_blockSizeLog2)|addr}}};
			}
			const auto newID = m_blockIndexAlloc.alloc_addr(1,1);
			if (newID!=block_id_alloc_t::invalid_address)
			{
				block_t* block = base_t::createBlock();
				if (const auto addr=block->alloc(bytes,alignment); addr!=invalid_address)
				{
					m_blocks[newID] = block;
					return {{{.value=(newID<<m_blockSizeLog2)|addr}}};
				}
				else
					base_t::deleteBlock(block);
			}
			return {};
		}
		inline void	deallocate(const typed_pointer_type<void> h, const size_type bytes) noexcept
		{
			assert(m_blocks.size()>=base_t::m_initBlockCount);
			auto found = m_blocks.find(getBlockIndex(h));
			assert(found!=m_blocks.end());
			auto* block = found->second;
			block->free(getOffsetInBlock(h),bytes);
			if (m_blocks.size()>base_t::m_initBlockCount && addr_alloc_traits::get_allocated_size(block->getAllocator())==size_type(0u))
			{
				base_t::deleteBlock(block);
				m_blockIndexAlloc.free_addr(found->first,1);
				m_blocks.erase(found);
			}
		}

    private:
		inline HandleValue getOffsetInBlock(const typed_pointer_type<const void> h) const {return h.value&m_loAddrMask;}
		inline block_t* getBlock(const typed_pointer_type<const void> h) {return m_blocks.find(getBlockIndex(h))->second;}

		inline HandleValue getBlockIndex(const typed_pointer_type<const void> h) const {return h.value>>m_blockSizeLog2;}

		// Either flat array, keeps our `deref()` fast, or hash map which isn't bad because lookup up objects from different blocks will trash anyway.
		using block_map_t = core::unordered_map<HandleValue,block_t*>;
		block_id_alloc_t m_blockIndexAlloc;
		block_map_t m_blocks;
		const uint8_t m_blockSizeLog2;
		const HandleValue m_loAddrMask;
};

template<class AddressAllocator, typename HandleValue=void*>
using SimpleBlockBasedAllocatorST = SimpleBlockBasedAllocator<AddressAllocator,HandleValue>;


template<class Composed, class RecursiveLockable=std::recursive_mutex> requires std::is_base_of_v<SimpleBlockBasedAllocator<typename Composed::addr_alloc_traits::allocator_type,typename Composed::handle_value_type>,Composed>
class SimpleBlockBasedAllocatorMT final
{
		using this_t = SimpleBlockBasedAllocatorMT<Composed,RecursiveLockable>;

	protected:
		Composed m_composed;
        RecursiveLockable m_lock;

	public:
        using size_type = typename Composed::size_type;
		template<typename T>
		using typed_pointer_type = Composed::template typed_pointer_type<T>;

		using creation_params_type = Composed::SCreationParams;
		inline SimpleBlockBasedAllocatorMT(creation_params_type&& params) : m_composed(std::move(params)), m_lock() {}
        virtual inline ~SimpleBlockBasedAllocatorMT() {}

		//
		template<typename T> requires (!std::is_const_v<T>)
		inline T* deref(typed_pointer_type<T> p)
		{
			return m_composed.deref<T>(p);
		}
		template<typename T> requires std::is_const_v<T>
		inline T* deref(typed_pointer_type<T> p) const
		{
			return m_composed.deref<T>(p);
		}

		//
        inline void	reset()
        {
			m_lock.lock();
			m_composed.reset();
			m_lock.unlock();
        }

		inline typed_pointer_type<void> allocate(size_type bytes, size_type alignment) noexcept
		{
			m_lock.lock();
			auto ret = m_composed.allocate(bytes, alignment);
			m_lock.unlock();
			return ret;
		}
		inline void	deallocate(typed_pointer_type<void> p, size_type bytes) noexcept
		{
			m_lock.lock();
			m_composed.deallocate(p,bytes);
			m_lock.unlock();
		}

		//! Extra == Use WITH EXTREME CAUTION
		inline RecursiveLockable& get_lock() noexcept
		{
			return m_lock;
		}
};
// no aliases

}

namespace std
{
template<typename HandleValue, HandleValue _Invalid>
struct hash<nbl::core::ConstHandle<HandleValue,_Invalid> >
{
	inline size_t operator()(const nbl::core::ConstHandle<HandleValue,_Invalid> handle) const
	{
		return std::hash<HandleValue>()(handle.value);
	}
};
template<typename HandleValue, HandleValue _Invalid>
struct hash<nbl::core::Handle<nbl::core::ConstHandle<HandleValue,_Invalid> > >
{
	inline size_t operator()(const nbl::core::Handle<nbl::core::ConstHandle<HandleValue,_Invalid> > handle) const
	{
		return std::hash<HandleValue>()(handle.value);
	}
};
template<typename T, typename HandleValue, HandleValue _Invalid>
struct hash<nbl::core::TypedHandle<T,nbl::core::Handle<nbl::core::ConstHandle<HandleValue,_Invalid> > > >
{
	inline size_t operator()(const nbl::core::TypedHandle<T,nbl::core::Handle<nbl::core::ConstHandle<HandleValue,_Invalid> > > handle) const
	{
		return std::hash<HandleValue>()(handle.value);
	}
};
}
#endif


