// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_MATERIAL_COMPILER_V3_C_NODE_POOL_H_INCLUDED_
#define _NBL_ASSET_MATERIAL_COMPILER_V3_C_NODE_POOL_H_INCLUDED_


#include "nbl/core/declarations.h"
#include "nbl/core/definitions.h"
#include "nbl/core/alloc/refctd_memory_resource.h"

#include <type_traits>


namespace nbl::asset::material_compiler3
{

// Class to manage all nodes' backing and hand them out as `uint32_t` handles
class CNodePool : public core::IReferenceCounted
{
	public:
		// everything is handed out by index not pointer
		struct Handle
		{
			using value_t = uint32_t;
			constexpr static inline value_t Invalid = ~value_t(0);

			explicit inline operator bool() const {return value!=Invalid;}
			inline bool operator==(const Handle& other) const {return value==other.value;}

			// also serves as a byte offset into the pool
			value_t value = Invalid;
		};
		class INode
		{
			public:
				//
				virtual const std::string_view getTypeName() const = 0;

			protected:
				//
				friend class CNodePool;

				// to not be able to make the variable length stuff on the stack
				virtual ~INode() = 0;

				// to support variable length stuff
				virtual uint32_t getSize() const = 0;
		};
		// Debug Info node
		class CDebugInfo : public INode
		{
			public:
				inline const std::string_view getTypeName() const override {return "nbl::CNodePool::CDebugInfo";}
				inline uint32_t getSize() const {return calc_size(nullptr,m_size);}
				
				static inline uint32_t calc_size(const void* data, const uint32_t size)
				{
					return sizeof(CDebugInfo)+size;
				}
				static inline uint32_t calc_size(const std::string_view& view)
				{
					return calc_size(nullptr,view.length()+1);
				}
				inline CDebugInfo(const void* data, const uint32_t size) : m_size(size)
				{
					if (data)
						memcpy(std::launder(this+1),data,m_size);
				}
				inline CDebugInfo(const std::string_view& view) : CDebugInfo(nullptr,view.length()+1)
				{
					auto* out = std::launder(reinterpret_cast<char*>(this+1));
					if (m_size>1)
						memcpy(out,view.data(),m_size);
					out[m_size-1] = 0;
				}

				inline const std::span<const uint8_t> data() const
				{
					return {std::launder(reinterpret_cast<const uint8_t*>(this+1)),m_size};
				}

			protected:
				const uint32_t m_size;
		};

		//
		template<typename T>
		struct TypedHandle
		{
			using node_type = T;
			
			explicit inline operator bool() const {return bool(untyped);}
			inline bool operator==(const TypedHandle& other) const {return untyped==other.untyped;}

			inline operator const TypedHandle<const T>&() const
			{
				static_assert(std::is_base_of_v<INode,std::remove_const_t<T>>);
				return *reinterpret_cast<const TypedHandle<const T>*>(this);
			}
			template<typename U> requires (std::is_base_of_v<U,T> && (std::is_const_v<U> || !std::is_const_v<T>)) 
			inline operator const TypedHandle<U>&() const
			{
				return *reinterpret_cast<const TypedHandle<U>*>(this);
			}

			Handle untyped = {};
		};
		template<typename T> requires (!std::is_const_v<T>)
		inline T* deref(const TypedHandle<T> h) {return deref<T>(h.untyped);}
		template<typename T>
		inline const T* deref(const TypedHandle<T> h) const {return deref<const T>(h.untyped);}

		template<typename T>
		inline const std::string_view getTypeName(const TypedHandle<T> h) const
		{
			const auto* node = deref<const T>(h.untyped);
			return node ? node->getTypeName():"nullptr";
		}

	protected:
		struct HandleHash
		{
			inline size_t operator()(const TypedHandle<const INode> handle) const
			{
				return std::hash<Handle::value_t>()(handle.untyped.value);
			}
		};
		// save myself some typing
		using refctd_pmr_t = core::smart_refctd_ptr<core::refctd_memory_resource>;

		inline Handle alloc(const uint32_t size, const uint16_t alignment)
		{
			Handle retval = {};
			auto allocFromChunk = [&](Chunk& chunk, const uint32_t chunkIx)
			{
				const auto localOffset = chunk.alloc(size,alignment);
				if (localOffset!=Chunk::allocator_t::invalid_address)
					retval.value = localOffset|(chunkIx<<m_chunkSizeLog2);
			};
			// try current back chunk
			if (!m_chunks.empty())
				allocFromChunk(m_chunks.back(),m_chunks.size()-1);
			// if fail try new chunk
			if (!retval)
			{
				const auto chunkSize = 0x1u<<m_chunkSizeLog2;
				const auto chunkAlign = 0x1u<<m_maxNodeAlignLog2;
				Chunk newChunk;
				newChunk.getAllocator() = Chunk::allocator_t(nullptr,0,0,chunkAlign,chunkSize),
				newChunk.m_data = reinterpret_cast<uint8_t*>(m_pmr->allocate(chunkSize,chunkAlign));
				if (newChunk.m_data)
				{
					allocFromChunk(newChunk,m_chunks.size());
					if (retval)
						m_chunks.push_back(std::move(newChunk));
					else
						m_pmr->deallocate(newChunk.m_data,chunkSize,chunkAlign);
				}
			}
			return retval;
		}
		inline void free(const Handle h, const uint32_t size)
		{
			assert(getChunkIx(h)<m_chunks.size());
		}

		// new
		template<typename T, typename... Args>
		inline TypedHandle<T> _new(Args&&... args)
		{
			const uint32_t size = T::calc_size(args...);
			const Handle retval = alloc(size,alignof(T));
			if (retval)
				new (deref<void>(retval)) T(std::forward<Args>(args)...);
			return {.untyped=retval};
		}
		// delete
		template<typename T>
		inline void _delete(const TypedHandle<T> h)
		{
			T* ptr = deref<T>(h);
			const uint32_t size = ptr->getSize();
			static_cast<INode*>(ptr)->~INode(); // can't use `std::destroy_at<T>(ptr);` because of destructor being non-public
			// wipe v-table to mark as dead (so `~CNodePool` doesn't run destructor twice)
			// NOTE: This won't work if we start reusing memory, even zeroing out the whole node won't work! Then need an accurate record of live nodes!
			const void* nullVTable = nullptr;
			assert(memcmp(ptr,&nullVTable,sizeof(nullVTable))!=0); // double free
			memset(static_cast<INode*>(ptr),0,sizeof(nullVTable));
			free(h.untyped,size);
		}

		inline CNodePool(const uint8_t _chunkSizeLog2, const uint8_t _maxNodeAlignLog2, refctd_pmr_t&& _pmr) :
			m_chunkSizeLog2(_chunkSizeLog2), m_maxNodeAlignLog2(_maxNodeAlignLog2), m_pmr(_pmr ? std::move(_pmr):core::getDefaultMemoryResource())
		{
			assert(m_chunkSizeLog2>=14 && m_maxNodeAlignLog2>=4);
		}
		// Destructor performs a form of garbage collection (just to make sure destructors are ran)
		// NOTE: C++26 reflection would allow us to find all the `Handle` and `TypedHandle<U>` in `T` and do actual mark-and-sweep Garbage Collection
		inline ~CNodePool()
		{
			const auto chunkSize = 0x1u<<m_chunkSizeLog2;
			const auto chunkAlign = 0x1u<<m_maxNodeAlignLog2;
			for (auto& chunk : m_chunks)
			{
				for (auto handleOff=chunk.getAllocator().get_total_size(); handleOff<chunkSize; handleOff+=sizeof(Handle))
				{
					const auto pHandle = reinterpret_cast<const Handle*>(chunk.m_data+handleOff);
					// NOTE: This won't work if we start reusing memory, even zeroing out the whole node won't work! Then need an accurate record of live nodes!
					if (auto* node=deref<INode>(*pHandle); node)
						node->~INode(); // can't use `std::destroy_at<T>(ptr);` because of destructor being non-public
				}
				m_pmr->deallocate(chunk.m_data,chunkSize,chunkAlign);
			}
		}

	private:
		struct Chunk
		{
			// for now using KISS, we can use geeneralpupose allocator later
			// Generalpurpose would require us to store the allocated handle list in a different way, so that handles can be quickly removed from it.
			// Maybe a doubly linked list around the original allocation?
			using allocator_t = core::LinearAddressAllocatorST<Handle::value_t>;

			inline allocator_t& getAllocator()
			{
				return *m_alloc.getStorage();
			}

			inline Handle::value_t alloc(const uint32_t size, const uint16_t alignment)
			{
				const auto retval = getAllocator().alloc_addr(size,alignment);
				// successful allocation, time for some book keeping
				constexpr auto invalid_address = allocator_t ::invalid_address;
				if (retval!=invalid_address)
				{
					// we keep a list of all the allocated nodes at the back of a chunk
					const auto newSize = getAllocator().get_total_size()-sizeof(retval);
					// handle no space left for bookkeeping case
					if (retval+size>newSize)
					{
						free(retval,size);
						return invalid_address;
					}
					// clear vtable to mark as not initialized yet
					// TODO: this won't work with reusable memory / not bump allocator
					memset(m_data+retval,0,sizeof(INode));
					*std::launder(reinterpret_cast<Handle::value_t*>(m_data+newSize)) = retval;
					// shrink allocator
					getAllocator() = allocator_t(newSize, std::move(getAllocator()), nullptr);
				}
				return retval;
			}
			inline void free(const Handle::value_t addr, const uint32_t size)
			{
				getAllocator().free_addr(addr,size);
			}

			// make the chunk plain data, it has to get initialized and deinitialized externally anyway
			core::StorageTrivializer<allocator_t> m_alloc;
			uint8_t* m_data;
		};
		inline uint32_t getChunkIx(const Handle h) {return h.value>>m_chunkSizeLog2;}

		template<typename T> requires (std::is_base_of_v<INode,T> && !std::is_const_v<T> || std::is_void_v<T>)
		inline T* deref(const Handle h)
		{
			if (!h)
				return nullptr;
			const auto hiAddr = getChunkIx(h);
			assert(hiAddr<m_chunks.size());
			{
				const auto loAddr = h.value&((0x1u<<m_chunkSizeLog2)-1);
				void* ptr = m_chunks[hiAddr].m_data+loAddr;
				if constexpr (std::is_void_v<T>)
					return ptr;
				else
				{
					if (*std::launder(reinterpret_cast<const void* const*>(ptr))) // vtable not wiped
					{
						auto* base = std::launder(reinterpret_cast<INode*>(ptr));
						return dynamic_cast<T*>(base);
					}
				}
			}
			return nullptr;
		}
		template<typename T> requires (std::is_base_of_v<INode,T> && std::is_const_v<T>)
		inline T* deref(const Handle h) const
		{
			return const_cast<CNodePool*>(this)->deref<std::remove_const_t<T>>(h);
		}

		core::vector<Chunk> m_chunks;
		refctd_pmr_t m_pmr;
		const uint8_t m_chunkSizeLog2; // maybe hardcode chunk sizes to 64kb ?
		const uint8_t m_maxNodeAlignLog2;
};

inline CNodePool::INode::~INode()
{
}

} // namespace nbl::asset::material_compiler3
#endif