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

			inline operator bool() const {return value!=Invalid;}

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
				inline uint32_t getSize() const {return m_size;}
				
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
						memcpy(this+1,data,m_size);
				}
				inline CDebugInfo(const std::string_view& view) : CDebugInfo(nullptr,view.length()+1)
				{
					auto* out = reinterpret_cast<char*>(this+1);
					if (m_size>1)
						memcpy(out,view.data(),m_size);
					out[m_size-1] = 0;
				}

			protected:
				const uint32_t m_size;
		};

		//
		template<typename T> requires std::is_base_of_v<INode,T>
		struct TypedHandle
		{
			using node_type = T;

			Handle untyped;
		};
		template<typename T>
		inline T* deref(const TypedHandle<T>& h) {return deref<T>(h.untyped);}
		template<typename T>
		inline const T* deref(const TypedHandle<const T>& h) const {return deref<const T>(h.untyped);}

	protected:
		// save myself some typing
		using refctd_pmr_t = core::smart_refctd_ptr<core::refctd_memory_resource>;

		inline Handle alloc(const uint32_t size, const uint16_t alignment)
		{
			Handle retval = {};
			auto allocFromChunk = [&](Chunk& chunk, const uint32_t chunkIx)
			{
				const auto localOffset = chunk.alloc(size,alignment);
				if (localOffset!=decltype(Chunk::m_alloc)::invalid_address)
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
				Chunk newChunk = {
					.m_alloc = decltype(Chunk::m_alloc)(nullptr,0,0,chunkAlign,chunkSize),
					.m_data = reinterpret_cast<uint8_t*>(m_pmr->allocate(chunkSize,chunkAlign))
				};
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
		inline void free(const Handle& h, const uint32_t size)
		{
			assert(getChunkIx(h)<m_chunks.size());
		}

		// new
		template<typename T, typename... Args>
		inline Handle _new(Args&&... args)
		{
			const uint32_t size = T::calc_size(args...);
			const Handle retval = alloc(size,alignof(T));
			if (retval)
				std::construct_at<T>(deref<T>(retval),std::forward<Args>(args)...);
			return retval;
		}
		// delete
		template<typename T>
		inline void _delete(const Handle& h)
		{
			T* ptr = deref<T>(h);
			const uint32_t size = ptr->getSize();
			std::destroy_at<T>(ptr);
			// wipe v-table to mark as dead (so `~CNodePool` doesn't run destructor twice)
			memset(static_cast<INode*>(ptr),0,sizeof(INode));
			free(h,size);
		}

		inline CNodePool(const uint8_t _chunkSizeLog2, const uint8_t _maxNodeAlignLog2, refctd_pmr_t&& _pmr) :
			m_chunkSizeLog2(_chunkSizeLog2), m_maxNodeAlignLog2(_maxNodeAlignLog2), m_pmr(_pmr ? std::move(_pmr):core::getDefaultMemoryResource())
		{
			assert(m_chunkSizeLog2>=14 && m_maxNodeAlignLog2>=4);
		}
		inline ~CNodePool()
		{
			const auto chunkSize = 0x1u<<m_chunkSizeLog2;
			const auto chunkAlign = 0x1u<<m_maxNodeAlignLog2;
			for (auto& chunk : m_chunks)
			{
				for (auto handleOff=chunk.m_alloc.get_total_size(); handleOff<chunkSize; handleOff+=sizeof(Handle))
				{
					const auto pHandle = reinterpret_cast<const Handle*>(chunk.m_data+handleOff);
					if (auto* node=deref<INode>({.value=*pHandle}); node)
						std::destroy_at(node);
				}
				m_pmr->deallocate(chunk.m_data,chunkSize,chunkAlign);
			}
		}

	private:
		struct Chunk
		{
			inline Handle::value_t alloc(const uint32_t size, const uint16_t alignment)
			{
				const auto retval = m_alloc.alloc_addr(size,alignment);
				// successful allocation, time for some book keeping
				constexpr auto invalid_address = decltype(m_alloc)::invalid_address;
				if (retval!=invalid_address)
				{
					// we keep a list of all the allocated nodes at the back of a chunk
					const auto newSize = m_alloc.get_total_size()-sizeof(retval);
					// handle no space left for bookkeeping case
					if (retval+size>newSize)
					{
						free(retval,size);
						return invalid_address;
					}
					// clear vtable to mark as not initialized yet
					memset(m_data+retval,0,sizeof(INode));
					*reinterpret_cast<Handle::value_t*>(m_data+newSize) = retval;
					// shrink allocator
					m_alloc = decltype(m_alloc)(newSize,std::move(m_alloc));
				}
				return retval;
			}
			inline void free(const Handle::value_t addr, const uint32_t size)
			{
				m_alloc.free_addr(addr,size);
			}

			// for now using KISS, we can use geeneralpupose allocator later
			core::LinearAddressAllocatorST<Handle::value_t> m_alloc;
			uint8_t* m_data;
		};
		inline uint32_t getChunkIx(const Handle& h) {h.value>>m_chunkSizeLog2;}

		template<typename T> requires (std::is_base_of_v<INode,T> && !std::is_const_v<T>)
		inline T* deref(const Handle& h)
		{
			const auto hiAddr = getChunkIx(h);
			assert(hiAddr<chunks.size());
			{
				const auto loAddr = h.value&(0x1u<<m_chunkSizeLog2);
				void* ptr = chunks[hiAddr].data()+loAddr;
				if (ptr) // vtable not wiped
					return dynamic_cast<T*>(ptr);
			}
			return nullptr;
		}
		template<typename T> requires (std::is_base_of_v<INode,T> && std::is_const_v<T>)
		inline T* deref(const Handle& h) const
		{
			return const_cast<CNodePool*>(this)->deref<std::remove_const_t<T>>(h);
		}

		core::vector<Chunk> m_chunks;
		refctd_pmr_t m_pmr;
		const uint8_t m_chunkSizeLog2; // maybe hardcode chunk sizes to 64kb ?
		const uint8_t m_maxNodeAlignLog2;
};

} // namespace nbl::asset::material_compiler3
#endif