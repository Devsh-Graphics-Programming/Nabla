// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef __NBL_ASSET_MATERIAL_COMPILER_V3_IR_H_INCLUDED__
#define __NBL_ASSET_MATERIAL_COMPILER_V3_IR_H_INCLUDED__


#include "nbl/core/declarations.h"
#include "nbl/core/definitions.h"

#include "nbl/asset/ICPUImageView.h"

//#include "IRNode.h"
#include <nbl/asset/ICPUImageView.h>


// temporary
#define NBL_API

namespace nbl::asset::material_compiler3
{

// Class to manage all nodes' backing and hand them out as `uint32_t` handles
class NBL_API CNodePool : public core::IReferenceCounted
{
		// save myself some typing
		using refctd_pmr_t = core::smart_refctd_ptr<core::refctd_memory_resource>;

	public:
		// constructor
		inline core::smart_refctd_ptr<CNodePool> create(const uint8_t chunkSizeLog2=19, const uint8_t maxNodeAlignLog2=4, refctd_pmr_t&& _pmr={})
		{
			if (chunkSizeLog2<14 || maxNodeAlignLog2<4)
				return nullptr;
			if (!_pmr)
				_pmr = core::getDefaultMemoryResource();
			return core::smart_refctd_ptr<CNodePool>(new CNodePool(chunkSizeLog2,maxNodeAlignLog2,std::move(_pmr)),core::dont_grab);
		}

		// everything is handed out by index not pointer
		struct Handle
		{
			using value_t = uint32_t;
			constexpr static inline value_t Invalid = ~value_t(0);

			inline operator bool() const {return value!=Invalid;}

			// also serves as a byte offset into the pool
			value_t value = Invalid;
		};
		template<typename T>
		inline T* deref(const Handle& h) {chunks[getChunkIx(h)].data()}

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
		inline void free(const Handle& h)
		{
			assert(getChunkIx(h)<m_chunks.size());
		}

		// new
		// delete

		//

	protected:
		// for now using KISS, we can use geeneralpupose allocator later
		struct Chunk
		{
			inline Handle::value_t alloc(const uint32_t size, const uint16_t alignment)
			{
				return m_alloc.alloc_addr(size,alignment);
			}
			inline void free(const Handle::value_t addr, const uint32_t size)
			{
				m_alloc.free_addr(addr,size);
			}

			core::LinearAddressAllocatorST<Handle::value_t> m_alloc;
			uint8_t* m_data;
		};
		inline CNodePool(const uint8_t _chunkSizeLog2, const uint8_t _maxNodeAlignLog2, refctd_pmr_t&& _pmr) :
			m_chunkSizeLog2(_chunkSizeLog2), m_maxNodeAlignLog2(_maxNodeAlignLog2), m_pmr(std::move(_pmr)) {}
		inline uint32_t getChunkIx(const Handle& h) {h.value>>m_chunkSizeLog2;}

		core::vector<Chunk> m_chunks;
		refctd_pmr_t m_pmr;
		const uint8_t m_chunkSizeLog2;
		const uint8_t m_maxNodeAlignLog2;
};

class NBL_API CForest
{
	public:
		class INode
		{
			public:
				enum class Type : uint8_t
				{
					Emission,
					BxDF,
					Multiply,
					Add,
					Sub,
					// 1 - C_0
					Complement,
					Function
				};
				virtual Type getType() const = 0;

				// Only sane child count allowed
				virtual uint8_t getChildCount() = 0;

				// based 
				virtual CNodePool::Handle getChild(const uint8_t index) const;
		};

	protected:
};

//! DAG (baked)
//! Nodes

} // namespace nbl::asset::material_compiler3

#endif