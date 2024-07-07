// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_CPU_BUFFER_VIEW_H_INCLUDED_
#define _NBL_ASSET_I_CPU_BUFFER_VIEW_H_INCLUDED_


#include <utility>

#include "nbl/asset/IAsset.h"
#include "nbl/asset/IBufferView.h"
#include "nbl/asset/ICPUBuffer.h"

namespace nbl::asset
{

class ICPUBufferView : public IBufferView<ICPUBuffer>, public IAsset
{
	public:
		ICPUBufferView(const SBufferRange<ICPUBuffer>& _underlying, const E_FORMAT _format) : IBufferView<ICPUBuffer>(_underlying,_format) {}

        core::smart_refctd_ptr<IAsset> clone(uint32_t _depth = ~0u) const override
        {
            auto buf = (_depth > 0u && m_buffer) ? core::smart_refctd_ptr_static_cast<ICPUBuffer>(m_buffer->clone(_depth-1u)) : m_buffer;
			return core::make_smart_refctd_ptr<ICPUBufferView>(SBufferRange<ICPUBuffer>{m_offset,m_size,m_buffer},m_format);
        }

		constexpr static inline bool HasDependents = true;

		constexpr static inline auto AssetType = ET_BUFFER_VIEW;
		inline IAsset::E_TYPE getAssetType() const override { return AssetType; }

		ICPUBuffer* getUnderlyingBuffer() 
		{
			assert(isMutable());
			return m_buffer.get(); 
		}
		const ICPUBuffer* getUnderlyingBuffer() const { return m_buffer.get(); }

		inline void setOffsetInBuffer(size_t _offset) 
		{
			assert(isMutable());
			m_offset = _offset;
		}
		inline void setSize(size_t _size) 
		{
			assert(isMutable());
			m_size = _size;
		}

	protected:
		virtual ~ICPUBufferView() = default;
};

}
#endif