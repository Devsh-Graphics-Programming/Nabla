// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_BUFFER_VIEW_H_INCLUDED_
#define _NBL_ASSET_I_BUFFER_VIEW_H_INCLUDED_

#include "nbl/macros.h"

#include "nbl/asset/format/EFormat.h"
#include "nbl/asset/IBuffer.h"

namespace nbl::asset
{

template<typename BufferType>
class IBufferView : public IDescriptor
{
	public:		
		E_CATEGORY getTypeCategory() const override { return EC_BUFFER_VIEW; }

		const BufferType* getUnderlyingBuffer() const { return m_buffer.get(); }

		E_FORMAT getFormat() const { return m_format; }
		size_t getOffsetInBuffer() const { return m_offset; }
		size_t getByteSize() const { return m_size; }

	protected:
		IBufferView(const SBufferRange<BufferType>& underlying, E_FORMAT _format) : m_size(underlying.size), m_offset(underlying.offset), m_buffer(underlying.buffer), m_format(_format)
		{
			if (m_size==asset::SBufferRange<BufferType>::WholeBuffer)
				m_size = m_buffer->getSize()-m_offset;
		}
		virtual ~IBufferView() = default;

		// TODO: change to asset::SBufferRange<BufferType> later
		size_t m_offset;
		size_t m_size;
		core::smart_refctd_ptr<BufferType> m_buffer;
		E_FORMAT m_format;

};

}
#endif