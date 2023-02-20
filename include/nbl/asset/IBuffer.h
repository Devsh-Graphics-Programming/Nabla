// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_BUFFER_H_INCLUDED_
#define _NBL_ASSET_I_BUFFER_H_INCLUDED_

#include "nbl/core/decl/smart_refctd_ptr.h"
#include "nbl/core/IBuffer.h"
#include "nbl/core/util/bitflag.h"

#include "nbl/asset/IDescriptor.h"

namespace nbl::asset
{

class NBL_API IBuffer : public core::IBuffer, public IDescriptor
{
	public:
		E_CATEGORY getTypeCategory() const override {return EC_BUFFER;}

		//!
        enum E_USAGE_FLAGS : uint32_t
        {
            EUF_NONE = 0x00000000,
            EUF_TRANSFER_SRC_BIT = 0x00000001,
            EUF_TRANSFER_DST_BIT = 0x00000002,
            EUF_UNIFORM_TEXEL_BUFFER_BIT = 0x00000004,
            EUF_STORAGE_TEXEL_BUFFER_BIT = 0x00000008,
            EUF_UNIFORM_BUFFER_BIT = 0x00000010,
            EUF_STORAGE_BUFFER_BIT = 0x00000020,
            EUF_INDEX_BUFFER_BIT = 0x00000040,
            EUF_VERTEX_BUFFER_BIT = 0x00000080,
            EUF_INDIRECT_BUFFER_BIT = 0x00000100,
            EUF_SHADER_DEVICE_ADDRESS_BIT = 0x00020000,
            EUF_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT = 0x00000800,
            EUF_TRANSFORM_FEEDBACK_COUNTER_BUFFER_BIT_EXT = 0x00001000,
            EUF_CONDITIONAL_RENDERING_BIT_EXT = 0x00000200,
            EUF_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT = 0x00080000,
            EUF_ACCELERATION_STRUCTURE_STORAGE_BIT = 0x00100000,
            EUF_SHADER_BINDING_TABLE_BIT = 0x00000400,
			//! synthetic Nabla inventions
			// whether `IGPUCommandBuffer::updateBuffer` can be used on this buffer
			EUF_INLINE_UPDATE_VIA_CMDBUF = 0x80000000u,
        };

		//!
		struct SCreationParams
		{
			size_t size = 0ull;
			core::bitflag<E_USAGE_FLAGS> usage = EUF_NONE;
		};

		//!
		inline const SCreationParams& getCreationParams() const {return m_creationParams;}

		//! Returns size in bytes.
		uint64_t getSize() const override { return m_creationParams.size; }

	protected:
		IBuffer(const SCreationParams& _creationParams) : m_creationParams(_creationParams) {}
		virtual ~IBuffer() = default;

		SCreationParams m_creationParams;
};

NBL_ENUM_ADD_BITWISE_OPERATORS(IBuffer::E_USAGE_FLAGS)

template<class BufferType>
struct NBL_API SBufferBinding
{
	bool isValid() const
	{
		return buffer && (offset<buffer->getSize());
	}

	uint64_t offset = 0ull;
	core::smart_refctd_ptr<BufferType> buffer = nullptr;

	inline bool operator==(const SBufferBinding<BufferType>& rhs) const { return buffer==rhs.buffer && offset==rhs.offset; }
	inline bool operator!=(const SBufferBinding<BufferType>& rhs) const { return !operator==(rhs); }
};

template<typename BufferType>
struct NBL_API SBufferRange
{
	// Temp Fix, If you want to uncomment this then fix every example having compile issues -> add core::smart_refctd_ptr around the buffer to be an r-value ref
	// SBufferRange(const size_t& _offset, const size_t& _size, core::smart_refctd_ptr<BufferType>&& _buffer)
	// 	: offset(_offset), size(_size), buffer(core::smart_refctd_ptr<BufferType>(_buffer)) {}
	// SBufferRange() : offset(0ull), size(0ull), buffer(nullptr) {}

	inline bool isValid() const
	{
		return buffer && size && (offset+size<=buffer->getSize());
	}

	size_t offset = 0ull;
	size_t size = 0ull;
	core::smart_refctd_ptr<BufferType> buffer = nullptr;

	inline bool operator==(const SBufferRange<BufferType>& rhs) const { return buffer==rhs.buffer && offset==rhs.offset && size==rhs.size; }
	inline bool operator!=(const SBufferRange<BufferType>& rhs) const { return !operator==(rhs); }
};

}

#endif