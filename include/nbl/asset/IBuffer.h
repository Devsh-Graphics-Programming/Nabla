// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_ASSET_I_BUFFER_H_INCLUDED_
#define _NBL_ASSET_I_BUFFER_H_INCLUDED_

#include "nbl/core/decl/smart_refctd_ptr.h"
#include "nbl/core/IBuffer.h"

#include "nbl/asset/IDescriptor.h"

namespace nbl::asset
{

class NBL_API IBuffer : public core::IBuffer, public IDescriptor
{
	public:
		E_CATEGORY getTypeCategory() const override { return EC_BUFFER; }

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
        };

	protected:
		IBuffer() = default;
		virtual ~IBuffer() = default;
};

template<class BufferType>
struct SBufferBinding
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
struct SBufferRange
{
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