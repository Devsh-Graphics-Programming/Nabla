// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_BUFFER_H_INCLUDED__
#define __NBL_ASSET_I_BUFFER_H_INCLUDED__

#include "nbl/core/decl/smart_refctd_ptr.h"
#include "nbl/core/IBuffer.h"

#include "nbl/asset/IDescriptor.h"

namespace nbl::asset
{

class IBuffer : public core::IBuffer, public IDescriptor
{
	public:
		E_CATEGORY getTypeCategory() const override { return EC_BUFFER; }

        enum E_USAGE_FLAGS : uint32_t
        {
            EUF_TRANSFER_SRC_BIT = 0x00000001,
            EUF_TRANSFER_DST_BIT = 0x00000002,
            EUF_UNIFORM_TEXEL_BUFFER_BIT = 0x00000004,
            EUF_STORAGE_TEXEL_BUFFER_BIT = 0x00000008,
            EUF_UNIFORM_BUFFER_BIT = 0x00000010,
            EUF_STORAGE_BUFFER_BIT = 0x00000020,
            EUF_INDEX_BUFFER_BIT = 0x00000040,
            EUF_VERTEX_BUFFER_BIT = 0x00000080,
            EUF_INDIRECT_BUFFER_BIT = 0x00000100
        };

        struct SCreationParams
        {
            uint64_t size;
            E_USAGE_FLAGS usage;
            E_SHARING_MODE sharingMode;
            uint32_t queueFamilyIndexCount;
            const uint32_t* queuueFamilyIndices;
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
	bool isValid() const
	{
		return buffer && (offset+size<=buffer->getSize());
	}

	size_t offset = 0ull;
	size_t size = 0ull;
	core::smart_refctd_ptr<BufferType> buffer = nullptr;
	inline bool operator==(const SBufferRange<BufferType>& rhs) const { return buffer==rhs.buffer && offset==rhs.offset && size==rhs.size; }
	inline bool operator!=(const SBufferRange<BufferType>& rhs) const { return !operator==(rhs); }
};

}

#endif