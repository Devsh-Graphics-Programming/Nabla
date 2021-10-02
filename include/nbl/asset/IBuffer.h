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
            EUF_NONE = 0x00000000,
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

        // TODO: @achal boot this to IGPUBuffer (but add `core::bitflag<E_USAGE_FLAGS> m_usage` along with getters,setter and adders to ICPU)
        struct SCreationParams
        {
            core::bitflag<E_USAGE_FLAGS> usage = EUF_NONE;
            E_SHARING_MODE sharingMode = ESM_CONCURRENT;
            uint32_t queueFamilyIndexCount = 0u;
            const uint32_t* queueFamilyIndices = nullptr;
        };

	protected:
		IBuffer() = default;
		virtual ~IBuffer() = default;
};

template<class BufferType>
struct SBufferBinding
{
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
};

}

#endif