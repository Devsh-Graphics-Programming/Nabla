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

class IBuffer : public core::IBuffer, public IDescriptor
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
            // we will not expose transform feedback
			//EUF_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT = 0x00000800,
            //EUF_TRANSFORM_FEEDBACK_COUNTER_BUFFER_BIT_EXT = 0x00001000,
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
struct SBufferBinding
{
	size_t offset = 0ull;
	core::smart_refctd_ptr<BufferType> buffer = nullptr;

	
	inline operator SBufferBinding<const BufferType>&() {return *reinterpret_cast<SBufferBinding<const BufferType>*>(this);}
	inline operator const SBufferBinding<const BufferType>&() const {return *reinterpret_cast<const SBufferBinding<const BufferType>*>(this);}

	inline bool isValid() const
	{
		return buffer && (offset<buffer->getSize());
	}

	inline bool operator==(const SBufferBinding<const BufferType>& rhs) const { return buffer==rhs.buffer && offset==rhs.offset; }
	inline bool operator!=(const SBufferBinding<const BufferType>& rhs) const { return !operator==(rhs); }
};

template<typename BufferType>
struct SBufferRange
{
	static constexpr inline size_t WholeBuffer = ~0ull;

	size_t offset = 0ull;
	size_t size = WholeBuffer;
	core::smart_refctd_ptr<BufferType> buffer = nullptr;
	
	
	inline operator SBufferRange<const BufferType>&() {return *reinterpret_cast<SBufferRange<const BufferType>*>(this);}
	inline operator const SBufferRange<const BufferType>&() const {return *reinterpret_cast<const SBufferRange<const BufferType>*>(this);}

	inline bool isValid() const
	{
		if (!buffer || offset>=buffer->getSize() || size==0ull)
			return false;
		return actualSize()<=buffer->getSize()-offset;
	}

	inline size_t actualSize() const
	{
		return size!=WholeBuffer ? size:buffer->getSize();
	}
	inline bool operator==(const SBufferRange<const BufferType>& rhs) const { return buffer==rhs.buffer && offset==rhs.offset && actualSize()==rhs.actualSize(); }
	inline bool operator!=(const SBufferRange<const BufferType>& rhs) const { return !operator==(rhs); }
};

}

#endif