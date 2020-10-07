// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_CPU_BUFFER_H_INCLUDED__
#define __NBL_ASSET_I_CPU_BUFFER_H_INCLUDED__

#include <type_traits>

#include "irr/core/alloc/null_allocator.h"

#include "irr/asset/IBuffer.h"
#include "irr/asset/IAsset.h"
#include "irr/asset/IDescriptor.h"
#include "irr/asset/bawformat/blobs/RawBufferBlob.h"

namespace irr
{
namespace asset
{

//! One of CPU class-object representing an Asset
/**
	One of Assets used for storage of large arrays, so that storage can be decoupled
	from other objects such as meshbuffers, images, animations and shader source/bytecode.

	@see IAsset
*/
class ICPUBuffer : public asset::IBuffer, public asset::IAsset
{
    protected:
        virtual ~ICPUBuffer()
        {
            this->convertToDummyObject();
        }

        //! Non-allocating constructor for CCustormAllocatorCPUBuffer derivative
        ICPUBuffer(size_t sizeInBytes, void* dat) : size(dat ? sizeInBytes : 0), data(dat)
        {}
    public:
		//! Constructor.
		/** @param sizeInBytes Size in bytes. If `dat` argument is present, it denotes size of data pointed by `dat`, otherwise - size of data to be allocated.
		*/
        ICPUBuffer(size_t sizeInBytes) : size(0)
        {
			data = _NBL_ALIGNED_MALLOC(sizeInBytes,_NBL_SIMD_ALIGNMENT);
            if (!data)
                return;

            size = sizeInBytes;
        }

        core::smart_refctd_ptr<IAsset> clone(uint32_t = ~0u) const override
        {
            auto cp = core::make_smart_refctd_ptr<ICPUBuffer>(size);
            clone_common(cp.get());
            memcpy(cp->getPointer(), data, size);

            return cp;
        }

        virtual void convertToDummyObject(uint32_t referenceLevelsBelowToConvert=0u) override
        {
            if (isDummyObjectForCacheAliasing)
                return;
            convertToDummyObject_common(referenceLevelsBelowToConvert);

            if (data)
                _NBL_ALIGNED_FREE(data);
            data = nullptr;
            size = 0ull;
            isDummyObjectForCacheAliasing = true;
        }

        _NBL_STATIC_INLINE_CONSTEXPR auto AssetType = ET_BUFFER;
        inline E_TYPE getAssetType() const override { return AssetType; }

        virtual size_t conservativeSizeEstimate() const override { return getSize(); }

        //! Returns size in bytes.
        virtual const uint64_t& getSize() const {return size;}

		//! Returns pointer to data.
        /** WARNING: RESIZE will invalidate pointer.
		*/
        virtual const void* getPointer() const {return data;}
		//! Returns pointer to data.
		/** WARNING: RESIZE will invalidate pointer.
		*/
        virtual void* getPointer() {return data;}

    protected:
        uint64_t size;
        void* data;
};

template<
    typename Allocator = _IRR_DEFAULT_ALLOCATOR_METATYPE<uint8_t>,
    bool = std::is_same<Allocator, core::null_allocator<typename Allocator::value_type> >::value
>
class CCustomAllocatorCPUBuffer;

template<typename Allocator>
class CCustomAllocatorCPUBuffer<Allocator, true> : public ICPUBuffer
{
		static_assert(sizeof(Allocator::value_type) == 1u, "Allocator::value_type must be of size 1");
	protected:
		Allocator m_allocator;

        virtual ~CCustomAllocatorCPUBuffer()
        {
            this->convertToDummyObject();
        }

	public:
		CCustomAllocatorCPUBuffer(size_t sizeInBytes, void* dat, core::adopt_memory_t, Allocator&& alctr = Allocator()) : ICPUBuffer(sizeInBytes, dat), m_allocator(std::move(alctr))
		{
		}

		virtual void convertToDummyObject(uint32_t referenceLevelsBelowToConvert = 0u) override
		{
            if (isDummyObjectForCacheAliasing)
                return;
            convertToDummyObject_common(referenceLevelsBelowToConvert);

			if (ICPUBuffer::data)
				m_allocator.deallocate(reinterpret_cast<typename Allocator::pointer>(ICPUBuffer::data), ICPUBuffer::size);
			ICPUBuffer::data = nullptr; // so that ICPUBuffer won't try deallocating
		}
};

template<typename Allocator>
class CCustomAllocatorCPUBuffer<Allocator, false> : public CCustomAllocatorCPUBuffer<Allocator, true>
{
		using Base = CCustomAllocatorCPUBuffer<Allocator, true>;
	protected:
		virtual ~CCustomAllocatorCPUBuffer() = default;

	public:
		using Base::Base;

		CCustomAllocatorCPUBuffer(size_t sizeInBytes, const void* dat, Allocator&& alctr = Allocator()) : Base(sizeInBytes, alctr.allocate(sizeInBytes), core::adopt_memory, std::move(alctr))
		{
			memcpy(Base::data, dat, sizeInBytes);
		}
};

} // end namespace asset
} // end namespace irr

#endif
