// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_CPU_BUFFER_H_INCLUDED__
#define __NBL_ASSET_I_CPU_BUFFER_H_INCLUDED__

#include <type_traits>

#include "nbl/core/alloc/null_allocator.h"

#include "nbl/asset/IBuffer.h"
#include "nbl/asset/IAsset.h"
#include "nbl/asset/IDescriptor.h"
#include "nbl/asset/bawformat/blobs/RawBufferBlob.h"

namespace nbl::asset
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
            if (!canBeConvertedToDummy())
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
        virtual uint64_t getSize() const override {return size;}

		//! Returns pointer to data.
        virtual const void* getPointer() const {return data;}
        virtual void* getPointer() 
        { 
            assert(!isImmutable_debug());
            return data; 
        }

        bool canBeRestoredFrom(const IAsset* _other) const override
        {
            auto* other = static_cast<const ICPUBuffer*>(_other);
            if (size != other->size)
                return false;

            return true;
        }
        
		inline core::bitflag<E_USAGE_FLAGS> getUsageFlags() const
		{
			return usage;
		}
		inline bool setUsageFlags(core::bitflag<E_USAGE_FLAGS> _usage)
		{
			assert(!isImmutable_debug());
			usage = _usage;
			return true;
		}
		inline bool addUsageFlags(core::bitflag<E_USAGE_FLAGS> _usage)
		{
			assert(!isImmutable_debug());
			usage |= _usage;
			return true;
		}
        
		inline bool getCanUpdateSubRange() const {return canUpdateSubRange;}
		inline bool setCanUpdateSubRange(const bool _canUpdateSubRange)
		{
			assert(!isImmutable_debug());
            canUpdateSubRange = _canUpdateSubRange;
			return true;
		}

    protected:
        void restoreFromDummy_impl(IAsset* _other, uint32_t _levelsBelow) override
        {
            auto* other = static_cast<ICPUBuffer*>(_other);

            if (willBeRestoredFrom(_other))
                std::swap(data, other->data);
        }

        uint64_t size;
        void* data;
        // this is a bit weird, but makes sense because the usages are for the IGPUBuffer that will be created from the data stored here
        core::bitflag<E_USAGE_FLAGS> usage = EUF_TRANSFER_DST_BIT;
        // whether `IGPUCommandBuffer::updateBuffer` can be used
        // TODO: in the new CPU2GPU converter make sure to ||= this value for all buffers
        bool canUpdateSubRange = false;
};

template<
    typename Allocator = _NBL_DEFAULT_ALLOCATOR_METATYPE<uint8_t>,
    bool = std::is_same<Allocator, core::null_allocator<typename Allocator::value_type> >::value
>
class CCustomAllocatorCPUBuffer;

//! Specialization of ICPUBuffer capable of taking custom allocators
/*
    Take a look that with this usage you have to specify custom alloctor
    passing an object type for allocation and a pointer to allocated
    data for it's storage by ICPUBuffer.

    So the need for the class existence is for common following tricks - among others creating an 
    \bICPUBuffer\b over an already existing \bvoid*\b array without any \imemcpy\i or \itaking over the memory ownership\i.
    You can use it with a \bnull_allocator\b that adopts memory (it is a bit counter intuitive because \badopt = take\b ownership, 
    but a \inull allocator\i doesn't do anything, even free the memory, so you're all good).
*/

template<typename Allocator>
class CCustomAllocatorCPUBuffer<Allocator, true> : public ICPUBuffer
{
		static_assert(sizeof(typename Allocator::value_type) == 1u, "Allocator::value_type must be of size 1");
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
            if (!canBeConvertedToDummy())
                return;

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

} // end namespace nbl::asset

#endif
