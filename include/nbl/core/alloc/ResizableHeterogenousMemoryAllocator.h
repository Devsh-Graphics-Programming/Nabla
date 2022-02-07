// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_CORE_RESIZABLE_HETEROGENOUS_MEMORY_ALLOCATOR_H___
#define __NBL_CORE_RESIZABLE_HETEROGENOUS_MEMORY_ALLOCATOR_H___

#include "nbl/core/alloc/HeterogenousMemoryAddressAllocatorAdaptor.h"

#include "nbl/core/alloc/PoolAddressAllocator.h"

namespace nbl
{
namespace core
{
// if only we could use c++17 if-constexpr
namespace impl
{
template<class HeterogenousMemoryAllocator>
class ResizableHeterogenousMemoryAllocatorBase;

template<class HeterogenousMemoryAllocator>
class ResizableHeterogenousMemoryAllocatorBase : public HeterogenousMemoryAllocator  // make protected?
{
protected:
    typedef HeterogenousMemoryAllocator Base;

    using Base::Base;

    inline void resizeAddressAllocator(void* newReserved)
    {
        using alloc_traits = typename Base::alloc_traits;
        using AddressAllocator = typename alloc_traits::allocator_type;

        AddressAllocator& mAddrAlloc = Base::getBaseAddrAllocRef();
        mAddrAlloc = AddressAllocator(Base::mDataSize, std::move(mAddrAlloc), newReserved);
    }
};
}

template<class HeterogenousMemoryAllocator>
class ResizableHeterogenousMemoryAllocator : public impl::ResizableHeterogenousMemoryAllocatorBase<HeterogenousMemoryAllocator>  // make protected?
{
    typedef impl::ResizableHeterogenousMemoryAllocatorBase<HeterogenousMemoryAllocator> Base;
    typedef typename Base::alloc_traits alloc_traits;

    typedef typename Base::alloc_traits::allocator_type AddressAllocator;

public:
    typedef ResizableHeterogenousMemoryAllocator<HeterogenousMemoryAllocator> ThisType;

    typedef typename HeterogenousMemoryAllocator::size_type size_type;

    template<typename... Args>
    ResizableHeterogenousMemoryAllocator(Args&&... args)
        : Base(std::forward<Args>(args)...), growPolicy(defaultGrowPolicy), shrinkPolicy(defaultShrinkPolicy)
    {
    }

    virtual ~ResizableHeterogenousMemoryAllocator() {}

    //!
    /** Warning outAddresses needs to be primed with `invalid_address` values,
        otherwise no allocation happens for elements not equal to `invalid_address`. */
    template<typename... Args>
    inline void multi_alloc_addr(uint32_t count, size_type* outAddresses, const size_type* bytes, const size_type* alignment, const Args&... args)
    {
        AddressAllocator& mAddrAlloc = Base::getBaseAddrAllocRef();

        Base::multi_alloc_addr(count, outAddresses, bytes, alignment, args...);

        size_type totalNeededExtraNewMem = 0u;
        for(uint32_t i = 0u; i < count; i++)
        {
            if(outAddresses[i] != AddressAllocator::invalid_address)
                continue;

            totalNeededExtraNewMem += std::max(bytes[i], alloc_traits::min_size(mAddrAlloc)) + alignment[i] - 1u;
        }

        if(totalNeededExtraNewMem == size_type(0u))
            return;

        size_type allAllocatorSpace = alloc_traits::get_total_size(mAddrAlloc) - alloc_traits::get_align_offset(mAddrAlloc);
        size_type newSize = growPolicy(this, totalNeededExtraNewMem);
        newSize = std::max(newSize, alloc_traits::get_allocated_size(mAddrAlloc) + totalNeededExtraNewMem);

        if(newSize <= allAllocatorSpace)
            return;

        auto maxAllocatableAlignment = alloc_traits::max_alignment(mAddrAlloc);  // for now, until we actually make Buffers from Vulkan memory
        newSize = mAddrAlloc.safe_shrink_size(newSize, maxAllocatableAlignment);  // for padding

        //resize
        size_type oldReservedSize = Base::mReservedSize;
        const void* oldReserved = alloc_traits::getReservedSpacePtr(mAddrAlloc);

        Base::mReservedSize = alloc_traits::reserved_size(mAddrAlloc, newSize);
        void* newReserved = Base::mReservedAlloc.allocate(Base::mReservedSize, _NBL_SIMD_ALIGNMENT);

        Base::mDataSize = newSize;
        Base::mDataAlloc.reallocate(Base::mAllocation, Base::mDataSize, maxAllocatableAlignment, mAddrAlloc);
        Base::resizeAddressAllocator(newReserved);

        if(oldReserved)
            Base::mReservedAlloc.deallocate(reinterpret_cast<uint8_t*>(const_cast<void*>(oldReserved)), oldReservedSize);

        Base::multi_alloc_addr(count, outAddresses, bytes, alignment, args...);
    }

    template<typename... Args>
    inline void multi_free_addr(Args&&... args)
    {
        AddressAllocator& mAddrAlloc = Base::getBaseAddrAllocRef();

        Base::multi_free_addr(std::forward<Args>(args)...);

        size_type allAllocatorSpace = alloc_traits::get_total_size(mAddrAlloc) - alloc_traits::get_align_offset(mAddrAlloc);
        size_type newSize = shrinkPolicy(this);
        if(newSize >= allAllocatorSpace)
            return;

        // some allocators may not be shrinkable because of fragmentation
        auto maxAllocatableAlignment = alloc_traits::max_alignment(mAddrAlloc);  // for now, until we actually make Buffers from Vulkan memory
        newSize = mAddrAlloc.safe_shrink_size(newSize, maxAllocatableAlignment);
        if(newSize >= allAllocatorSpace)
            return;

        // resize
        size_type oldReservedSize = Base::mReservedSize;
        const void* oldReserved = alloc_traits::getReservedSpacePtr(mAddrAlloc);

        Base::mReservedSize = alloc_traits::reserved_size(mAddrAlloc, newSize);
        void* newReserved = Base::mReservedAlloc.allocate(Base::mReservedSize, _NBL_SIMD_ALIGNMENT);

        Base::mDataSize = newSize;
        Base::mDataAlloc.reallocate(Base::mAllocation, Base::mDataSize, maxAllocatableAlignment, mAddrAlloc);
        Base::resizeAddressAllocator(newReserved);

        if(oldReserved)
            Base::mReservedAlloc.deallocate(reinterpret_cast<uint8_t*>(const_cast<void*>(oldReserved)), oldReservedSize);
    }

protected:
    constexpr static size_type defaultGrowStep = 32u * 4096u;  //128k at a time
    constexpr static size_type defaultGrowStepMinus1 = defaultGrowStep - 1u;
    static inline size_type defaultGrowPolicy(ThisType* _this, size_type totalRequestedNewMem)
    {
        const auto& underlyingAddrAlloc = _this->getAddressAllocator();

        size_type allAllocatorSpace = alloc_traits::get_total_size(underlyingAddrAlloc) - alloc_traits::get_align_offset(underlyingAddrAlloc);
        size_type nextAllocTotal = alloc_traits::get_allocated_size(underlyingAddrAlloc) + totalRequestedNewMem;
        if(nextAllocTotal > allAllocatorSpace)
            return (nextAllocTotal + defaultGrowStepMinus1) & (~defaultGrowStepMinus1);

        return allAllocatorSpace;
    }
    static inline size_type defaultShrinkPolicy(ThisType* _this)
    {
        constexpr size_type shrinkStep = 256u * 4096u;  //1M at a time

        const auto& underlyingAddrAlloc = _this->getAddressAllocator();

        size_type allFreeSpace = alloc_traits::get_free_size(underlyingAddrAlloc);
        if(allFreeSpace > shrinkStep)
            return (alloc_traits::get_allocated_size(underlyingAddrAlloc) + defaultGrowStepMinus1) & (~defaultGrowStepMinus1);

        return alloc_traits::get_total_size(underlyingAddrAlloc) - alloc_traits::get_align_offset(underlyingAddrAlloc);
    }

    size_type (*growPolicy)(ThisType*, size_type);
    size_type (*shrinkPolicy)(ThisType*);

public:
    //! Grow Policies return
    inline const decltype(growPolicy)& getGrowPolicy() const { return growPolicy; }
    inline void setGrowPolicy(const decltype(growPolicy)& newGrowPolicy) { growPolicy = newGrowPolicy; }

    inline const decltype(shrinkPolicy)& getShrinkPolicy() const { return shrinkPolicy; }
    inline void setShrinkPolicy(const decltype(shrinkPolicy)& newShrinkPolicy) { shrinkPolicy = newShrinkPolicy; }
};

}
}

#endif
