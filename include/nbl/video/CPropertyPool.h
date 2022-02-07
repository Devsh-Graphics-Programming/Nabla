// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_C_PROPERTY_POOL_H_INCLUDED__
#define __NBL_VIDEO_C_PROPERTY_POOL_H_INCLUDED__

#include "nbl/video/IPropertyPool.h"

namespace nbl
{
namespace video
{
template<template<class...> class allocator = core::allocator, typename... Properties>
class CPropertyPool final : public IPropertyPool
{
    using this_t = CPropertyPool<allocator, Properties...>;

    static auto propertyCombinedSize()
    {
        return (sizeof(Properties) + ...);
    }

    _NBL_STATIC_INLINE_CONSTEXPR auto PropertyCount = sizeof...(Properties);

public:
    static inline uint32_t calcApproximateCapacity(size_t bufferSize)
    {
        return bufferSize / propertyCombinedSize();
    }

    static inline core::smart_refctd_ptr<this_t> create(asset::SBufferRange<IGPUBuffer>&& _memoryBlock, allocator<uint8_t>&& alloc = allocator<uint8_t>())
    {
        const auto reservedSize = getReservedSize(calcApproximateCapacity(_memoryBlock.size));
        auto reserved = std::allocator_traits<allocator<uint8_t>>::allocate(alloc, reservedSize);
        if(!reserved)
            return nullptr;

        auto retval = create(std::move(_memoryBlock), reserved, std::move(alloc));
        if(!retval)
            std::allocator_traits<allocator<uint8_t>>::deallocate(alloc, reserved, reservedSize);

        return retval;
    }
    // if this method fails to create the pool, the callee must free the reserved memory themselves, also the reserved pointer must be compatible with the allocator so it can free it
    static inline core::smart_refctd_ptr<this_t> create(asset::SBufferRange<IGPUBuffer>&& _memoryBlock, void* reserved, allocator<uint8_t>&& alloc = allocator<uint8_t>())
    {
        assert(_memoryBlock.isValid());
        assert(reserved);

        const size_t approximateCapacity = calcApproximateCapacity(_memoryBlock.size);
        auto capacity = approximateCapacity;
        while(capacity)
        {
            auto wouldBeSize = PropertySizes[0] * capacity;
            // now compute with padding and alignments
            for(auto i = 1; i < PropertyCount; i++)
            {
                // align
                wouldBeSize = core::roundUp(wouldBeSize, PropertySizes[i]);
                // increase
                wouldBeSize += PropertySizes[i] * capacity;
            }
            // if still manage to fit, then ok
            if(wouldBeSize <= _memoryBlock.size)
                break;
            capacity--;
        }
        capacity = core::min<decltype(capacity)>(~uint32_t(0), capacity);
        if(!capacity)
            return nullptr;

        auto* pool = new CPropertyPool(std::move(_memoryBlock), static_cast<uint32_t>(capacity), reserved, std::move(alloc));
        return core::smart_refctd_ptr<CPropertyPool>(pool, core::dont_grab);
    }

    //
    uint32_t getPropertyCount() const override { return PropertyCount; }
    size_t getPropertyOffset(uint32_t ix) const override { return propertyOffsets[ix]; }
    uint32_t getPropertySize(uint32_t ix) const override { return static_cast<uint32_t>(PropertySizes[ix]); }

protected:
    CPropertyPool(asset::SBufferRange<IGPUBuffer>&& _memoryBlock, uint32_t capacity, void* _reserved, allocator<uint8_t>&& _alloc)
        : IPropertyPool(std::move(_memoryBlock), capacity, _reserved), alloc(std::move(_alloc)), reserved(_reserved)
    {
        propertyOffsets[0] = memoryBlock.offset;
        for(uint32_t i = 1u; i < PropertyCount; i++)
            propertyOffsets[i] = core::roundUp(propertyOffsets[i - 1u] + PropertySizes[i - 1u] * size_t(capacity), PropertySizes[i]);
    }

    ~CPropertyPool()
    {
        std::allocator_traits<allocator<uint8_t>>::deallocate(alloc, reinterpret_cast<uint8_t*>(reserved), getReservedSize(getCapacity()));
    }

    allocator<uint8_t> alloc;
    void* reserved;
    size_t propertyOffsets[PropertyCount];

    _NBL_STATIC_INLINE_CONSTEXPR std::array<size_t, PropertyCount> PropertySizes = {sizeof(Properties)...};
};

}
}

#endif