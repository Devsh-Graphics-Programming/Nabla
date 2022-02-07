// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_CORE_C_DOUBLE_BUFFERING_ALLOCATOR_H__
#define __NBL_CORE_C_DOUBLE_BUFFERING_ALLOCATOR_H__

namespace nbl
{
namespace core
{
template<class AddressAllocator, bool onlySwapRangesMarkedDirty>
class MultiBufferingAllocatorBase;

//! Specializations for marking dirty areas for swapping
template<class AddressAllocator>
class MultiBufferingAllocatorBase<AddressAllocator, false>
{
protected:
    std::pair<typename AddressAllocator::size_type, typename AddressAllocator::size_type> pushRange;
    std::pair<typename AddressAllocator::size_type, typename AddressAllocator::size_type> pullRange;

    MultiBufferingAllocatorBase()
        : pushRange(0u, ~static_cast<typename AddressAllocator::size_type>(0u)), pullRange(0u, ~static_cast<typename AddressAllocator::size_type>(0u)) {}
    virtual ~MultiBufferingAllocatorBase() {}

    inline void resetPushRange() {}
    inline void resetPullRange() {}

    _NBL_STATIC_INLINE_CONSTEXPR bool alwaysSwapEntireRange = true;

public:
    inline void markRangeForPush(typename AddressAllocator::size_type begin, typename AddressAllocator::size_type end)
    {
    }
    inline void markRangeForPull(typename AddressAllocator::size_type begin, typename AddressAllocator::size_type end)
    {
    }
};
template<class AddressAllocator>
class MultiBufferingAllocatorBase<AddressAllocator, true>
{
protected:
    std::pair<typename AddressAllocator::size_type, typename AddressAllocator::size_type> pushRange;
    std::pair<typename AddressAllocator::size_type, typename AddressAllocator::size_type> pullRange;

    MultiBufferingAllocatorBase()
    {
        resetPushRange();
        resetPullRange();
    }
    virtual ~MultiBufferingAllocatorBase() {}

    inline void resetPushRange()
    {
        pushRange.first = ~static_cast<typename AddressAllocator::size_type>(0u);
        pushRange.second = 0u;
    }
    inline void resetPullRange()
    {
        pullRange.first = ~static_cast<typename AddressAllocator::size_type>(0u);
        pullRange.second = 0u;
    }

    _NBL_STATIC_INLINE_CONSTEXPR bool alwaysSwapEntireRange = false;

public:
    inline decltype(pushRange) getPushRange() const { return pushRange; }
    inline decltype(pullRange) getPullRange() const { return pullRange; }

    inline void markRangeForPush(typename AddressAllocator::size_type begin, typename AddressAllocator::size_type end)
    {
        if(begin < pushRange.first)
            pushRange.first = begin;
        if(end > pushRange.second)
            pushRange.second = end;
    }
    inline void markRangeForPull(typename AddressAllocator::size_type begin, typename AddressAllocator::size_type end)
    {
        if(begin < pushRange.first)
            pushRange.first = begin;
        if(end > pushRange.second)
            pushRange.second = end;
    }
};

}
}

#endif
