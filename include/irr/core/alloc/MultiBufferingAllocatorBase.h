#ifndef __IRR_C_DOUBLE_BUFFERING_ALLOCATOR_H__
#define __IRR_C_DOUBLE_BUFFERING_ALLOCATOR_H__


#include "irr/core/alloc/AddressAllocatorBase.h"

namespace irr
{
namespace core
{

//! Specializations for marking dirty areas for swapping
template<class AddressAllocator, bool onlySwapRangesMarkedDirty>
class MultiBufferingAllocatorBase;
template<class AddressAllocator>
class MultiBufferingAllocatorBase<AddressAllocator,false>
{
    protected:
        std::pair<typename AddressAllocator::size_type,typename AddressAllocator::size_type> dirtyRange;

        MultiBufferingAllocatorBase() {}
        virtual ~MultiBufferingAllocatorBase() {}

        inline void resetDirtyRange() {}

        static constexpr bool alwaysSwapEntireRange = true;
};
template<class AddressAllocator>
class MultiBufferingAllocatorBase<AddressAllocator,true>
{
    protected:
        std::pair<typename AddressAllocator::size_type,typename AddressAllocator::size_type> dirtyRange;

        MultiBufferingAllocatorBase() {resetDirtyRange();}
        virtual ~MultiBufferingAllocatorBase() {}

        inline void resetDirtyRange() {dirtyRange.first = ~static_cast<typename AddressAllocator::size_type>(0u); dirtyRange.second = 0u;}

        static constexpr bool alwaysSwapEntireRange = false;
    public:
        inline decltype(dirtyRange) getDirtyRange() const {return dirtyRange;}

        inline void markRangeDirty(typename AddressAllocator::size_type begin, typename AddressAllocator::size_type end)
        {
            if (begin<dirtyRange.first) dirtyRange.first = begin;
            if (end>dirtyRange.second) dirtyRange.second = end;
        }
};

}
}

#endif // __IRR_C_DOUBLE_BUFFERING_ALLOCATOR_H__


