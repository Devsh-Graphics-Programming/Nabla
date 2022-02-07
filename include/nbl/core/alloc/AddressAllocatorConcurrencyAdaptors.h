// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_CORE_ADDRESS_ALLOCATOR_CONCURRENCY_ADAPTORS_H_INCLUDED__
#define __NBL_CORE_ADDRESS_ALLOCATOR_CONCURRENCY_ADAPTORS_H_INCLUDED__

#include "nbl/core/alloc/address_allocator_traits.h"

namespace nbl
{
namespace core
{
//! TODO: Solve priority inversion issue by providing a default FIFO and recursive lock/mutex
template<class AddressAllocator, class RecursiveLockable>
class AddressAllocatorBasicConcurrencyAdaptor : private AddressAllocator
{
    static_assert(std::is_standard_layout<RecursiveLockable>::value, "Lock class is not standard layout");
    RecursiveLockable lock;

    AddressAllocator& getBaseRef() { return reinterpret_cast<AddressAllocator&>(*this); }

public:
    _NBL_DECLARE_ADDRESS_ALLOCATOR_TYPEDEFS(typename AddressAllocator::size_type);

    typedef address_allocator_traits<AddressAllocator> traits;
    static_assert(address_allocator_traits<AddressAllocator>::supportsArbitraryOrderFrees, "AddressAllocator does not support arbitrary order frees!");

    using AddressAllocator::AddressAllocator;
    virtual ~AddressAllocatorBasicConcurrencyAdaptor() {}

    inline size_type get_real_addr(size_type allocated_addr) const noexcept
    {
        lock.lock();
        auto retval = traits::get_real_addr(getBaseRef(), allocated_addr);
        lock.unlock();
        return retval;
    }

    template<typename... Args>
    inline void multi_alloc_addr(Args&&... args) noexcept
    {
        lock.lock();
        traits::multi_alloc_addr(getBaseRef(), std::forward<Args>(args)...);
        lock.unlock();
    }
    template<typename... Args>
    inline void multi_free_addr(Args&&... args) noexcept
    {
        lock.lock();
        traits::multi_free_addr(getBaseRef(), std::forward<Args>(args)...);
        lock.unlock();
    }

    inline void reset() noexcept
    {
        lock.lock();
        AddressAllocator::reset();
        lock.unlock();
    }

    //! Conservative estimate, max_size() gives largest size we are sure to be able to allocate
    inline size_type max_size() const noexcept
    {
        lock.lock();
        auto retval = AddressAllocator::max_size();
        lock.unlock();
        return retval;
    }

    //! Most address allocators do not support e.g. 1-byte allocations
    inline size_type min_size() const noexcept
    {
        lock.lock();
        auto retval = AddressAllocator::min_size();
        lock.unlock();
        return retval;
    }

    inline size_type max_alignment() const noexcept
    {
        lock.lock();
        auto retval = AddressAllocator::max_alignment();
        lock.unlock();
        return retval;
    }

    template<typename... Args>
    inline size_type safe_shrink_size(const Args&... args) const noexcept
    {
        lock.lock();
        auto retval = AddressAllocator::safe_shrink_size(args...);
        lock.unlock();
        return retval;
    }

    template<typename... Args>
    static inline size_type reserved_size(const Args&... args) noexcept
    {
        return AddressAllocator::reserved_size(args...);
    }

    //! Extra == USE WITH EXTREME CAUTION
    inline RecursiveLockable& get_lock() noexcept
    {
        // TODO: Some static assert to check that lock type is recursive
        return lock;
    }
};

}
}

#endif
