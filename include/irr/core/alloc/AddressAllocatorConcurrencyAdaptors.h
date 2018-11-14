// Copyright (C) 2018 Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW Engine"
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_ADDRESS_ALLOCATOR_CONCURRENCY_ADAPTORS_H_INCLUDED__
#define __IRR_ADDRESS_ALLOCATOR_CONCURRENCY_ADAPTORS_H_INCLUDED__

#include "IrrCompileConfig.h"

#include "irr/core/alloc/address_allocator_traits.h"

namespace irr
{
namespace core
{


//! TODO: Solve priority inversion issue by providing a default FIFO lock/mutex
template<class AddressAllocator, class RecursiveLockable>
class AddressAllocatorBasicConcurrencyAdaptor : private AddressAllocator
{
        static_assert(std::is_standard_layout<RecursiveLockable>::value,"Lock class is not standard layout");
        RecursiveLockable lock;

        AddressAllocator& getBaseRef() {return reinterpret_cast<AddressAllocator&>(*this);}
    public:
        _IRR_DECLARE_ADDRESS_ALLOCATOR_TYPEDEFS(typename AddressAllocator::size_type);

        typedef address_allocator_traits<AddressAllocator>              traits;
        typedef typename traits::has_supportsArbitraryOrderFrees    has_supportsArbitraryOrderFrees;
        static_assert(impl::address_allocator_traits_base<AddressAllocator,traits::has_supportsArbitraryOrderFrees::value>::supportsArbitraryOrderFrees,"AddressAllocator does not support arbitrary order frees!");


        using AddressAllocator::AddressAllocator;
        virtual ~AddressAllocatorBasicConcurrencyAdaptor() {}

        inline size_type    get_real_addr(size_type allocated_addr) const noexcept
        {
            lock.lock();
            auto retval = traits::get_real_addr(getBaseRef(),allocated_addr);
            lock.unlock();
            return retval;
        }

        template<typename... Args>
        inline void         multi_alloc_addr(Args&&... args) noexcept
        {
            lock.lock();
            traits::multi_alloc_addr(getBaseRef(),std::forward<Args>(args)...);
            lock.unlock();
        }
        template<typename... Args>
        inline void         multi_free_addr(Args&&... args) noexcept
        {
            lock.lock();
            traits::multi_free_addr(getBaseRef(),std::forward<Args>(args)...);
            lock.unlock();
        }

        inline void         reset() noexcept
        {
            lock.lock();
            AddressAllocator::reset();
            lock.unlock();
        }

        //! Conservative estimate, max_size() gives largest size we are sure to be able to allocate
        inline size_type    max_size() const noexcept
        {
            lock.lock();
            auto retval = AddressAllocator::max_size();
            lock.unlock();
            return retval;
        }

        //! Most allocators do not support e.g. 1-byte allocations
        inline size_type    min_size() const noexcept
        {
            lock.lock();
            auto retval = AddressAllocator::min_size();
            lock.unlock();
            return retval;
        }

        inline size_type    max_alignment() const noexcept
        {
            lock.lock();
            auto retval = AddressAllocator::max_alignment();
            lock.unlock();
            return retval;
        }

        inline size_type    safe_shrink_size(size_type bound=0u, size_type newBuffAlignmentWeCanGuarantee=1u) const noexcept
        {
            lock.lock();
            auto retval = AddressAllocator::safe_shrink_size(bound,newBuffAlignmentWeCanGuarantee);
            lock.unlock();
            return retval;
        }

        template<typename... Args>
        static inline size_type reserved_size(const Args&... args) noexcept
        {
            return AddressAllocator::reserved_size(args...);
        }
        static inline size_type reserved_size(size_type bufSz, const AddressAllocator& other) noexcept
        {
            return AddressAllocator::reserved_size(bufSz,other);
        }


        //! Extra == Use WIHT EXTREME CAUTION
        inline RecursiveLockable&   get_lock() noexcept
        {
            return lock;
        }
};

}
}

#endif // __IRR_ADDRESS_ALLOCATOR_CONCURRENCY_ADAPTORS_H_INCLUDED__
