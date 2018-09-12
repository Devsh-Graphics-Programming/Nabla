// Copyright (C) 2018 Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW Engine"
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_ADDRESS_ALLOCATOR_CONCURRENCY_ADAPTORS_H_INCLUDED__
#define __IRR_ADDRESS_ALLOCATOR_CONCURRENCY_ADAPTORS_H_INCLUDED__

#include "IrrCompileConfig.h"

namespace irr
{
namespace core
{

template<class AddressAllocator, class BasicLockable>
class AddressAllocatorBasicConcurrencyAdaptor : private AddressAllocator
{
        static_assert(std::is_standard_layout<BasicLockable>::value,"Lock class is not standard layout");
        BasicLockable lock;
    public:
        typedef typename AddressAllocator::size_type        size_type;
        typedef typename AddressAllocator::difference_type  difference_type;
        typedef typename AddressAllocator::ubyte_pointer    ubyte_pointer;

        static constexpr size_type  invalid_address = AddressAllocator::invalid_address;


        using AddressAllocator::AddressAllocator;
        virtual ~AddressAllocatorBasicConcurrencyAdaptor() {}

        inline size_type    alloc_addr( size_type bytes, size_type alignment, size_type hint=0ull) noexcept
        {
            lock.lock();
            auto tmp = AddressAllocator::alloc_addr(bytes,alignment,hint);
            lock.unlock();
            return tmp;
        }

        inline void         free_addr(size_type addr, size_type bytes) noexcept
        {
            lock.lock();
            AddressAllocator::free_addr(addr,bytes);
            lock.unlock();
        }

        inline void         reset() noexcept
        {
            lock.lock();
            AddressAllocator::reset();
            lock.unlock();
        }

        inline size_type    max_size() const noexcept {return AddressAllocator::max_size();}

        inline size_type    max_alignment() const noexcept {return AddressAllocator::max_alignment();}

        template<typename... Args>
        static inline size_type reserved_size(size_t alignOff, size_type bufSz, Args&&... args) noexcept
        {
            return AddressAllocator::reserved_size(alignOff,bufSz,std::forward<Args>(args)...);
        }
};

}
}

#endif // __IRR_ADDRESS_ALLOCATOR_CONCURRENCY_ADAPTORS_H_INCLUDED__
