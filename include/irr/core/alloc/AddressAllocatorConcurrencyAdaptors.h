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

template<class AddressAllocator, class BasicLockable>
class AddressAllocatorBasicConcurrencyAdaptor : public address_allocator_traits<AddressAllocator>
{
        static_assert(std::is_standard_layout<BasicLockable>::value,"Lock class is not standard layout");
        BasicLockable lock;

        AddressAllocator& getBaseRef() {return reinterpret_cast<AddressAllocator&>(*this);}
    public:
        _IRR_DECLARE_ADDRESS_ALLOCATOR_TYPEDEFS(typename AddressAllocator::size_type);

        typedef address_allocator_traits<AddressAllocator> traits;


        using AddressAllocator::AddressAllocator;
        virtual ~AddressAllocatorBasicConcurrencyAdaptor() {}

        inline size_type    get_real_addr(size_type allocated_addr) const noexcept
        {
            ///lock.lock();
            auto retval = traits::get_real_addr(getBaseRef(),allocated_addr);
            ///lock.unlock();
            return retval;
        }
        inline void         multi_alloc_addr(size_type* outAddresses, uint32_t count, const size_type* bytes,
                                             const size_type* alignment, const size_type* hint=nullptr) noexcept
        {
            lock.lock();
            traits::multi_alloc_addr(getBaseRef(),outAddresses,count,bytes,alignment,hint);
            lock.unlock();
        }
        inline void         multi_free_addr(uint32_t count, const size_type* addr, const size_type* bytes) noexcept
        {
            lock.lock();
            traits::multi_free_addr(getBaseRef(),count,addr,bytes);
            lock.unlock();
        }

        inline size_type    alloc_addr( size_type bytes, size_type alignment, size_type hint=0ull) noexcept
        {
            lock.lock();
            auto retval = AddressAllocator::alloc_addr(bytes,alignment,hint);
            lock.unlock();
            return retval;
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

        inline size_type    max_alignment() const noexcept
        {
            lock.lock();
            auto retval = AddressAllocator::max_alignment();
            lock.unlock();
            return retval;
        }

        inline size_type    safe_shrink_size(size_type bound=0u) const noexcept
        {
            lock.lock();
            auto retval = AddressAllocator::safe_shrink_size();
            lock.unlock();
            return retval;
        }

        template<typename... Args>
        static inline size_type reserved_size(size_type bufSz, const Args&... args) noexcept
        {
            return AddressAllocator::reserved_size(bufSz,args...);
        }
        static inline size_type reserved_size(size_type bufSz, const AddressAllocator& other) noexcept
        {
            return AddressAllocator::reserved_size(bufSz,other);
        }
};

}
}

#endif // __IRR_ADDRESS_ALLOCATOR_CONCURRENCY_ADAPTORS_H_INCLUDED__
