// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_VIDEO_C_SINGLE_BUFFER_SUB_ALLOCATOR_H_
#define _NBL_VIDEO_C_SINGLE_BUFFER_SUB_ALLOCATOR_H_

#include "nbl/video/alloc/IBufferAllocator.h"

#include <type_traits>

namespace nbl::video
{

// address allocator gives offsets
// reserved allocator allocates memory to keep the address allocator state inside
template<class AddrAllocator, class ReservAllocator=core::allocator<uint8_t>>
class CSingleBufferSubAllocator : public IBufferAllocator
{
  public:
        using AddressAllocator = AddrAllocator;
        using ReservedAllocator = ReservAllocator;
        using size_type = AddressAllocator::size_type;
        using value_type = AddressAllocator::size_type;
        static constexpr value_type invalid_value = AddressAllocator::invalid_address;

        // constructors
        template<typename... Args>
        inline CSingleBufferSubAllocator(asset::SBufferRange<IGPUBuffer>&& _bufferRange, ReservedAllocator&& _reservedAllocator, const value_type maxAllocatableAlignment, Args&&... args) :
            m_addressAllocator(
                _reservedAllocator.allocate(AddressAllocator::reserved_size(maxAllocatableAlignment,_bufferRange.size,args...),_NBL_SIMD_ALIGNMENT),
                _bufferRange.offset, 0u, maxAllocatableAlignment, _bufferRange.size, std::forward<Args>(args)...
            ), m_reservedAllocator(std::move(_reservedAllocator)), m_buffer(std::move(_bufferRange.buffer))
        {
            assert(_bufferRange.isValid());
        }
        // version with default constructed reserved allocator
        template<typename... Args>
        explicit inline CSingleBufferSubAllocator(asset::SBufferRange<IGPUBuffer>&& _bufferRange, const value_type maxAllocatableAlignment, Args&&... args) :
            CSingleBufferSubAllocator(std::move(_bufferRange),ReservedAllocator(),maxAllocatableAlignment,std::forward<Args>(args)...)
        {
        }
        ~CSingleBufferSubAllocator()
        {
            auto ptr = reinterpret_cast<const uint8_t*>(core::address_allocator_traits<AddressAllocator>::getReservedSpacePtr(m_addressAllocator));
            m_reservedAllocator.deallocate(const_cast<uint8_t*>(ptr),m_reservedSize);
        }

        // anyone gonna use it?
        inline const AddressAllocator& getAddressAllocator() const {return m_addressAllocator;}

        //
        inline const ReservedAllocator& getReservedAllocator() const {return m_reservedAllocator;}

        // buffer getters
        inline IGPUBuffer* getBuffer() {return m_buffer.get();}
        inline const IGPUBuffer* getBuffer() const {return m_buffer.get();}

        // main methods

        //! Warning `outAddresses` needs to be primed with `invalid_value` values, otherwise no allocation happens for elements not equal to `invalid_value`
        template<typename... Args>
        inline void multi_allocate(uint32_t count, value_type* outAddresses, const size_type* byteSizes, const size_type* alignments, const Args&... args)
        {
            core::address_allocator_traits<AddressAllocator>::multi_alloc_addr(m_addressAllocator,count,outAddresses,byteSizes,alignments,args...);
        }
        template<typename... Args>
        inline void multi_deallocate(Args&&... args)
        {
            core::address_allocator_traits<AddressAllocator>::multi_free_addr(m_addressAllocator,std::forward<Args>(args)...);
        }

        // to conform to IBufferAllocator concept
        template<typename... Args>
        inline value_type allocate(const size_type bytes, const size_type alignment, const Args&... args)
        {
            value_type retval = invalid_value;
            multi_allocate(&retval,&bytes,&alignment,args...);
            return retval;
        }
        template<typename... Args>
        inline void deallocate(value_type& allocation, Args&&... args)
        {
            multi_deallocate(std::forward<Args>(args)...);
            allocation = invalid_value;
        }

    protected:
        AddressAllocator                    m_addressAllocator;
        ReservedAllocator                   m_reservedAllocator;
        size_t                              m_reservedSize;
        core::smart_refctd_ptr<IGPUBuffer>  m_buffer;
};

}

#endif

