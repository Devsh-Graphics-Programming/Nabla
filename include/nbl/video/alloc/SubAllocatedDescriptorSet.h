// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_VIDEO_SUB_ALLOCATED_DESCRIPTOR_SET_H_
#define _NBL_VIDEO_SUB_ALLOCATED_DESCRIPTOR_SET_H

#include "nbl/video/alloc/IBufferAllocator.h"

#include <type_traits>

namespace nbl::video
{

// address allocator gives offsets
// reserved allocator allocates memory to keep the address allocator state inside
template<class AddrAllocator, class ReservAllocator=core::allocator<uint8_t>>
class SubAllocatedDescriptorSet : public core::IReferenceCounted 
{
  public:
        using AddressAllocator = AddrAllocator;
        using ReservedAllocator = ReservAllocator;
        using size_type = typename AddressAllocator::size_type;
        using value_type = typename AddressAllocator::size_type;
        static constexpr value_type invalid_value = AddressAllocator::invalid_address;

        // constructors
        template<typename... Args>
        inline SubAllocatedDescriptorSet(const std::span<const video::IGPUDescriptorSetLayout::SBinding> bindings, 
            ReservedAllocator&& _reservedAllocator, const value_type maxAllocatableAlignment, Args&&... args)
        {
            auto allocatableDescriptors = 0;
            m_allocatableRanges.reserve(bindings.size());

            for (auto& binding : bindings)
            {
                SubAllocDescriptorSetRange range;
                range.offset = allocatableDescriptors;
                range.binding = binding;
                // Only bindings with these flags will be allocatable
                if (binding.createFlags.hasFlags(core::bitflag(IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_UPDATE_AFTER_BIND_BIT) 
					| IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_UPDATE_UNUSED_WHILE_PENDING_BIT 
					| IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_PARTIALLY_BOUND_BIT))
                {
                    allocatableDescriptors += binding.count;
                }
				m_allocatableRanges.push_back(range);
            }

			m_addressAllocator = AddrAllocator(
				_reservedAllocator.allocate(AddressAllocator::reserved_size(maxAllocatableAlignment, static_cast<size_type>(allocatableDescriptors), args...), _NBL_SIMD_ALIGNMENT),
				static_cast<size_type>(0), 0u, maxAllocatableAlignment, static_cast<size_type>(allocatableDescriptors), std::forward<Args>(args)...
			);
			m_reservedAllocator = ReservedAllocator(std::move(_reservedAllocator));
            m_reservedSize = allocatableDescriptors;
        }
        // version with default constructed reserved allocator
        template<typename... Args>
        explicit inline SubAllocatedDescriptorSet(const std::span<const video::IGPUDescriptorSetLayout::SBinding> bindings, 
            const value_type maxAllocatableAlignment, Args&&... args) :
            SubAllocatedDescriptorSet(bindings,ReservedAllocator(),maxAllocatableAlignment,std::forward<Args>(args)...)
        {
        }
        ~SubAllocatedDescriptorSet()
        {
            auto ptr = reinterpret_cast<const uint8_t*>(core::address_allocator_traits<AddressAllocator>::getReservedSpacePtr(m_addressAllocator));
            m_reservedAllocator.deallocate(const_cast<uint8_t*>(ptr),m_reservedSize);
        }

        // anyone gonna use it?
        inline const AddressAllocator& getAddressAllocator() const {return m_addressAllocator;}

        //
        inline ReservedAllocator& getReservedAllocator() {return m_reservedAllocator;}

        // main methods

        //! Warning `outAddresses` needs to be primed with `invalid_value` values, otherwise no allocation happens for elements not equal to `invalid_value`
        template<typename... Args>
        inline void multi_allocate(uint32_t count, value_type* outAddresses, const size_type* sizes, const Args&... args)
        {
            core::address_allocator_traits<AddressAllocator>::multi_alloc_addr(m_addressAllocator,count,outAddresses,sizes,1,args...);
        }
        inline void multi_deallocate(uint32_t count, const size_type* addr, const size_type* sizes)
        {
            core::address_allocator_traits<AddressAllocator>::multi_free_addr(m_addressAllocator,count,addr,sizes);
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
        size_t                              m_reservedSize; // FIXME: uninitialized variable

        struct SubAllocDescriptorSetRange {
            uint32_t offset;
            video::IGPUDescriptorSetLayout::SBinding binding;
        };
        std::vector<SubAllocDescriptorSetRange> m_allocatableRanges = {};
};

}

#endif

