#ifndef __IRR_VIDEO_C_PROPERTY_POOL_H_INCLUDED__
#define __IRR_VIDEO_C_PROPERTY_POOL_H_INCLUDED__

#include "irr/video/IPropertyPool.h"


namespace irr
{
namespace video
{

    
template<template<class...> class allocator=core::allocator, typename... Properties>
class CPropertyPool final : public IPropertyPool
{
        using this_t = CPropertyPool<allocator,Properties...>;

        static auto propertyCombinedSize()
        {
            return (sizeof(Properties) + ...);
        }
        static size_t calcApproximateCapacity(size_t bufferSize)
        {
            return bufferSize/propertyCombinedSize();
        }

        _IRR_STATIC_INLINE_CONSTEXPR auto PropertyCount = sizeof...(Properties);

	public:
		static inline core::smart_refctd_ptr<this_t> create(asset::SBufferRange<IGPUBuffer>&& _memoryBlock, allocator<uint8_t>&& alloc = allocator<uint8_t>())
		{
			const auto reservedSize = getReservedSize(capacity);
			auto reserved = std::allocator_traits<allocator<uint8_t>>::allocate(alloc,reservedSize);
			if (!reserved)
				return nullptr;

			auto retval = create(std::move(_memoryBlock),reserved,std::move(alloc));
			if (!retval)
				std::allocator_traits<allocator<uint8_t>>::deallocate(alloc,reserved,reservedSize);

			return retval;
		}
		// if this method fails to create the pool, the callee must free the reserved memory themselves, also the reserved pointer must be compatible with the allocator so it can free it
        static inline core::smart_refctd_ptr<this_t> create(asset::SBufferRange<IGPUBuffer>&& _memoryBlock, void* reserved, allocator<uint8_t>&& alloc=allocator<uint8_t>())
        {
			assert(_memoryBlock.isValid());
			assert(reserved);

            const auto approximateCapacity = calcApproximateCapacity(_memoryBlock.size);
            auto capacity = approximateCapacity;
            while (capacity)
            {
                size_t wouldBeSize = PropertySizes[0]*capacity;
                // now compute with padding and alignments
                for (auto i=1; i<PropertyCount; i++)
                {
                    // align
                    wouldBeSize = core::roundUp(wouldBeSize,PropertySizes[i]);
                    // increase
                    wouldBeSize += PropertySizes[i]*capacity;
                }
                // if still manage to fit, then ok
                if (wouldBeSize<=_memoryBlock.size)
                    break;
                capacity--;
            }
            capacity = core::min<decltype(capacity)>(~uint32_t(0), capacity);
            if (!capacity)
                return nullptr;

            return core::make_smart_refctd_ptr<CPropertyPool>(std::move(_memoryBlock),capacity,reserved,std::move(alloc));
        }

		//
		virtual uint32_t getPropertyCount() const {return PropertyCount;}
		virtual uint32_t getPropertySize(uint32_t ix) const {return PropertySizes[ix];}

	protected:
        CPropertyPool(core::SBufferRange<IGPUBuffer>&& _memoryBlock, uint32_t capacity, void* reserved, allocator<uint8_t>&& _alloc)
            : IPropertyPool(std::move(_memoryBlock),capacity,reserved), alloc(std::move(_alloc))
        {
        }

        ~CPropertyPool()
        {
            void* reserved = const_cast<void*>(indexAllocator.getReservedSpacePtr());
            std::allocator_traits<allocator<uint8_t>>::deallocate(alloc,reserved,getReservedSize(getCapacity()));
        }

        
        //
        allocator<uint8_t> alloc;

        _IRR_STATIC_INLINE_CONSTEXPR std::array<uint32_t,PropertyCount> PropertySizes = {sizeof(Properties)...};
};


}
}

#endif