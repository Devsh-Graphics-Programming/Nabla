#ifndef __IRR_RESIZABLE_HETEROGENOUS_MEMORY_ALLOCATOR_H___
#define __IRR_RESIZABLE_HETEROGENOUS_MEMORY_ALLOCATOR_H___


#include "irr/core/alloc/HeterogenousMemoryAddressAllocatorAdaptor.h"


namespace irr
{
namespace core
{


template<class HeterogenousMemoryAllocator>
class ResizableHeterogenousMemoryAllocator : public HeterogenousMemoryAllocator
{
        typedef HeterogenousMemoryAllocator                                         Base;
        typedef typename Base::alloc_traits::allocator_type                         AddressAllocator;
        inline AddressAllocator&                                                    getBaseAddrAllocRef() noexcept {return *this;}
    public:
        typedef ResizableHeterogenousMemoryAllocator<HeterogenousMemoryAllocator>   ThisType;

        _IRR_DECLARE_ADDRESS_ALLOCATOR_TYPEDEFS(_size_type);


        template<typename... Args>
        ResizableHeterogenousMemoryAllocator(Args&&... args) : HeterogenousMemoryAllocator(std::forward<Args>(args)...),growPolicy(defaultGrowPolicy),shrinkPolicy(defaultShrinkPolicy)
        {
        }

        virtual ~ResizableHeterogenousMemoryAllocator() {}


        //!
        /** Warning outAddresses needs to be primed with `invalid_address` values,
        otherwise no allocation happens for elements not equal to `invalid_address`. */
        template<typename... Args>
        inline void                             multi_alloc_addr(uint32_t count, size_type* outAddresses, const size_type* bytes, const size_type* alignment, const Args&... args)
        {
            Base::multi_alloc_addr(count,outAddresses,bytes,alignment,args...);

            size_type totalNeededExtraNewMem = 0u;
            for (uint32_t i=0u; i<count; i++)
            {
                if (outAddresses[i]!=invalid_address)
                    continue;

                totalNeededExtraNewMem += std::max(bytes[i],getBaseAddrAllocRef().min_size())+alignment[i]-1u;
            }

            if (totalNeededExtraNewMem==size_type(0u))
                return;

            size_type allAllocatorSpace = alloc_traits::get_total_size(getBaseAddrAllocRef())-AddressAllocator::alignOffset;
            size_type newSize = growPolicy(this,totalNeededExtraNewMem);
            newSize = std::max(newSize,alloc_traits::get_allocated_size(getBaseAddrAllocRef())+totalNeededExtraNewMem);

            if (newSize<=allAllocatorSpace)
                return;

            size_type guaranteedAlign = Base::mDataAlloc.min_alignment();
            newSize = AddressAllocator::safe_shrink_size(newSize,guaranteedAlign); // for padding

            //resize
            size_type oldReservedSize = Base::mReservedSize;
            void* oldReserved = AddressAllocator::reservedSpace;

            Base::mReservedSize = AddressAllocator::reserved_size(getBaseAddrAllocRef(),newSize);
            void* newReserved = Base::mReservedAlloc.allocate(Base::mReservedSize,_IRR_SIMD_ALIGNMENT);


            Base::mDataSize = newSize;
            getBaseAddrAllocRef() = AddressAllocator(getBaseAddrAllocRef(),newReserved,Base::mDataAlloc.reallocate(Base::mDataSize),Base::mDataSize);


            if (oldReserved)
                Base::mReservedAlloc.deallocate(reinterpret_cast<uint8_t*>(oldReserved),oldReservedSize);

            Base::multi_alloc_addr(count,outAddresses,bytes,alignment,std::forward<Args>(args)...);
        }

        template<typename... Args>
        inline void                             multi_free_addr(Args&&... args)
        {
            Base::multi_free_addr(std::forward<Args>(args)...);

            size_type allAllocatorSpace = alloc_traits::get_total_size(getBaseAddrAllocRef())-AddressAllocator::alignOffset;
            size_type newSize = shrinkPolicy(this);
            if (newSize>=allAllocatorSpace)
                return;

            size_type guaranteedAlign = Base::mDataAlloc.min_alignment();
            // some allocators may not be shrinkable because of fragmentation
            newSize = AddressAllocator::safe_shrink_size(newSize,guaranteedAlign);
            if (newSize>=allAllocatorSpace)
                return;

            // resize
            size_type oldReservedSize = Base::mReservedSize;
            void* oldReserved = AddressAllocator::reservedSpace;

            Base::mReservedSize = AddressAllocator::reserved_size(getBaseAddrAllocRef(),newSize);
            void* newReserved = Base::mReservedAlloc.allocate(Base::mReservedSize,_IRR_SIMD_ALIGNMENT);


            Base::mDataSize = newSize;
            getBaseAddrAllocRef() = AddressAllocator(getBaseAddrAllocRef(),newReserved,Base::mDataAlloc.reallocate(Base::mDataSize),Base::mDataSize);


            if (oldReserved)
                Base::mReservedAlloc.deallocate(reinterpret_cast<uint8_t*>(oldReserved),oldReservedSize);
        }


    protected:
        //

        constexpr static size_type defaultGrowStep = 32u*4096u; //128k at a time
        constexpr static size_type defaultGrowStepMinus1 = defaultGrowStep-1u;
        static inline size_type                 defaultGrowPolicy(ThisType* _this, size_type totalRequestedNewMem)
        {
            size_type allAllocatorSpace = alloc_traits::get_total_size(*_this)-alloc_traits::get_align_offset(*_this);
            size_type nextAllocTotal = alloc_traits::get_allocated_size(*_this)+totalRequestedNewMem;
            if (nextAllocTotal>allAllocatorSpace)
                return (nextAllocTotal+defaultGrowStepMinus1)&(~defaultGrowStepMinus1);

            return allAllocatorSpace;
        }
        static inline size_type                 defaultShrinkPolicy(ThisType* _this)
        {
            constexpr size_type shrinkStep = 256u*4096u; //1M at a time

            size_type allFreeSpace = alloc_traits::get_free_size(*_this);
            if (allFreeSpace>shrinkStep)
                return (alloc_traits::get_allocated_size(*_this)+defaultGrowStepMinus1)&(~defaultGrowStepMinus1);

            return alloc_traits::get_total_size(*_this)-alloc_traits::get_align_offset(*_this);
        }

        size_type(*growPolicy)(ThisType*,size_type);
        size_type(*shrinkPolicy)(ThisType*);

    public:
        //! Grow Policies return
        inline const decltype(growPolicy)&      getGrowPolicy() const {return growPolicy;}
        inline void                             setGrowPolicy(const decltype(growPolicy)& newGrowPolicy) {growPolicy=newGrowPolicy;}

        inline const decltype(shrinkPolicy)&    getShrinkPolicy() const {return shrinkPolicy;}
        inline void                             setShrinkPolicy(const decltype(shrinkPolicy)& newShrinkPolicy) {shrinkPolicy=newShrinkPolicy;}
};

}
}

#endif // __IRR_RESIZABLE_HETEROGENOUS_MEMORY_ALLOCATOR_H___




