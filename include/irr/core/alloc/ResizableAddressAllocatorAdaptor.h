#ifndef __IRR_RESIZABLE_ADDRESS_ALLOCATOR_ADAPTOR_H___
#define __IRR_RESIZABLE_ADDRESS_ALLOCATOR_ADAPTOR_H___

#include "irr/core/alloc/AddressAllocatorBase.h"

namespace irr
{
namespace core
{

namespace impl
{
    template<class AddressAllocator, class BufferAllocator, class CPUAllocator>
    class ResizableAddressAllocatorAdaptorBase
    {
            typedef typename AddressAllocator::size_type    size_type;
        protected:
            template<typename... Args>
            ResizableAddressAllocatorAdaptorBase(const CPUAllocator& reservedMemAllocator, const BufferAllocator& dataMemAllocator,
                                                 size_type maxAllocatableAlignment, size_type bufSz, const Args&... args) :
                                                        mReservedSize(AddressAllocator::reserved_size(bufSz,maxAllocatableAlignment,args...)),
                                                        mReservedAlloc(reservedMemAllocator), mDataAlloc(dataMemAllocator) {}

            size_type           mReservedSize;
            CPUAllocator        mReservedAlloc;
            BufferAllocator     mDataAlloc;
    };
}

/** The BufferAllocator concept

struct BufferDataAllocatorExample
{
    video::IDriver* mDriver;

    inline void*    allocate(size_t bytes) noexcept
    {
        // some implementation
        return pointerWhichCanBeWrittenTo;
    }

    inline void*    reallocate(void* addr, size_t bytes, const AddressAllocator& allocToQueryOffsets) noexcept
    {
        // some implementation
        return possiblyDifferentPointerToAddrArg;
    }

    inline void     deallocate(void* addr) noexcept
    {
        // some implementation
    }

    inline size_t   min_alignment() const
    {
        return mDriver->getMinimumMemoryMapAlignment();
    }
};
*/
template<class AddressAllocator, class BufferAllocator, class CPUAllocator=core::allocator<uint8_t> >
class ResizableAddressAllocatorAdaptor : protected ResizableAddressAllocatorAdaptorBase<AddressAllocator,BufferAllocator,CPUAllocator>, private AddressAllocator
{
        typedef ResizableAddressAllocatorAdaptorBase<AddressAllocator,BufferAllocator,CPUAllocator> ImplBase;
        typedef AddressAllocator                                                                    Base;
        inline Base&                                                                                getBaseRef() noexcept {return *this;}
    protected:
        typedef address_allocator_traits<AddressAllocator>                                          alloc_traits;
    public:
        typedef ResizableAddressAllocatorAdaptor<AddressAllocator,CPUAllocator>                     ThisType;

        _IRR_DECLARE_ADDRESS_ALLOCATOR_TYPEDEFS(_size_type);


        template<typename... Args>
        ResizableAddressAllocatorAdaptor(const CPUAllocator& reservedMemAllocator, const BufferAllocator& dataMemAllocator,
                                         size_type maxAllocatableAlignment, size_type bufSz, Args&&... args) :
                                                ImplBase(reservedMemAllocator,dataMemAllocator,maxAllocatableAlignment,bufSz,args...),
                                                Base(ImplBase::mReservedAlloc.allocate(ImplBase::mReservedSize,_IRR_SIMD_ALIGNMENT),
                                                     ImplBase::mDataAlloc.allocate(bufSz),bufSz,maxAllocatableAlignment,std::forward<Args>(args)...),
                                                growPolicy(defaultGrowPolicy), shrinkPolicy(defaultShrinkPolicy)
        {
        }

        virtual ~ResizableAddressAllocatorAdaptor()
        {
            if (Base::reservedSpace)
                ImplBase::mReservedAlloc.deallocate(reinterpret_cast<uint8_t*>(Base::reservedSpace),ImplBase::mReservedSize);

            // deallocate should handle nullptr without issue
            ImplBase::mDataAlloc.deallocate(reinterpret_cast<uint8_t*>(Base::bufferStart));
        }


        //!
        /** Warning outAddresses needs to be primed with `invalid_address` values,
        otherwise no allocation happens for elements not equal to `invalid_address`. */
        template<typename... Args>
        inline void                             multi_alloc_addr(uint32_t count, size_type* outAddresses, const size_type* bytes, const size_type* alignment, const Args&... args)
        {
            alloc_traits::multi_alloc_addr(getBaseRef(),count,outAddresses,bytes,alignment,args...);

            size_type totalNeededExtraNewMem = 0u;
            for (uint32_t i=0u; i<count; i++)
            {
                if (outAddresses[i]!=invalid_address)
                    continue;

                totalNeededExtraNewMem += std::max(bytes[i],getBaseRef().min_size())+alignment[i]-1u;
            }

            if (totalNeededExtraNewMem==size_type(0u))
                return;

            size_type allAllocatorSpace = alloc_traits::get_total_size(getBaseRef())-Base::alignOffset;
            size_type newSize = growPolicy(this,totalNeededExtraNewMem);
            newSize = std::max(newSize,alloc_traits::get_allocated_size(getBaseRef())+totalNeededExtraNewMem);

            if (newSize<=allAllocatorSpace)
                return;

            newSize = Base::safe_shrink_size(newSize,ImplBase::mDataAlloc::min_alignment()); // for padding

            //resize
            size_type oldReservedSize = ImplBase::mReservedSize;
            void* oldReserved = Base::reservedSpace;

            ImplBase::mReservedSize = AddressAllocator::reserved_size(getBaseRef(),newSize);
            void* newReserved = ImplBase::mReservedAlloc.allocate(ImplBase::mReservedSize,_IRR_SIMD_ALIGNMENT);


            getBaseRef() = AddressAllocator(getBaseRef(),newReserved,
                                            ImplBase::mDataAlloc.reallocate(newSize),newSize);


            if (oldReserved)
                ImplBase::mReservedAlloc.deallocate(reinterpret_cast<uint8_t*>(oldReserved),oldReservedSize);

            alloc_traits::multi_alloc_addr(getBaseRef(),count,outAddresses,bytes,alignment,std::forward<Args>(args)...);
        }

        template<typename... Args>
        inline void                             multi_free_addr(Args&&... args)
        {
            alloc_traits::multi_free_addr(getBaseRef(),std::forward<Args>(args)...);

            size_type allAllocatorSpace = alloc_traits::get_total_size(getBaseRef())-Base::alignOffset;
            size_type newSize = shrinkPolicy(this);
            if (newSize>=allAllocatorSpace)
                return;

            size_type guaranteedAlign = ImplBase::mDataAlloc.min_alignment();
            // some allocators may not be shrinkable because of fragmentation
            newSize = Base::safe_shrink_size(newSize,guaranteedAlign);
            if (newSize>=allAllocatorSpace)
                return;

            // resize
            size_type oldReservedSize = ImplBase::mReservedSize;
            void* oldReserved = Base::reservedSpace;
            void* oldDataBuff = Base::bufferStart;

            ImplBase::mReservedSize = AddressAllocator::reserved_size(getBaseRef(),newSize);
            void* newReserved = ImplBase::mReservedAlloc.allocate(ImplBase::mReservedSize,_IRR_SIMD_ALIGNMENT);
            void* newDataBuff = ImplBase::mDataAlloc.allocate(newSize);

            getBaseRef() = AddressAllocator(getBaseRef(),newReserved,newDataBuff,newSize);

            if (oldReserved)
                ImplBase::mReservedAlloc.deallocate(reinterpret_cast<uint8_t*>(oldReserved),oldReservedSize);
            // see comment in destructor
            ImplBase::mDataAlloc.deallocate(oldDataBuff);
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

#endif // __IRR_RESIZABLE_ADDRESS_ALLOCATOR_ADAPTOR_H___



