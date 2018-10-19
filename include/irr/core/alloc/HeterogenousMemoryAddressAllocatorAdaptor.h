#ifndef __IRR_HETEROGENOUS_MEMORY_ADDRESS_ALLOCATOR_ADAPTOR_H___
#define __IRR_HETEROGENOUS_MEMORY_ADDRESS_ALLOCATOR_ADAPTOR_H___

#include "irr/core/alloc/address_allocator_traits.h"
#include "irr/core/alloc/AddressAllocatorBase.h"

namespace irr
{
namespace core
{

/** The BufferAllocator concept

struct BufferDataAllocatorExample
{
    video::IDriver* mDriver;

    inline void*    allocate(size_t bytes) noexcept
    {
        // some implementation
        return pointerWhichCanBeWrittenTo;
    }

    inline void*    reallocate(void* addr, size_t bytes, const AddressAllocator& allocToQueryOffsets, ...) noexcept
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
namespace impl
{
    class FriendOfHeterogenousMemoryAddressAllocatorAdaptor
    {
        protected:
            FriendOfHeterogenousMemoryAddressAllocatorAdaptor() = default;
            virtual ~FriendOfHeterogenousMemoryAddressAllocatorAdaptor() {}

            template<class HeterogenousMemoryAddressAllocatorAdaptorT>
            inline typename HeterogenousMemoryAddressAllocatorAdaptorT::OtherAllocatorType&     getDataAllocator(HeterogenousMemoryAddressAllocatorAdaptorT& object)
            {
                return object.mDataAlloc;
            }

            template<class HeterogenousMemoryAddressAllocatorAdaptorT>
            inline typename HeterogenousMemoryAddressAllocatorAdaptorT::HostAllocatorType&      getHostAllocator(HeterogenousMemoryAddressAllocatorAdaptorT& object)
            {
                return object.mReservedAlloc;
            }

            template<class HeterogenousMemoryAddressAllocatorAdaptorT>
            inline const typename HeterogenousMemoryAddressAllocatorAdaptorT::Base&             getAddressAllocator(HeterogenousMemoryAddressAllocatorAdaptorT& object) const
            {
                return object.getBaseRef();
            }
    };


    template<class AddressAllocator, class OtherAllocator, class HostAllocator>
    class HeterogenousMemoryAddressAllocatorAdaptorBase
    {
            friend class FriendOfResizableAddressAllocatorAdaptor;

            typedef typename AddressAllocator::size_type    size_type;
        protected:
            template<typename... Args>
            HeterogenousMemoryAddressAllocatorAdaptorBase(const HostAllocator& reservedMemAllocator, const OtherAllocator& dataMemAllocator, size_type bufSz, const Args&... args) :
                                                                        mDataSize(bufSz), mDataAlloc(dataMemAllocator),
                                                                        mReservedSize(AddressAllocator::reserved_size(bufSz,mDataAlloc.min_alignment(),args...)),
                                                                        mReservedAlloc(reservedMemAllocator) {}

            size_type                           mDataSize;
            OtherAllocator                      mDataAlloc;
            size_type                           mReservedSize;
            HostAllocator                       mReservedAlloc;
        public:
            typedef typename OtherAllocator     OtherAllocatorType;
            typedef typename HostAllocator      HostAllocatorType;

            inline size_type                    getDataBufferSize() const {return mDataSize;}
    };
}




template<class AddressAllocator, class BufferAllocator, class HostAllocator=core::allocator<uint8_t> >
class HeterogenousMemoryAddressAllocatorAdaptor : public impl::HeterogenousMemoryAddressAllocatorAdaptorBase<AddressAllocator,BufferAllocator,HostAllocator>, private AddressAllocator
{
        //friend class impl::FriendOfHeterogenousMemoryAddressAllocatorAdaptor;

        typedef impl::HeterogenousMemoryAddressAllocatorAdaptorBase<AddressAllocator,BufferAllocator,HostAllocator> ImplBase;
        inline AddressAllocator&                                                                                    getBaseAddrAllocRef() noexcept {return *this;}
    protected:
        typedef address_allocator_traits<AddressAllocator>                                                          alloc_traits;
    public:
        typedef HeterogenousMemoryAddressAllocatorAdaptor<AddressAllocator,BufferAllocator,HostAllocator>           ThisType;

        _IRR_DECLARE_ADDRESS_ALLOCATOR_TYPEDEFS(_size_type);


        template<typename... Args>
        HeterogenousMemoryAddressAllocatorAdaptor(const HostAllocator& reservedMemAllocator, const BufferAllocator& dataMemAllocator, size_type bufSz, Args&&... args) :
                                            ImplBase(reservedMemAllocator,dataMemAllocator,bufSz,args...),
                                            AddressAllocator(ImplBase::mReservedAlloc.allocate(ImplBase::mReservedSize,_IRR_SIMD_ALIGNMENT),
                                                            ImplBase::mDataAlloc.allocate(bufSz),bufSz,ImplBase::mDataAlloc.min_alignment(),std::forward<Args>(args)...)
        {
        }

        virtual ~HeterogenousMemoryAddressAllocatorAdaptor()
        {
            if (AddressAllocator::reservedSpace)
                ImplBase::mReservedAlloc.deallocate(reinterpret_cast<uint8_t*>(AddressAllocator::reservedSpace),ImplBase::mReservedSize);

            // deallocate should handle nullptr without issue
            ImplBase::mDataAlloc.deallocate(reinterpret_cast<uint8_t*>(AddressAllocator::bufferStart));
        }


        //!
        /** Warning outAddresses needs to be primed with `invalid_address` values,
        otherwise no allocation happens for elements not equal to `invalid_address`. */
        template<typename... Args>
        inline void                             multi_alloc_addr(uint32_t count, size_type* outAddresses, const size_type* bytes, const size_type* alignment, const Args&... args)
        {
            alloc_traits::multi_alloc_addr(getBaseAddrAllocRef(),count,outAddresses,bytes,alignment,args...);
        }

        template<typename... Args>
        inline void                             multi_free_addr(Args&&... args)
        {
            alloc_traits::multi_free_addr(getBaseAddrAllocRef(),std::forward<Args>(args)...);
        }
};

}
}

#endif // __IRR_HETEROGENOUS_MEMORY_ADDRESS_ALLOCATOR_ADAPTOR_H___




