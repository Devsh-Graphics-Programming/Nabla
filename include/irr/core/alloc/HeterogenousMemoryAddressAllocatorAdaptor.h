#ifndef __IRR_HETEROGENOUS_MEMORY_ADDRESS_ALLOCATOR_ADAPTOR_H___
#define __IRR_HETEROGENOUS_MEMORY_ADDRESS_ALLOCATOR_ADAPTOR_H___

#include "irr/static_if.h"
#include "irr/core/Types.h"
#include "irr/core/alloc/address_allocator_traits.h"
#include "irr/core/alloc/AddressAllocatorBase.h"

namespace irr
{
namespace core
{

/** The BufferAllocator concept

class BufferDataAllocatorExample
{
    public:
        typedef ______ value_type;

        inline value_type   allocate(size_t bytes, size_t alignment) noexcept
        {
            // some implementation
            return pointerWhichCanBeWrittenTo;
        }

        inline void                 reallocate(value_type& allocation, size_t bytes,  size_t alignment, const AddressAllocator& allocToQueryOffsets, ...) noexcept
        {
            // some implementation
            return possiblyDifferentPointerToAddrArg;
        }

        inline void                 deallocate(value_type& allocation) noexcept
        {
            // some implementation
        }

        IVideoDriver*   getDriver() noexcept;
};
*/
namespace impl
{
    class FriendOfHeterogenousMemoryAddressAllocatorAdaptor;


    template<class AddrAllocator, class OtherAllocator, class HostAllocator>
    class HeterogenousMemoryAddressAllocatorAdaptorBase
    {
        public:
            typedef AddrAllocator AddressAllocator;
            typedef OtherAllocator  OtherAllocatorType;
            typedef HostAllocator   HostAllocatorType;
        private:
            friend class FriendOfHeterogenousMemoryAddressAllocatorAdaptor;
            typedef  typename AddressAllocator::size_type    size_type;

        public:
            inline size_type        getDataBufferSize() const {return mDataSize;}
        protected:
            template<typename... Args>
            HeterogenousMemoryAddressAllocatorAdaptorBase(const HostAllocator& reservedMemAllocator, const OtherAllocator& dataMemAllocator, size_type maxAllocatableAlignment, size_type bufSz, const Args&... args) :
                                                                        mDataSize(bufSz), mDataAlloc(dataMemAllocator),
                                                                        mReservedSize(AddressAllocator::reserved_size(maxAllocatableAlignment,bufSz,args...)),
                                                                        mReservedAlloc(reservedMemAllocator) {}

            size_type               mDataSize;
            OtherAllocator          mDataAlloc;
            size_type               mReservedSize;
            HostAllocator           mReservedAlloc;
    };


    class FriendOfHeterogenousMemoryAddressAllocatorAdaptor
    {
        protected:
            FriendOfHeterogenousMemoryAddressAllocatorAdaptor() = default;
            virtual ~FriendOfHeterogenousMemoryAddressAllocatorAdaptor() {}

            template<class AddressAllocator, class OtherAllocator, class HostAllocator>
            inline OtherAllocator&  getDataAllocator(HeterogenousMemoryAddressAllocatorAdaptorBase<AddressAllocator, OtherAllocator, HostAllocator>& object)
            {
                return object.mDataAlloc;
            }

            template<class AddressAllocator, class OtherAllocator, class HostAllocator>
            inline HostAllocator&   getHostAllocator(HeterogenousMemoryAddressAllocatorAdaptorBase<AddressAllocator, OtherAllocator, HostAllocator>& object)
            {
                return object.mReservedAlloc;
            }
    };
}



template<class AddressAllocator, class BufferAllocator, class HostAllocator=core::allocator<uint8_t> >
class HeterogenousMemoryAddressAllocatorAdaptor : public impl::HeterogenousMemoryAddressAllocatorAdaptorBase<AddressAllocator,BufferAllocator,HostAllocator>, /* This is supposed to be private inheritance */protected AddressAllocator
{
        typedef impl::HeterogenousMemoryAddressAllocatorAdaptorBase<AddressAllocator,BufferAllocator,HostAllocator> ImplBase;
    protected:
        inline AddressAllocator&                                                                                    getBaseAddrAllocRef() noexcept {return *this;}

        typedef typename BufferAllocator::value_type allocation_type;
        allocation_type mAllocation;
    public:
        typedef address_allocator_traits<AddressAllocator>  alloc_traits;
        typedef typename AddressAllocator::size_type        size_type;
        static constexpr size_type invalid_address              = AddressAllocator::invalid_address;

        template<typename... Args>
        HeterogenousMemoryAddressAllocatorAdaptor(const HostAllocator& reservedMemAllocator, const BufferAllocator& dataMemAllocator,
                                                                                    size_type addressOffsetToApply, size_type alignOffsetNeeded, size_type maxAllocatableAlignment, size_type bufSz, Args&&... args) :
                                            ImplBase(reservedMemAllocator,dataMemAllocator,maxAllocatableAlignment,bufSz,args...),
                                            AddressAllocator(ImplBase::mReservedAlloc.allocate(ImplBase::mReservedSize,_IRR_SIMD_ALIGNMENT),
                                                                        addressOffsetToApply,alignOffsetNeeded,maxAllocatableAlignment,bufSz,std::forward<Args>(args)...)
        {
            mAllocation = ImplBase::mDataAlloc.allocate(bufSz,maxAllocatableAlignment);
			IRR_PSEUDO_IF_CONSTEXPR_BEGIN(!alloc_traits::supportsNullBuffer)
			{
                this->getBaseAddrAllocRef().setDataBufferPtr(std::get<1u>(mAllocation));
            }
			IRR_PSEUDO_IF_CONSTEXPR_END
        }

        virtual ~HeterogenousMemoryAddressAllocatorAdaptor()
        {
            if (AddressAllocator::reservedSpace)
                ImplBase::mReservedAlloc.deallocate(reinterpret_cast<uint8_t*>(AddressAllocator::reservedSpace),ImplBase::mReservedSize);

            // deallocate should handle nullptr without issue
            ImplBase::mDataAlloc.deallocate(mAllocation);
        }

        inline allocation_type getCurrentBufferAllocation() noexcept {return mAllocation;}
        inline allocation_type getCurrentBufferAllocation() const noexcept {return mAllocation;}


        //!
        /** Warning outAddresses needs to be primed with `invalid_address` values,
        otherwise no allocation happens for elements not equal to `invalid_address`. */
        template<typename... Args>
        inline void                             multi_alloc_addr(uint32_t count, size_type* outAddresses, const size_type* bytes, const size_type* alignment, const Args&... args)
        {
            address_allocator_traits<AddressAllocator>::multi_alloc_addr(getBaseAddrAllocRef(),count,outAddresses,bytes,alignment,args...);
        }

        template<typename... Args>
        inline void                             multi_free_addr(Args&&... args)
        {
            address_allocator_traits<AddressAllocator>::multi_free_addr(getBaseAddrAllocRef(),std::forward<Args>(args)...);
        }


        inline const AddressAllocator&  getAddressAllocator() const
        {
            return *this;
        }
};

}
}

#endif // __IRR_HETEROGENOUS_MEMORY_ADDRESS_ALLOCATOR_ADAPTOR_H___




