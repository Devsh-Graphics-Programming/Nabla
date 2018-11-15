#define _IRR_STATIC_LIB_

#include <irrlicht.h>

#include "irr/core/alloc/address_allocator_traits.h"
#include "irr/core/alloc/LinearAddressAllocator.h"
#include "irr/core/alloc/StackAddressAllocator.h"

using namespace irr;


int main()
{
    printf("SINGLE THREADED======================================================\n");
	printf("Linear \n");
	irr::core::address_allocator_traits<core::LinearAddressAllocatorST<uint32_t> >::printDebugInfo();
	printf("Stack \n");
	irr::core::address_allocator_traits<core::StackAddressAllocatorST<uint32_t> >::printDebugInfo();
	printf("Pool \n");
	irr::core::address_allocator_traits<core::PoolAddressAllocatorST<uint32_t> >::printDebugInfo();
	printf("Cont \n");
	irr::core::address_allocator_traits<core::ContiguousPoolAddressAllocatorST<uint32_t> >::printDebugInfo();
	printf("General \n");
	irr::core::address_allocator_traits<core::GeneralpurposeAddressAllocatorST<uint32_t> >::printDebugInfo();

    printf("MULTI THREADED=======================================================\n");
	printf("Linear \n");
	irr::core::address_allocator_traits<core::LinearAddressAllocatorMT<uint32_t,std::recursive_mutex> >::printDebugInfo();
	printf("Pool \n");
	irr::core::address_allocator_traits<core::PoolAddressAllocatorMT<uint32_t,std::recursive_mutex> >::printDebugInfo();
	printf("Cont \n");
	irr::core::address_allocator_traits<core::ContiguousPoolAddressAllocatorMT<uint32_t,std::recursive_mutex> >::printDebugInfo();
	printf("General \n");
	irr::core::address_allocator_traits<core::GeneralpurposeAddressAllocatorMT<uint32_t,std::recursive_mutex> >::printDebugInfo();

	return 0;
}
