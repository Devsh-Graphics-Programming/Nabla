#define _IRR_STATIC_LIB_

#include <irrlicht.h>

#include "irr/core/alloc/LinearAddressAllocator.h"
#include "irr/core/alloc/StackAddressAllocator.h"

using namespace irr;


int main()
{
	printf("Linear \n");
	irr::core::address_allocator_traits<core::LinearAddressAllocatorST<uint32_t> >::printDebugInfo();
	printf("Stack \n");
	irr::core::address_allocator_traits<core::StackAddressAllocatorST<uint32_t> >::printDebugInfo();
	printf("Pool \n");
	irr::core::address_allocator_traits<core::PoolAddressAllocatorST<uint32_t> >::printDebugInfo();
	printf("Cont \n");
	irr::core::address_allocator_traits<core::ContiguousPoolAddressAllocator<uint32_t> >::printDebugInfo();
	printf("General \n");
	irr::core::address_allocator_traits<core::GeneralpurposeAddressAllocator<uint32_t> >::printDebugInfo();

	return 0;
}
