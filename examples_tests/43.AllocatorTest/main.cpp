#define _IRR_STATIC_LIB_

#include <irrlicht.h>
#include "irr/core/alloc/GeneralpurposeAddressAllocator.h"

using namespace irr;


/*
	Problems with GeneralPurposeallocator:
		1. GeneralPurposeallocator is not able to allocate as much address space as it is expected to
		2. Program doesn't compile, when I try to call safe_shrink_size
		3. "Assertion failed: false, file *\include\irr\core\alloc\GeneralpurposeAddressAllocator.h, line 368"
*/

int main()
{
	//1
	{
		void* resSpc = _IRR_ALIGNED_MALLOC(core::GeneralpurposeAddressAllocator<uint32_t>::reserved_size(alignof(uint32_t), 16u, 1u), _IRR_SIMD_ALIGNMENT);
		assert(resSpc != nullptr);
		auto alctr = core::GeneralpurposeAddressAllocator<uint32_t>(resSpc, 0u, 0u, alignof(uint32_t), 16u, 1u);

		for (int i = 0; i < 16u; i++)
		{
			auto addr = alctr.alloc_addr(1u, 1u);
			if (addr == alctr.invalid_address)
				os::Printer::print("invalid adress");
			else
				os::Printer::print(std::to_string(addr));
		}
		_IRR_ALIGNED_FREE(resSpc);
	}
	
	//2
	{
		void* resSpc = _IRR_ALIGNED_MALLOC(core::GeneralpurposeAddressAllocator<uint32_t>::reserved_size(alignof(uint32_t), 16u, 1u), _IRR_SIMD_ALIGNMENT);
		auto alctr = core::GeneralpurposeAddressAllocator<uint32_t>(resSpc, 0u, 0u, alignof(uint32_t), 16u, 1u);
		alctr.safe_shrink_size(8u);
		_IRR_ALIGNED_FREE(resSpc);
	}

	//3
	{
		void* resSpc = _IRR_ALIGNED_MALLOC(core::GeneralpurposeAddressAllocator<uint32_t>::reserved_size(alignof(uint32_t), 1413, 32), _IRR_SIMD_ALIGNMENT);
		auto alctr = core::GeneralpurposeAddressAllocator<uint32_t>(resSpc, 0u, 0u, alignof(uint32_t), 1413, 32);

		auto a = alctr.alloc_addr(1401, 1);
		_IRR_ALIGNED_FREE(resSpc);
	}

	return 0;
}
