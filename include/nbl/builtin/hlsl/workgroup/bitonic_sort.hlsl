#ifndef NBL_BUILTIN_HLSL_WORKGROUP_BITONIC_SORT_INCLUDED
#define NBL_BUILTIN_HLSL_WORKGROUP_BITONIC_SORT_INCLUDED
#include "nbl/builtin/hlsl/bitonic_sort/common.hlsl"
#include "nbl/builtin/hlsl/memory_accessor.hlsl"
#include "nbl/builtin/hlsl/functional.hlsl"
#include "nbl/builtin/hlsl/subgroup/bitonic_sort.hlsl"
#include "nbl/builtin/hlsl/bit.hlsl"
#include "nbl/builtin/hlsl/workgroup/shuffle.hlsl"
#include "nbl/builtin/hlsl/workgroup/basic.hlsl"

namespace nbl
{
namespace hlsl
{
namespace workgroup
{
namespace bitonic_sort
{
// Reorder: non-type parameters FIRST, then typename parameters with defaults
// This matches FFT's pattern and avoids DXC bugs
template<uint16_t _ElementsPerInvocationLog2, uint16_t _WorkgroupSizeLog2, typename KeyType, typename ValueType, typename Comparator = less<KeyType> >
struct bitonic_sort_config
{
	using key_t = KeyType;
	using value_t = ValueType;
	using comparator_t = Comparator;

	NBL_CONSTEXPR_STATIC_INLINE uint16_t ElementsPerInvocationLog2 = _ElementsPerInvocationLog2;
	NBL_CONSTEXPR_STATIC_INLINE uint16_t WorkgroupSizeLog2 = _WorkgroupSizeLog2;

	NBL_CONSTEXPR_STATIC_INLINE uint32_t ElementsPerInvocation = 1u << ElementsPerInvocationLog2;
	NBL_CONSTEXPR_STATIC_INLINE uint32_t WorkgroupSize = 1u << WorkgroupSizeLog2;
};
}

template<typename Config, class device_capabilities = void>
struct BitonicSort;


template<uint16_t ElementsPerInvocationLog2, uint16_t WorkgroupSizeLog2, typename KeyType, typename ValueType, typename Comparator, class device_capabilities>
struct BitonicSort<bitonic_sort::bitonic_sort_config<ElementsPerInvocationLog2, WorkgroupSizeLog2, KeyType, ValueType, Comparator>, device_capabilities>
{
	using config_t = bitonic_sort::bitonic_sort_config<ElementsPerInvocationLog2, WorkgroupSizeLog2, KeyType, ValueType, Comparator>;
	using key_t = KeyType;
	using value_t = ValueType;
	using comparator_t = Comparator;

	using SortConfig = subgroup::bitonic_sort_config<key_t, value_t, comparator_t>;

	template<typename SharedMemoryAccessor>
	static void mergeStage(NBL_REF_ARG(SharedMemoryAccessor) sharedmemAccessor, uint32_t stage, bool bitonicAscending, uint32_t invocationID, NBL_REF_ARG(key_t) loKey, NBL_REF_ARG(key_t) hiKey,
	NBL_REF_ARG(value_t) loVal, NBL_REF_ARG(value_t) hiVal)
	{
		NBL_CONSTEXPR_STATIC_INLINE uint32_t WorkgroupSize = config_t::WorkgroupSize;
		using adaptor_t = accessor_adaptors::StructureOfArrays<SharedMemoryAccessor, key_t, value_t, 1, WorkgroupSize>;
		adaptor_t sharedmemAdaptor;
		sharedmemAdaptor.accessor = sharedmemAccessor;

		const uint32_t subgroupSizeLog2 = glsl::gl_SubgroupSizeLog2();

		[unroll]
		for (uint32_t pass = 0; pass <= stage; pass++)
		{
			// Stride calculation: stage S merges 2^(S+1) subgroups
			const uint32_t stridePower = (stage - pass + 1) + subgroupSizeLog2;
			const uint32_t stride = 1u << stridePower;
			const uint32_t threadStride = stride >> 1;

			// Separate shuffles for lo/hi streams (two-round shuffle as per PR review)
			// TODO: Consider single-round shuffle of key-value pairs for better performance
			key_t pLoKey = loKey;
			shuffleXor(pLoKey, threadStride, sharedmemAdaptor);
			value_t pLoVal = loVal;
			shuffleXor(pLoVal, threadStride, sharedmemAdaptor);

			key_t pHiKey = hiKey;
			shuffleXor(pHiKey, threadStride, sharedmemAdaptor);
			value_t pHiVal = hiVal;
			shuffleXor(pHiVal, threadStride, sharedmemAdaptor);

			const bool isUpper = (invocationID & threadStride) != 0;
			const bool takeLarger = isUpper == bitonicAscending;

			comparator_t comp;

			// lo update
			const bool loSelfSmaller = comp(loKey, pLoKey);
			const bool takePartnerLo = takeLarger ? loSelfSmaller : !loSelfSmaller;
			loKey = takePartnerLo ? pLoKey : loKey;
			loVal = takePartnerLo ? pLoVal : loVal;

			// hi update
			const bool hiSelfSmaller = comp(hiKey, pHiKey);
			const bool takePartnerHi = takeLarger ? hiSelfSmaller : !hiSelfSmaller;
			hiKey = takePartnerHi ? pHiKey : hiKey;
			hiVal = takePartnerHi ? pHiVal : hiVal;

			sharedmemAdaptor.workgroupExecutionAndMemoryBarrier();
		}
	}

	template<typename Accessor, typename SharedMemoryAccessor>
	static void __call(
	NBL_REF_ARG(Accessor) accessor, NBL_REF_ARG(SharedMemoryAccessor) sharedmemAccessor,
	NBL_REF_ARG(key_t) loKey, NBL_REF_ARG(key_t) hiKey,
	NBL_REF_ARG(value_t) loVal, NBL_REF_ARG(value_t) hiVal)
	{
		NBL_CONSTEXPR_STATIC_INLINE uint32_t WorkgroupSize = config_t::WorkgroupSize;

		const uint32_t invocationID = glsl::gl_LocalInvocationID().x;
		const uint32_t subgroupSizeLog2 = glsl::gl_SubgroupSizeLog2();
		const uint32_t subgroupSize = 1u << subgroupSizeLog2;
		const uint32_t subgroupID = glsl::gl_SubgroupID();
		const uint32_t numSubgroups = WorkgroupSize / subgroupSize;
		const uint32_t numSubgroupsLog2 = findMSB(numSubgroups);


		const bool subgroupAscending = (subgroupID & 1) == 0;
		subgroup::bitonic_sort<SortConfig>::__call(subgroupAscending, loKey, hiKey, loVal, hiVal);

		
		[unroll]
		for (uint32_t stage = 0; stage < numSubgroupsLog2; ++stage)
		{
			const bool isLastStage = (stage == numSubgroupsLog2 - 1);
			const bool bitonicAscending = isLastStage ? true : !bool(invocationID & (subgroupSize << (stage + 1)));

			mergeStage(sharedmemAccessor, stage, bitonicAscending, invocationID, loKey, hiKey, loVal, hiVal);

			const uint32_t subgroupInvocationID = glsl::gl_SubgroupInvocationID();
			subgroup::bitonic_sort<SortConfig>::mergeStage(subgroupSizeLog2, bitonicAscending, subgroupInvocationID, loKey, hiKey, loVal, hiVal);
		}
		

		// Final: ensure lo <= hi within each thread (for ascending sort)
		comparator_t comp;
		if (comp(hiKey, loKey))
		{
			// Swap keys
			key_t tempKey = loKey;
			loKey = hiKey;
			hiKey = tempKey;
			// Swap values
			value_t tempVal = loVal;
			loVal = hiVal;
			hiVal = tempVal;
		}
	}
};

}
}
}

#endif
