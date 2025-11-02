#ifndef NBL_BUILTIN_HLSL_SUBGROUP_BITONIC_SORT_INCLUDED
#define NBL_BUILTIN_HLSL_SUBGROUP_BITONIC_SORT_INCLUDED
#include "nbl/builtin/hlsl/bitonic_sort/common.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/subgroup_basic.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/subgroup_shuffle.hlsl"
#include "nbl/builtin/hlsl/functional.hlsl"
namespace nbl
{
namespace hlsl
{
namespace subgroup
{

template<typename KeyType, typename ValueType, typename Comparator = less<KeyType> >
struct bitonic_sort_config
{
	using key_t = KeyType;
	using value_t = ValueType;
	using comparator_t = Comparator;
};

template<typename Config, class device_capabilities = void>
struct bitonic_sort;

template<typename KeyType, typename ValueType, typename Comparator, class device_capabilities>
struct bitonic_sort<bitonic_sort_config<KeyType, ValueType, Comparator>, device_capabilities>
{
	using config_t = bitonic_sort_config<KeyType, ValueType, Comparator>;
	using key_t = typename config_t::key_t;
	using value_t = typename config_t::value_t;
	using comparator_t = typename config_t::comparator_t;

	static void mergeStage(uint32_t stage, bool bitonicAscending, uint32_t invocationID, NBL_REF_ARG(key_t) loKey, NBL_REF_ARG(key_t) hiKey,
		NBL_REF_ARG(value_t) loVal, NBL_REF_ARG(value_t) hiVal)
	{
		comparator_t comp;

		[unroll]
		for (uint32_t pass = 0; pass <= stage; pass++)
		{
			const uint32_t stride = 1u << (stage - pass); // Element stride
			const uint32_t threadStride = stride >> 1;
			if (threadStride == 0)
			{
				// Local compare and swap for stage 0
				nbl::hlsl::bitonic_sort::compareSwap(bitonicAscending, loKey, hiKey, loVal, hiVal, comp);
			}
			else
			{
				// Shuffle from partner using XOR
				const key_t pLoKey = glsl::subgroupShuffleXor<key_t>(loKey, threadStride);
				const key_t pHiKey = glsl::subgroupShuffleXor<key_t>(hiKey, threadStride);
				const value_t pLoVal = glsl::subgroupShuffleXor<value_t>(loVal, threadStride);
				const value_t pHiVal = glsl::subgroupShuffleXor<value_t>(hiVal, threadStride);

				const bool isUpper = bool(invocationID & threadStride);
				const bool takeLarger = isUpper == bitonicAscending;

				nbl::hlsl::bitonic_sort::compareExchangeWithPartner(takeLarger, loKey, pLoKey, hiKey, pHiKey, loVal, pLoVal, hiVal, pHiVal, comp);
			}
		}
	}

	static void __call(bool ascending, NBL_REF_ARG(key_t) loKey, NBL_REF_ARG(key_t) hiKey,
		NBL_REF_ARG(value_t) loVal, NBL_REF_ARG(value_t) hiVal)
	{
		const uint32_t invocationID = glsl::gl_SubgroupInvocationID();
		const uint32_t subgroupSizeLog2 = glsl::gl_SubgroupSizeLog2();
		[unroll]
		for (uint32_t stage = 0; stage <= subgroupSizeLog2; stage++)
		{
			const bool bitonicAscending = (stage == subgroupSizeLog2) ? ascending : !bool(invocationID & (1u << stage));
			mergeStage(stage, bitonicAscending, invocationID, loKey, hiKey, loVal, hiVal);
		}
	}
};

}
}
}
#endif
