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
namespace bitonic_sort
{
template<typename KeyType, typename ValueType, typename Comparator = less<KeyType>>
struct bitonic_sort_config
{
	using key_t = KeyType;
	using value_t = ValueType;
	using comparator_t = Comparator;
};

template<bool Ascending, typename Config, class device_capabilities = void>
struct bitonic_sort;

template<bool Ascending, typename KeyType, typename ValueType, typename Comparator, class device_capabilities>
struct bitonic_sort<Ascending, bitonic_sort_config<KeyType, ValueType, Comparator>, device_capabilities>
{
	using config_t = bitonic_sort_config<KeyType, ValueType, Comparator>;
	using key_t = typename config_t::key_t;
	using value_t = typename config_t::value_t;
	using comparator_t = typename config_t::comparator_t;

	struct KeyValuePair
	{
		key_t key;
		value_t val;
	};

	static void compareAndSwap(bool ascending, NBL_REF_ARG(key_t) loKey, NBL_REF_ARG(key_t) hiKey,
	                           NBL_REF_ARG(value_t) loVal, NBL_REF_ARG(value_t) hiVal)
	{
		comparator_t comp;
		const bool shouldSwap = ascending ? comp(hiKey, loKey) : comp(loKey, hiKey);

		if (shouldSwap)
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

	static void bitonicMergeStep(uint32_t stride, bool ascending,
	                              NBL_REF_ARG(key_t) loKey, NBL_REF_ARG(key_t) hiKey,
	                              NBL_REF_ARG(value_t) loVal, NBL_REF_ARG(value_t) hiVal)
	{
		const uint32_t invocationID = glsl::gl_SubgroupInvocationID();

		const bool topHalf = bool(invocationID & stride);

		KeyValuePair toTrade;
		toTrade.key = topHalf ? loKey : hiKey;
		toTrade.val = topHalf ? loVal : hiVal;

		KeyValuePair exchanged = glsl::subgroupShuffleXor<KeyValuePair>(toTrade, stride);

		if (topHalf)
		{
			loKey = exchanged.key;
			loVal = exchanged.val;
		}
		else
		{
			hiKey = exchanged.key;
			hiVal = exchanged.val;
		}

		compareAndSwap(ascending, loKey, hiKey, loVal, hiVal);
	}

	static void __call(NBL_REF_ARG(key_t) loKey, NBL_REF_ARG(key_t) hiKey,
	                   NBL_REF_ARG(value_t) loVal, NBL_REF_ARG(value_t) hiVal)
	{
		const uint32_t subgroupSize = glsl::gl_SubgroupSize();
		const uint32_t invocationID = glsl::gl_SubgroupInvocationID();

		compareAndSwap(Ascending, loKey, hiKey, loVal, hiVal);

		[unroll]
		for (uint32_t k = 2; k <= subgroupSize; k <<= 1)
		{
			const bool sequenceAscending = ((invocationID & (k >> 1)) == 0);

			const bool dir = Ascending ? sequenceAscending : !sequenceAscending;

			[unroll]
			for (uint32_t stride = k >> 1; stride > 0; stride >>= 1)
			{
				bitonicMergeStep(stride, dir, loKey, hiKey, loVal, hiVal);
			}
		}

		[unroll]
		for (uint32_t stride = subgroupSize; stride > 0; stride >>= 1)
		{
			bitonicMergeStep(stride, Ascending, loKey, hiKey, loVal, hiVal);
		}
	}
};

}
}
}

#endif