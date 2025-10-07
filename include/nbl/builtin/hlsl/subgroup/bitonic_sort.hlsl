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
			template<bool Ascending, typename Config, class device_capabilities = void>
			struct bitonic_sort;
			template<bool Ascending, typename KeyType, typename ValueType, typename Comparator, class device_capabilities>
			struct bitonic_sort<Ascending, bitonic_sort_config<KeyType, ValueType, Comparator>, device_capabilities>
			{
				using config_t = bitonic_sort_config<KeyType, ValueType, Comparator>;
				using key_t = typename config_t::key_t;
				using value_t = typename config_t::value_t;
				using comparator_t = typename config_t::comparator_t;
				// Thread-level compare and swap (operates on lo/hi in registers)
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
				static void __call(NBL_REF_ARG(key_t) loKey, NBL_REF_ARG(key_t) hiKey,
					NBL_REF_ARG(value_t) loVal, NBL_REF_ARG(value_t) hiVal)
				{
					const uint32_t invocationID = glsl::gl_SubgroupInvocationID();
					const uint32_t subgroupSizeLog2 = glsl::gl_SubgroupSizeLog2();
					[unroll]
						for (uint32_t stage = 0; stage <= subgroupSizeLog2; stage++)
						{
							const bool bitonicAscending = (stage == subgroupSizeLog2) ? Ascending : !bool(invocationID & (1u << stage));
							// Passes within this stage
							[unroll]
								for (uint32_t pass = 0; pass <= stage; pass++)
								{
									const uint32_t stride = 1u << (stage - pass); // Element stride
									const uint32_t threadStride = stride >> 1;
									if (threadStride == 0)
									{
										// Local compare and swap for stage 0
										compareAndSwap(bitonicAscending, loKey, hiKey, loVal, hiVal);
									}
									else
									{
										// Shuffle from partner using XOR
										const key_t pLoKey = glsl::subgroupShuffleXor<key_t>(loKey, threadStride);
										const key_t pHiKey = glsl::subgroupShuffleXor<key_t>(hiKey, threadStride);
										const value_t pLoVal = glsl::subgroupShuffleXor<value_t>(loVal, threadStride);
										const value_t pHiVal = glsl::subgroupShuffleXor<value_t>(hiVal, threadStride);
										// Determine if we're upper or lower half
										const bool upperHalf = bool(invocationID & threadStride);
										const bool takeLarger = upperHalf == bitonicAscending;
										comparator_t comp;
										if (takeLarger)
										{
											if (comp(loKey, pLoKey)) { loKey = pLoKey; loVal = pLoVal; }
											if (comp(hiKey, pHiKey)) { hiKey = pHiKey; hiVal = pHiVal; }
										}
										else
										{
											if (comp(pLoKey, loKey)) { loKey = pLoKey; loVal = pLoVal; }
											if (comp(pHiKey, hiKey)) { hiKey = pHiKey; hiVal = pHiVal; }
										}
									}
								}
						}
				}
			};
		}
	}
}
#endif