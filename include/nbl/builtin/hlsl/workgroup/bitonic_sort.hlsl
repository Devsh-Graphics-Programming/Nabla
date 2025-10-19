#ifndef NBL_BUILTIN_HLSL_SUBGROUP_BITONIC_SORT_INCLUDED
#define NBL_BUILTIN_HLSL_SUBGROUP_BITONIC_SORT_INCLUDED
#include "nbl/builtin/hlsl/bitonic_sort/common.hlsl"
#include "nbl/builtin/hlsl/functional.hlsl"
namespace nbl
{
	namespace hlsl
	{
		namespace workgroup
		{
			namespace bitonic_sort
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

					using SortConfig = subgroup::bitonic_sort_config<uint32_t, uint32_t, less<uint32_t> >;


					static void mergeWGStage(uint32_t stage, bool bitonicAscending, uint32_t invocationID, NBL_REF_ARG(key_t) loKey, NBL_REF_ARG(key_t) hiKey,
						NBL_REF_ARG(value_t) loVal, NBL_REF_ARG(value_t) hiVal)
					{
						[unroll]
						for (uint32_t pass = 0; pass <= stage; pass++)
						{
							const uint32_t stride = 1u << ((stage - pass) + subgroupSizeLog2); // Element stride shifts to inter-subgroup scale
							// Shuffle from partner using WG XOR need to implument
							
						}
					}


					static void __call(NBL_REF_ARG(key_t) loKey, NBL_REF_ARG(key_t) hiKey,
						NBL_REF_ARG(value_t) loVal, NBL_REF_ARG(value_t) hiVal)
					{
						const uint32_t invocationID = glsl::gl_SubgroupInvocationID();
						const uint32_t subgroupSizeLog2 = glsl::gl_SubgroupSizeLog2();

						//first sort all subgroup inside wg
						subgroup::bitonic_sort<true, SortConfig>::__call(loKey, hiKey, loVal, hiVal);
						//then we go over first work group shuffle
						//we have n = log2(x), where n is how many wgshuffle we have to do on x(subgroup num) 

						[unroll]
							for (uint32_t stage = 1; stage <= n; ++stage)
							{
								mergeWGStage(stage, Ascending, invocationID, hiKey, loKey, loVal, hiVal);
								subgroup::bitonic_sort<true, SortConfig>::lastMergeStage(subgroupSizeLog2, invocationIDloKey, hiKey, loKey,loVal, hiVal);
								workgroupExecutionAndMemoryBarrier();
							}


					}
				};

			}
		}
	}
}
#endif