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

// ==================== ElementsPerThreadLog2 = 1 Specialization (No Virtual Threading) ====================
// This handles arrays of size WorkgroupSize * 2 using subgroup + workgroup operations
template<uint16_t WorkgroupSizeLog2, typename KeyType, typename ValueType, typename Comparator, class device_capabilities>
struct BitonicSort<bitonic_sort::bitonic_sort_config<1, WorkgroupSizeLog2, KeyType, ValueType, Comparator>, device_capabilities>
{
    using config_t = bitonic_sort::bitonic_sort_config<1, WorkgroupSizeLog2, KeyType, ValueType, Comparator>;
    using key_t = KeyType;
    using value_t = ValueType;
    using comparator_t = Comparator;

    using SortConfig = subgroup::bitonic_sort_config<key_t, value_t, comparator_t>;

    template<typename SharedMemoryAccessor>
    static void mergeStage(NBL_REF_ARG(SharedMemoryAccessor) sharedmemAccessor, uint32_t stage, bool bitonicAscending, uint32_t invocationID, NBL_REF_ARG(key_t) loKey, NBL_REF_ARG(key_t) hiKey,
        NBL_REF_ARG(value_t) loVal, NBL_REF_ARG(value_t) hiVal)
    {
        NBL_CONSTEXPR_STATIC_INLINE uint32_t WorkgroupSize = config_t::WorkgroupSize;
        using key_adaptor = accessor_adaptors::StructureOfArrays<SharedMemoryAccessor, key_t, uint32_t, 1, WorkgroupSize>;
        using value_adaptor = accessor_adaptors::StructureOfArrays<SharedMemoryAccessor, value_t, uint32_t, 1, WorkgroupSize>;

        key_adaptor sharedmemAdaptorKey;
        sharedmemAdaptorKey.accessor = sharedmemAccessor;

        value_adaptor sharedmemAdaptorValue;
        sharedmemAdaptorValue.accessor = sharedmemAccessor;

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
            shuffleXor(pLoKey, threadStride, sharedmemAdaptorKey);
            value_t pLoVal = loVal;
            shuffleXor(pLoVal, threadStride, sharedmemAdaptorValue);
            
            sharedmemAdaptorKey.workgroupExecutionAndMemoryBarrier();
            sharedmemAdaptorValue.workgroupExecutionAndMemoryBarrier();

            key_t pHiKey = hiKey;
            shuffleXor(pHiKey, threadStride, sharedmemAdaptorKey);
            value_t pHiVal = hiVal;
            shuffleXor(pHiVal, threadStride, sharedmemAdaptorValue);

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

            sharedmemAdaptorKey.workgroupExecutionAndMemoryBarrier();
            sharedmemAdaptorValue.workgroupExecutionAndMemoryBarrier();
        }
    }

    template<typename Accessor, typename SharedMemoryAccessor>
    static void __call(NBL_REF_ARG(Accessor) accessor, NBL_REF_ARG(SharedMemoryAccessor) sharedmemAccessor)
    {
        NBL_CONSTEXPR_STATIC_INLINE uint32_t WorkgroupSize = config_t::WorkgroupSize;

        const uint32_t invocationID = glsl::gl_LocalInvocationID().x;
        const uint32_t subgroupSizeLog2 = glsl::gl_SubgroupSizeLog2();
        const uint32_t subgroupSize = 1u << subgroupSizeLog2;
        const uint32_t subgroupID = glsl::gl_SubgroupID();
        const uint32_t numSubgroups = WorkgroupSize / subgroupSize;
        const uint32_t numSubgroupsLog2 = findMSB(numSubgroups);

        // Load this thread's 2 elements from accessor
        const uint32_t loIdx = invocationID * 2;
        const uint32_t hiIdx = loIdx + 1;
        key_t loKey, hiKey;
        value_t loVal, hiVal;
        accessor.template get<key_t>(loIdx, loKey);
        accessor.template get<key_t>(hiIdx, hiKey);
        accessor.template get<value_t>(loIdx, loVal);
        accessor.template get<value_t>(hiIdx, hiVal);

        const bool subgroupAscending = (subgroupID & 1) == 0;
        subgroup::bitonic_sort<SortConfig>::__call(subgroupAscending, loKey, hiKey, loVal, hiVal);

        const uint32_t subgroupInvocationID = glsl::gl_SubgroupInvocationID();

        [unroll]
        for (uint32_t stage = 0; stage < numSubgroupsLog2; ++stage)
        {
            const bool bitonicAscending = !bool(invocationID & (subgroupSize << (stage + 1)));

            mergeStage(sharedmemAccessor, stage, bitonicAscending, invocationID, loKey, hiKey, loVal, hiVal);

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

        // Store results back
        accessor.template set<key_t>(loIdx, loKey);
        accessor.template set<key_t>(hiIdx, hiKey);
        accessor.template set<value_t>(loIdx, loVal);
        accessor.template set<value_t>(hiIdx, hiVal);
    }
};
// ==================== ElementsPerThreadLog2 = 2 Specialization (Virtual Threading) ====================
template<uint16_t WorkgroupSizeLog2, typename KeyType, typename ValueType, typename Comparator, class device_capabilities>
struct BitonicSort<bitonic_sort::bitonic_sort_config<2, WorkgroupSizeLog2, KeyType, ValueType, Comparator>, device_capabilities>
{
	using config_t = bitonic_sort::bitonic_sort_config<2, WorkgroupSizeLog2, KeyType, ValueType, Comparator>;
	using simple_config_t = bitonic_sort::bitonic_sort_config<1, WorkgroupSizeLog2, KeyType, ValueType, Comparator>;

	using key_t = KeyType;
    using value_t = ValueType;
	using comparator_t = Comparator;

	template<typename Accessor, typename SharedMemoryAccessor>
	static void __call(NBL_REF_ARG(Accessor) accessor, NBL_REF_ARG(SharedMemoryAccessor) sharedmemAccessor)
	{
	    NBL_CONSTEXPR_STATIC_INLINE uint32_t WorkgroupSize = config_t::WorkgroupSize;
	    NBL_CONSTEXPR_STATIC_INLINE uint32_t ElementsPerThread = config_t::ElementsPerInvocation;
		NBL_CONSTEXPR_STATIC_INLINE uint32_t TotalElements = WorkgroupSize * ElementsPerThread;
		NBL_CONSTEXPR_STATIC_INLINE uint32_t ElementsPerSimpleSort = WorkgroupSize * 2; // E=1 handles WG*2 elements

        const uint32_t threadID = glsl::gl_LocalInvocationID().x;
		comparator_t comp;

		accessor_adaptors::Offset<Accessor> offsetAccessor;
		offsetAccessor.accessor = accessor;

		[unroll]
		for (uint32_t k = 0; k < ElementsPerThread; k += 2)
		{
		    if (k)
			    sharedmemAccessor.workgroupExecutionAndMemoryBarrier();

			offsetAccessor.offset = ElementsPerSimpleSort * (k / 2);

			BitonicSort<simple_config_t, device_capabilities>::template __call(offsetAccessor, sharedmemAccessor);
		}

		accessor = offsetAccessor.accessor;

		const uint32_t simpleLog = hlsl::findMSB(ElementsPerSimpleSort - 1) + 1u;
		const uint32_t totalLog = hlsl::findMSB(TotalElements - 1) + 1u;

		[unroll]
		for (uint32_t blockLog = simpleLog + 1u; blockLog <= totalLog; blockLog++)
		{
		    // Reverse odd halves for bitonic property
			const uint32_t halfLog = blockLog - 1u;
			const uint32_t halfSize = 1u << halfLog;
			const uint32_t numHalves = TotalElements >> halfLog;

			// Process only odd-indexed halves (no thread divergence)
			[unroll]
			for (uint32_t halfIdx = 1u; halfIdx < numHalves; halfIdx += 2u)
			{
				const uint32_t halfBaseIdx = halfIdx << halfLog;

				[unroll]
				for (uint32_t strideLog = halfLog - 1u; strideLog + 1u > 0u; strideLog--)
				{
					const uint32_t stride = 1u << strideLog;
					const uint32_t virtualThreadsInHalf = halfSize >> 1u;

					[unroll]
					for (uint32_t virtualThreadID = threadID; virtualThreadID < virtualThreadsInHalf; virtualThreadID += WorkgroupSize)
					{
						const uint32_t localLoIx = ((virtualThreadID & (~(stride - 1u))) << 1u) | (virtualThreadID & (stride - 1u));
						const uint32_t loIx = halfBaseIdx + localLoIx;
						const uint32_t hiIx = loIx | stride;

						key_t loKeyGlobal, hiKeyGlobal;
						value_t loValGlobal, hiValGlobal;
						accessor.template get<key_t>(loIx, loKeyGlobal);
						accessor.template get<key_t>(hiIx, hiKeyGlobal);
						accessor.template get<value_t>(loIx, loValGlobal);
						accessor.template get<value_t>(hiIx, hiValGlobal);

						key_t tempKey = loKeyGlobal;
						loKeyGlobal = hiKeyGlobal;
						hiKeyGlobal = tempKey;

						value_t tempVal = loValGlobal;
						loValGlobal = hiValGlobal;
						hiValGlobal = tempVal;

						accessor.template set<key_t>(loIx, loKeyGlobal);
						accessor.template set<key_t>(hiIx, hiKeyGlobal);
						accessor.template set<value_t>(loIx, loValGlobal);
						accessor.template set<value_t>(hiIx, hiValGlobal);
					}
					sharedmemAccessor.workgroupExecutionAndMemoryBarrier();
				}
			}

			const uint32_t k = 1u << blockLog;
			[unroll]
			for (uint32_t strideLog = blockLog - 1u; strideLog + 1u > 0u; strideLog--)
			{
			    const uint32_t stride = 1u << strideLog;
                
				[unroll]
				for (uint32_t virtualThreadID = threadID; virtualThreadID < TotalElements / 2; virtualThreadID += WorkgroupSize)
				{
					const uint32_t loIx = ((virtualThreadID & (~(stride - 1u))) << 1u) | (virtualThreadID & (stride - 1u));
					const uint32_t hiIx = loIx | stride;
                       
					const bool bitonicAscending = ((loIx & k) == 0u);

					key_t loKeyGlobal, hiKeyGlobal;
					value_t loValGlobal, hiValGlobal;
					accessor.template get<key_t>(loIx, loKeyGlobal);
					accessor.template get<key_t>(hiIx, hiKeyGlobal);
					accessor.template get<value_t>(loIx, loValGlobal);
					accessor.template get<value_t>(hiIx, hiValGlobal);

					const bool shouldSwap = comp(hiKeyGlobal, loKeyGlobal);
					if (shouldSwap == bitonicAscending)
					{
					    key_t tempKey = loKeyGlobal;
						loKeyGlobal = hiKeyGlobal;
						hiKeyGlobal = tempKey;

						value_t tempVal = loValGlobal;
						loValGlobal = hiValGlobal;
						hiValGlobal = tempVal;
					}

					accessor.template set<key_t>(loIx, loKeyGlobal);
					accessor.template set<key_t>(hiIx, hiKeyGlobal);
					accessor.template set<value_t>(loIx, loValGlobal);
					accessor.template set<value_t>(hiIx, hiValGlobal);
				}
				sharedmemAccessor.workgroupExecutionAndMemoryBarrier();
			}
		}
	}
};
// ==================== ElementsPerThreadLog2 > 2 Specialization (Virtual Threading) ====================
// This handles larger arrays by combining global memory stages with recursive E=1 workgroup sorts
template<uint16_t ElementsPerThreadLog2, uint16_t WorkgroupSizeLog2, typename KeyType, typename ValueType, typename Comparator, class device_capabilities>
struct BitonicSort<bitonic_sort::bitonic_sort_config<ElementsPerThreadLog2, WorkgroupSizeLog2, KeyType, ValueType, Comparator>, device_capabilities>
{
    using config_t = bitonic_sort::bitonic_sort_config<ElementsPerThreadLog2, WorkgroupSizeLog2, KeyType, ValueType, Comparator>;
    using simple_config_t = bitonic_sort::bitonic_sort_config<1, WorkgroupSizeLog2, KeyType, ValueType, Comparator>;

    using key_t = KeyType;
    using value_t = ValueType;
    using comparator_t = Comparator;

    template<typename Accessor, typename SharedMemoryAccessor>
    static void __call(NBL_REF_ARG(Accessor) accessor, NBL_REF_ARG(SharedMemoryAccessor) sharedmemAccessor)
    {
        NBL_CONSTEXPR_STATIC_INLINE uint32_t WorkgroupSize = config_t::WorkgroupSize;
        NBL_CONSTEXPR_STATIC_INLINE uint32_t ElementsPerThread = config_t::ElementsPerInvocation;
        NBL_CONSTEXPR_STATIC_INLINE uint32_t TotalElements = WorkgroupSize * ElementsPerThread;
        NBL_CONSTEXPR_STATIC_INLINE uint32_t ElementsPerSimpleSort = WorkgroupSize * 2;

        const uint32_t threadID = glsl::gl_LocalInvocationID().x;
        comparator_t comp;

        // PHASE 1: Sub-sorts in chunks of size WorkgroupSize*2
        accessor_adaptors::Offset<Accessor> offsetAccessor;
        offsetAccessor.accessor = accessor;

        const uint32_t numSub = TotalElements / ElementsPerSimpleSort;

        [unroll]
        for (uint32_t sub = 0; sub < numSub; sub++)
        {
            if (sub)
                sharedmemAccessor.workgroupExecutionAndMemoryBarrier();

            offsetAccessor.offset = sub * ElementsPerSimpleSort;

            // Call E=1 workgroup sort
            BitonicSort<simple_config_t, device_capabilities>::template __call(offsetAccessor, sharedmemAccessor);
        }

        // PHASE 2: Reverse odd-indexed chunks to form bitonic sequences
        const uint32_t simpleLog = hlsl::findMSB(ElementsPerSimpleSort - 1) + 1u;
        [unroll]
        for (uint32_t sub = 1; sub < numSub; sub += 2) // Only odd-indexed chunks
        {
            offsetAccessor.offset = sub * ElementsPerSimpleSort;
            [unroll]
            for (uint32_t strideLog = simpleLog - 1u; strideLog + 1u > 0u; strideLog--)
            {
                const uint32_t stride = 1u << strideLog;
                [unroll]
                for (uint32_t virtualThreadID = threadID; virtualThreadID < ElementsPerSimpleSort / 2; virtualThreadID += WorkgroupSize)
                {
                    const uint32_t loIx = (((virtualThreadID & (~(stride - 1u))) << 1u) | (virtualThreadID & (stride - 1u))) + offsetAccessor.offset;
                    const uint32_t hiIx = loIx | stride;

                    key_t loKeyGlobal, hiKeyGlobal;
                    value_t loValGlobal, hiValGlobal;
                    accessor.template get<key_t>(loIx, loKeyGlobal);
                    accessor.template get<key_t>(hiIx, hiKeyGlobal);
                    accessor.template get<value_t>(loIx, loValGlobal);
                    accessor.template get<value_t>(hiIx, hiValGlobal);

                    // Always swap to reverse
                    key_t tempKey = loKeyGlobal;
                    loKeyGlobal = hiKeyGlobal;
                    hiKeyGlobal = tempKey;

                    value_t tempVal = loValGlobal;
                    loValGlobal = hiValGlobal;
                    hiValGlobal = tempVal;

                    accessor.template set<key_t>(loIx, loKeyGlobal);
                    accessor.template set<key_t>(hiIx, hiKeyGlobal);
                    accessor.template set<value_t>(loIx, loValGlobal);
                    accessor.template set<value_t>(hiIx, hiValGlobal);
                }
                sharedmemAccessor.workgroupExecutionAndMemoryBarrier();
            }
        }

        // PHASE 3: Global memory bitonic merge
        const uint32_t totalLog = hlsl::findMSB(TotalElements - 1) + 1u;
        [unroll]
        for (uint32_t blockLog = simpleLog + 1u; blockLog <= totalLog; blockLog++)
        {
            const uint32_t k = 1u << blockLog;
            [unroll]
            for (uint32_t strideLog = blockLog - 1u; strideLog + 1u > 0u; strideLog--)
            {
                const uint32_t stride = 1u << strideLog;
                [unroll]
                for (uint32_t virtualThreadID = threadID; virtualThreadID < TotalElements / 2; virtualThreadID += WorkgroupSize)
                {
                    const uint32_t loIx = ((virtualThreadID & (~(stride - 1u))) << 1u) | (virtualThreadID & (stride - 1u));
                    const uint32_t hiIx = loIx | stride;

                    const bool bitonicAscending = ((loIx & k) == 0u);

                    key_t loKeyGlobal, hiKeyGlobal;
                    value_t loValGlobal, hiValGlobal;
                    accessor.template get<key_t>(loIx, loKeyGlobal);
                    accessor.template get<key_t>(hiIx, hiKeyGlobal);
                    accessor.template get<value_t>(loIx, loValGlobal);
                    accessor.template get<value_t>(hiIx, hiValGlobal);

                    const bool shouldSwap = comp(hiKeyGlobal, loKeyGlobal);
                    if (shouldSwap == bitonicAscending)
                    {
                        key_t tempKey = loKeyGlobal;
                        loKeyGlobal = hiKeyGlobal;
                        hiKeyGlobal = tempKey;

                        value_t tempVal = loValGlobal;
                        loValGlobal = hiValGlobal;
                        hiValGlobal = tempVal;
                    }

                    accessor.template set<key_t>(loIx, loKeyGlobal);
                    accessor.template set<key_t>(hiIx, hiKeyGlobal);
                    accessor.template set<value_t>(loIx, loValGlobal);
                    accessor.template set<value_t>(hiIx, hiValGlobal);
                }
                sharedmemAccessor.workgroupExecutionAndMemoryBarrier();
            }
        }
    }
};

}
}
}

#endif
