#ifndef NBL_BUILTIN_HLSL_WORKGROUP_BITONIC_SORT_INCLUDED
#define NBL_BUILTIN_HLSL_WORKGROUP_BITONIC_SORT_INCLUDED
#include "nbl/builtin/hlsl/bitonic_sort/common.hlsl"
#include "nbl/builtin/hlsl/memory_accessor.hlsl"
#include "nbl/builtin/hlsl/functional.hlsl"
#include "nbl/builtin/hlsl/subgroup/bitonic_sort.hlsl"
#include "nbl/builtin/hlsl/bit.hlsl"
#include "nbl/builtin/hlsl/workgroup/shuffle.hlsl"
#include "nbl/builtin/hlsl/workgroup/basic.hlsl"
#include "nbl/builtin/hlsl/concepts/accessors/bitonic_sort.hlsl"

namespace nbl
{
namespace hlsl
{
namespace workgroup
{
namespace bitonic_sort
{

template<uint16_t _ElementsPerInvocationLog2, uint16_t _WorkgroupSizeLog2, typename KeyType, typename ValueType, typename Comparator = less<KeyType> NBL_PRIMARY_REQUIRES(_ElementsPerInvocationLog2 >= 1 && _WorkgroupSizeLog2 >= 5)
struct bitonic_sort_config
{
    using key_t = KeyType;
    using value_t = ValueType;
    using comparator_t = Comparator;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t ElementsPerInvocationLog2 = _ElementsPerInvocationLog2;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t WorkgroupSizeLog2 = _WorkgroupSizeLog2;

    NBL_CONSTEXPR_STATIC_INLINE uint32_t ElementsPerInvocation = 1u << ElementsPerInvocationLog2;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t WorkgroupSize = 1u << WorkgroupSizeLog2;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t SharedmemDWORDs = sizeof(pair<key_t, value_t>) / sizeof(uint32_t) * WorkgroupSize;

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
    static void mergeStage(NBL_REF_ARG(SharedMemoryAccessor) sharedmemAccessor, uint32_t stage, bool bitonicAscending, uint32_t invocationID,
        NBL_REF_ARG(nbl::hlsl::pair<key_t, value_t>) loPair, NBL_REF_ARG(nbl::hlsl::pair<key_t, value_t>) hiPair)
    {
        const uint32_t WorkgroupSize = config_t::WorkgroupSize;
        const uint32_t subgroupSizeLog2 = glsl::gl_SubgroupSizeLog2();
        comparator_t comp;

        [unroll]
        for (uint32_t pass = 0; pass <= stage; pass++)
        {
            if (pass)
                sharedmemAccessor.workgroupExecutionAndMemoryBarrier();

            const uint32_t stridePower = (stage - pass + 1) + subgroupSizeLog2;
            const uint32_t stride = 1u << stridePower;
            const uint32_t threadStride = stride >> 1;

            nbl::hlsl::pair<key_t, value_t> pLoPair = loPair;
            shuffleXor(pLoPair, threadStride, sharedmemAccessor);
            sharedmemAccessor.workgroupExecutionAndMemoryBarrier();

            nbl::hlsl::pair<key_t, value_t> pHiPair = hiPair;
            shuffleXor(pHiPair, threadStride, sharedmemAccessor);

            const bool isUpper = (invocationID & threadStride) != 0;
            const bool takeLarger = isUpper == bitonicAscending;

            nbl::hlsl::bitonic_sort::compareExchangeWithPartner(takeLarger, loPair, pLoPair, hiPair, pHiPair, comp);
        }
    }

    template<typename Accessor, typename SharedMemoryAccessor NBL_FUNC_REQUIRES(bitonic_sort::BitonicSortAccessor<Accessor, key_t, value_t>&& bitonic_sort::BitonicSortSharedMemoryAccessor<SharedMemoryAccessor>)
    static void __call(NBL_REF_ARG(Accessor) accessor, NBL_REF_ARG(SharedMemoryAccessor) sharedmemAccessor)
    {
        const uint32_t WorkgroupSize = config_t::WorkgroupSize;

        const uint32_t invocationID = glsl::gl_LocalInvocationID().x;
        const uint32_t subgroupSizeLog2 = glsl::gl_SubgroupSizeLog2();
        const uint32_t subgroupSize = 1u << subgroupSizeLog2;
        const uint32_t subgroupID = glsl::gl_SubgroupID();
        const uint32_t numSubgroups = WorkgroupSize / subgroupSize;
        const uint32_t numSubgroupsLog2 = findMSB(numSubgroups);

        const uint32_t loIdx = invocationID * 2;
        const uint32_t hiIdx = loIdx | 1;

        nbl::hlsl::pair<key_t, value_t> loPair, hiPair;
        accessor.template get<nbl::hlsl::pair<key_t, value_t> >(loIdx, loPair);
        accessor.template get<nbl::hlsl::pair<key_t, value_t> >(hiIdx, hiPair);

        const bool subgroupAscending = (subgroupID & 1) == 0;
        subgroup::bitonic_sort<SortConfig>::__call(subgroupAscending, loPair.first, hiPair.first, loPair.second, hiPair.second);

        const uint32_t subgroupInvocationID = glsl::gl_SubgroupInvocationID();

        [unroll]
        for (uint32_t stage = 0; stage < numSubgroupsLog2; ++stage)
        {
            const bool bitonicAscending = !bool(invocationID & (subgroupSize << (stage + 1)));

            mergeStage(sharedmemAccessor, stage, bitonicAscending, invocationID, loPair, hiPair);

            subgroup::bitonic_sort<SortConfig>::mergeStage(subgroupSizeLog2, bitonicAscending, subgroupInvocationID, loPair.first, hiPair.first, loPair.second, hiPair.second);
        }

        accessor.template set<nbl::hlsl::pair<key_t, value_t> >(loIdx, loPair);
        accessor.template set<nbl::hlsl::pair<key_t, value_t> >(hiIdx, hiPair);
    }
};

// ==================== ElementsPerThreadLog2 > 1 Specialization (Virtual Threading) ====================
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
        const uint32_t WorkgroupSize = config_t::WorkgroupSize;
        const uint32_t ElementsPerThread = config_t::ElementsPerInvocation;
        const uint32_t TotalElements = WorkgroupSize * ElementsPerThread;
        const uint32_t ElementsPerSimpleSort = WorkgroupSize * 2;

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
        sharedmemAccessor.workgroupExecutionAndMemoryBarrier();

        // PHASE 2: Reverse odd-indexed chunks to form bitonic sequences
        const uint32_t simpleLog = hlsl::findMSB(ElementsPerSimpleSort - 1) + 1u;
        [unroll]
        for (uint32_t sub = 1; sub < numSub; sub += 2)
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

                    nbl::hlsl::pair<key_t, value_t> loPair, hiPair;
                    accessor.template get<nbl::hlsl::pair<key_t, value_t> >(loIx, loPair);
                    accessor.template get<nbl::hlsl::pair<key_t, value_t> >(hiIx, hiPair);

                    nbl::hlsl::bitonic_sort::swap(loPair.first, hiPair.first, loPair.second, hiPair.second);

                    accessor.template set<nbl::hlsl::pair<key_t, value_t> >(loIx, loPair);
                    accessor.template set<nbl::hlsl::pair<key_t, value_t> >(hiIx, hiPair);
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

                        nbl::hlsl::pair<key_t, value_t> loPair, hiPair;
                        accessor.template get<nbl::hlsl::pair<key_t, value_t> >(loIx, loPair);
                        accessor.template get<nbl::hlsl::pair<key_t, value_t> >(hiIx, hiPair);

                        nbl::hlsl::bitonic_sort::compareSwap(bitonicAscending, loPair.first, hiPair.first, loPair.second, hiPair.second, comp);

                        accessor.template set<nbl::hlsl::pair<key_t, value_t> >(loIx, loPair);
                        accessor.template set<nbl::hlsl::pair<key_t, value_t> >(hiIx, hiPair);
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
