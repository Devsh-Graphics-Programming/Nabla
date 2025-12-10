#ifndef _NBL_BUILTIN_HLSL_WORKGROUP_BITONIC_SORT_INCLUDED_
#define _NBL_BUILTIN_HLSL_WORKGROUP_BITONIC_SORT_INCLUDED_

#include "nbl/builtin/hlsl/bitonic_sort/common.hlsl"
#include "nbl/builtin/hlsl/subgroup/bitonic_sort.hlsl"
#include "nbl/builtin/hlsl/workgroup/basic.hlsl"
#include "nbl/builtin/hlsl/memory_accessor.hlsl"

namespace nbl
{
namespace hlsl
{
namespace workgroup
{
namespace bitonic_sort
{
using namespace nbl::hlsl::bitonic_sort;

template<
    uint16_t E_log2,
    uint16_t WG_log2,
    typename WorkgroupType,
    typename KeyType,
    typename SortableType,
    typename Comparator,
    uint32_t SharedMemDWORDs = 16384u
>
struct bitonic_sort_config
{
    static const uint32_t ElementsPerThreadLog2 = E_log2;
    static const uint32_t WorkgroupSizeLog2 = WG_log2;
    static const uint32_t E = 1u << E_log2;
    static const uint32_t WG = 1u << WG_log2;
    static const uint32_t WorkgroupSize = WG;
    static const uint32_t Total = WG * E;
    static const uint32_t SharedmemDWORDs = SharedMemDWORDs;

    // R = number of elements that can fit in shared memory before barrier
    // Required shared mem per element = 2 * sizeof(uint32_t) for key + index
    static const uint32_t ElementsInShared = SharedMemDWORDs / 2u;  // 2 DWORDs per element

    static const uint32_t R = (ElementsInShared < WG) ? ElementsInShared : WG;

    static const uint32_t Batches = (WG + R - 1u) / R;

    using wg_type = WorkgroupType;
    using key_type = KeyType;
    using sortable_type = SortableType;
    using comparator_t = Comparator;
};

template<typename config, typename Comp>
inline void atomicOpPairs(
    uint32_t BigStride,
    uint32_t tid,
    typename config::wg_type elems[config::E],
    NBL_CONST_REF_ARG(Comp) comp)
{
    LocalPasses<typename config::wg_type, 1, Comp> localSort;

    [unroll]
    for (uint32_t pairIdx = 0u; pairIdx < config::E / 2u; ++pairIdx)
    {
        uint32_t globalPos = tid * config::E + pairIdx * 2u;
        bool ascending = (globalPos & BigStride) == 0u;

        typename config::wg_type pair[2];
        pair[0] = elems[pairIdx * 2u];
        pair[1] = elems[pairIdx * 2u + 1u];

        localSort(ascending, pair, comp);

        elems[pairIdx * 2u] = pair[0];
        elems[pairIdx * 2u + 1u] = pair[1];
    }
}

template<typename config, typename Comp>
inline void inThreadShuffleSortPairs(
    uint32_t stride,
    uint32_t BigStride,
    uint32_t tid,
    typename config::wg_type elems[config::E],
    NBL_CONST_REF_ARG(Comp) comp)
{
    uint32_t partner = stride >> 1;

    [unroll]
    for (uint32_t i = 0u; i < config::E; i += stride)
    {
        uint32_t j = i + partner;
        bool valid = j < config::E;

        uint32_t globalPos = tid * config::E + i;
        bool ascending = (globalPos & BigStride) == 0u;

        bool swap = valid && (comp(elems[j], elems[i]) == ascending);

        typename config::wg_type tmp = elems[i];

        elems[i].key = swap ? elems[j].key : elems[i].key;
        elems[i].workgroupRelativeIndex = swap ? elems[j].workgroupRelativeIndex : elems[i].workgroupRelativeIndex;
        elems[j].key = swap ? tmp.key : elems[j].key;
        elems[j].workgroupRelativeIndex = swap ? tmp.workgroupRelativeIndex : elems[j].workgroupRelativeIndex;
    }
}

template<typename config, typename Comp>
inline void subgroupShuffleSortPairs(
    uint32_t stride,
    bool ascending,
    typename config::wg_type elems[config::E],
    uint32_t subgroupInvocationID,
    NBL_CONST_REF_ARG(Comp) comp)
{
    bool oddThread = (subgroupInvocationID & 0x1u) != 0u;

    [unroll]
    for (uint32_t i = 0u; i < config::E; i += 2u)
    {
        uint32_t j = i + 1u;

        typename config::key_type tradingKey = oddThread ? elems[j].key : elems[i].key;
        uint32_t tradingIdx = oddThread ? elems[j].workgroupRelativeIndex : elems[i].workgroupRelativeIndex;

        tradingKey = glsl::subgroupShuffleXor(tradingKey, 0x1u);
        tradingIdx = glsl::subgroupShuffleXor(tradingIdx, 0x1u);

        if (oddThread)
        {
            elems[i].key = tradingKey;
            elems[i].workgroupRelativeIndex = tradingIdx;
        }
        else
        {
            elems[j].key = tradingKey;
            elems[j].workgroupRelativeIndex = tradingIdx;
        }

        bool swap = comp(elems[j], elems[i]) == ascending;
        typename config::wg_type tmp = elems[i];

        elems[i].key = swap ? elems[j].key : elems[i].key;
        elems[i].workgroupRelativeIndex = swap ? elems[j].workgroupRelativeIndex : elems[i].workgroupRelativeIndex;
        elems[j].key = swap ? tmp.key : elems[j].key;
        elems[j].workgroupRelativeIndex = swap ? tmp.workgroupRelativeIndex : elems[j].workgroupRelativeIndex;
    }
}

template<typename config, typename Comp, typename SMem>
inline void workgroupShuffleSortPairs(
    uint32_t stride,
    bool ascending,
    uint32_t BigStride,
    uint32_t tid,
    typename config::wg_type elems[config::E],
    NBL_REF_ARG(SMem) smem,
    NBL_CONST_REF_ARG(Comp) comp)
{
    using K = accessor_adaptors::StructureOfArrays<SMem,uint32_t,uint32_t,1,config::Total>;
    using I = accessor_adaptors::StructureOfArrays<SMem,uint32_t,uint32_t,1,config::Total,
        integral_constant<uint32_t,config::Total*sizeof(typename config::key_type)/4u> >;

    K k = K(smem);
    I idx = I(smem);

    const uint32_t R = config::R;
    const uint32_t A = config::Batches;

    bool isUpper = ((tid * config::E) & stride) != 0u;

    [unroll]
    for (uint32_t a = 0u; a < A; ++a)
    {
        uint32_t start = a * R;
        uint32_t end = min((a+1u)*R, config::WG);
        bool active = (tid >= start) && (tid < end);

        [unroll]
        for (uint32_t i = 0u; i < config::E/2; ++i)
        {
            uint32_t localIdx = isUpper ? (i + config::E/2) : i;
            if (active && (localIdx < config::E))
            {
                k.set(tid*config::E + localIdx, elems[localIdx].key);
                idx.set(tid*config::E + localIdx, elems[localIdx].workgroupRelativeIndex);
            }
        }
        smem.workgroupExecutionAndMemoryBarrier();

        [unroll]
        for (uint32_t i = 0u; i < config::E/2; ++i)
        {
            uint32_t localIdx = isUpper ? i : (i + config::E/2);
            if (active && (localIdx < config::E))
            {
                uint32_t myElemIdx = tid * config::E + (isUpper ? (i + config::E/2) : i);
                uint32_t partnerElemIdx = myElemIdx ^ stride;
                uint32_t pTid = partnerElemIdx / config::E;
                uint32_t partnerLocalIdx = partnerElemIdx % config::E;

                k.get(pTid*config::E + partnerLocalIdx, elems[localIdx].key);
                idx.get(pTid*config::E + partnerLocalIdx, elems[localIdx].workgroupRelativeIndex);
            }
        }
        smem.workgroupExecutionAndMemoryBarrier();
    }

    atomicOpPairs<config>(BigStride, tid, elems, comp);
}

template<typename config, typename Comp, typename SMem>
inline void ShufflesSort(
    uint32_t stride,
    uint32_t BigStride,
    typename config::wg_type elems[config::E],
    uint32_t tid,
    uint32_t subgroupInvocationID,
    NBL_REF_ARG(SMem) smem,
    NBL_CONST_REF_ARG(Comp) comp)
{
    if (stride == 1u)
    {
        atomicOpPairs<config>(BigStride, tid, elems, comp);
        return;
    }

    const uint32_t E_half = config::E / 2u;
    const uint32_t subgroupThreshold = E_half * glsl::gl_SubgroupSize();
    const uint32_t workgroupThreshold = E_half * config::WG;

    [flatten]
    if (stride < E_half)
    {
        inThreadShuffleSortPairs<config>(stride, BigStride, tid, elems, comp);
    }
    else [flatten] if (stride < subgroupThreshold)
    {
        bool ascending = ((tid * config::E) & BigStride) == 0u;
        subgroupShuffleSortPairs<config>(stride, ascending, elems, subgroupInvocationID, comp);
    }
    else
    {
        bool ascending = ((tid * config::E) & BigStride) == 0u;
        workgroupShuffleSortPairs<config>(stride, ascending, BigStride, tid, elems, smem, comp);
    }
}

template<typename config>
struct BitonicSort
{
    template<typename Accessor, typename SMem, typename ToWorkgroupType, typename FromWorkgroupType>
    static void __call(Accessor acc, SMem smem, ToWorkgroupType toWgType, FromWorkgroupType fromWgType)
    {
        uint32_t tid = glsl::gl_LocalInvocationID().x;
        uint32_t subgroupInvocationID = glsl::gl_SubgroupInvocationID();
        typename config::wg_type elems[config::E];

        using sortable_t = typename config::sortable_type;

        [unroll]
        for (uint32_t i = 0u; i < config::E; ++i)
        {
            uint32_t idx = tid * config::E + i;
            sortable_t sortable;
            acc.template get<sortable_t>(idx, sortable);
            elems[i] = toWgType(sortable, idx);
        }

        typename config::comparator_t comp;

        atomicOpPairs<config>(1u, tid, elems, comp);

        for (uint32_t BigStride = 2u; BigStride <= config::Total; BigStride <<= 1u)
        {
            ShufflesSort<config>(BigStride, BigStride, elems, tid, subgroupInvocationID, smem, comp);

            for (uint32_t stride = 1u; stride < BigStride; stride <<= 1u)
            {
                ShufflesSort<config>(stride, BigStride, elems, tid, subgroupInvocationID, smem, comp);
            }
        }

        [unroll]
        for (uint32_t i = 0u; i < config::E; ++i)
        {
            sortable_t output = fromWgType(elems[i]);
            acc.template set<sortable_t>(tid * config::E + i, output);
        }
    }
};

} // bitonic_sort
} // workgroup
} // hlsl
} // nbl

#endif