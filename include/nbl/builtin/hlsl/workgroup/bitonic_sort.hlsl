#ifndef _NBL_BUILTIN_HLSL_WORKGROUP_BITONIC_SORT_INCLUDED_
#define _NBL_BUILTIN_HLSL_WORKGROUP_BITONIC_SORT_INCLUDED_

#include "nbl/builtin/hlsl/bitonic_sort/common.hlsl"
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
    typename KeyType,
    typename ValueType,
    typename Comparator = less<KeyType>,
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
    

    using key_t = KeyType;
    using value_t = ValueType;
    using comparator_t = Comparator;
    using WGType = WorkgroupType<key_t>;
};

template<typename config, typename Comp>
inline void atomicOpPairs(
    bool ascending,
    WorkgroupType<typename config::key_t> elems[config::E],
    NBL_CONST_REF_ARG(Comp) comp)
{
    
    LocalPasses<WorkgroupType<typename config::key_t>, 1, Comp> localSort;

    [unroll]
    for (uint32_t pairIdx = 0u; pairIdx < config::E / 2u; ++pairIdx)
    {
        WorkgroupType<typename config::key_t> pair[2];
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
    bool ascending,
    WorkgroupType<typename config::key_t> elems[config::E],
    NBL_CONST_REF_ARG(Comp) comp)
{
    uint32_t partner = stride;

    [unroll]
    for (uint32_t i = 0u; i < config::E; i += stride * 2u)
    {
        uint32_t j = i + partner;
        bool valid = j < config::E;
        bool swap = valid && (ascending ? comp(elems[j], elems[i]) : comp(elems[i], elems[j]));

        WorkgroupType<typename config::key_t> tmp = elems[i];

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
    WorkgroupType<typename config::key_t> elems[config::E],
    uint32_t subgroupInvocationID,
    NBL_CONST_REF_ARG(Comp) comp)
{
    bool isUpper = (subgroupInvocationID & stride) != 0u;

    [unroll]
    for (uint32_t i = 0u; i < config::E; ++i)
    {
        typename config::key_t partnerKey = glsl::subgroupShuffleXor(elems[i].key, stride);
        uint32_t partnerIdx = glsl::subgroupShuffleXor(elems[i].workgroupRelativeIndex, stride);

        WorkgroupType<typename config::key_t> partnerElem;
        partnerElem.key = partnerKey;
        partnerElem.workgroupRelativeIndex = partnerIdx;

     
        bool keepPartner = (ascending == isUpper) ? comp(partnerElem, elems[i]) : comp(elems[i], partnerElem);
        if (keepPartner) {
            elems[i] = partnerElem;
        }
    }

    atomicOpPairs<config>(ascending, elems, comp);
}

template<typename config, typename Comp, typename SMem>
inline void workgroupShuffleSortPairs(
    uint32_t stride,
    bool ascending,
    WorkgroupType<typename config::key_t> elems[config::E],
    uint32_t tid,
    NBL_REF_ARG(SMem) smem,
    NBL_CONST_REF_ARG(Comp) comp)
{
    using K = accessor_adaptors::StructureOfArrays<SMem,uint32_t,uint32_t,1,config::Total>;
    using I = accessor_adaptors::StructureOfArrays<SMem,uint32_t,uint32_t,1,config::Total,
        integral_constant<uint32_t,config::Total*sizeof(typename config::key_t)/4u> >;

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
        for (uint32_t i = 0u; i < config::E; ++i)
        {
            if (active)
            {
                k.set(tid*config::E + i, elems[i].key);
                idx.set(tid*config::E + i, elems[i].workgroupRelativeIndex);
            }
        }
        smem.workgroupExecutionAndMemoryBarrier();

        [unroll]
        for (uint32_t i = 0u; i < config::E; ++i)
        {
            if (active)
            {
                uint32_t myElemIdx = tid * config::E + i;
                uint32_t partnerElemIdx = myElemIdx ^ stride;
                uint32_t pTid = partnerElemIdx / config::E;
                uint32_t partnerLocalIdx = partnerElemIdx % config::E;

                WorkgroupType<typename config::key_t> partnerElem;
                k.get(pTid*config::E + partnerLocalIdx, partnerElem.key);
                idx.get(pTid*config::E + partnerLocalIdx, partnerElem.workgroupRelativeIndex);

                bool isUpper = (myElemIdx & stride) != 0u;

                bool keepPartner = (ascending == isUpper) ? comp(partnerElem, elems[i]) : comp(elems[i], partnerElem);
                if (keepPartner) {
                    elems[i] = partnerElem;
                }
            }
        }
        smem.workgroupExecutionAndMemoryBarrier();
    }

    atomicOpPairs<config>(ascending, elems, comp);
}

template<typename config, typename Comp, typename SMem>
inline void ShufflesSort(
    uint32_t stride,
    bool ascending,
    WorkgroupType<typename config::key_t> elems[config::E],
    uint32_t tid,
    uint32_t subgroupInvocationID,
    NBL_REF_ARG(SMem) smem,
    NBL_CONST_REF_ARG(Comp) comp)
{
    if (stride == 1u)
    {
        atomicOpPairs<config>(ascending, elems, comp);
        return;
    }

    const uint32_t E_half = config::E / 2u;
    const uint32_t subgroupThreshold = E_half * glsl::gl_SubgroupSize();
    const uint32_t workgroupThreshold = E_half * config::WG;

    [flatten]
    if (stride < E_half)
    {
        inThreadShuffleSortPairs<config>(stride, ascending, elems, comp);
    }
    else [flatten] if (stride < subgroupThreshold)
    {
        subgroupShuffleSortPairs<config>(stride, ascending, elems, subgroupInvocationID, comp);
    }
    else
    {
        workgroupShuffleSortPairs<config>(stride, ascending, elems, tid, smem, comp);
    }
}

template<typename config>
struct BitonicSort
{
    template<typename Accessor, typename SMem>
    static void __call(Accessor acc, SMem smem)
    {
        uint32_t tid = glsl::gl_LocalInvocationID().x;
        uint32_t subgroupInvocationID = glsl::gl_SubgroupInvocationID();
        WorkgroupType<typename config::key_t> elems[config::E];

        using KVPair = nbl::hlsl::pair<typename config::key_t, typename config::value_t>;

        [unroll]
        for (uint32_t i = 0u; i < config::E; ++i)
        {
            uint32_t idx = tid * config::E + i;
            KVPair kvpair;
            acc.template get<KVPair>(idx, kvpair);
            elems[i].key = kvpair.first;                      // The key to sort by (random number)
            elems[i].workgroupRelativeIndex = kvpair.second;  // The original index to track
        }

        typename config::comparator_t comp;

        atomicOpPairs<config>(true, elems, comp);

        for (uint32_t BigStride = 2u; BigStride <= config::Total; BigStride <<= 1u)
        {
            bool bitonicDir = ((tid * config::E) & BigStride) == 0u;

            ShufflesSort<config>(BigStride, bitonicDir, elems, tid, subgroupInvocationID, smem, comp);

            [unroll]
            for (uint32_t smallStride = BigStride >> 1u; smallStride >= 1u; smallStride >>= 1u)
            {
                bool mergeDir = ((tid * config::E) & smallStride) == 0u;
                ShufflesSort<config>(smallStride, mergeDir, elems, tid, subgroupInvocationID, smem, comp);
            }
        }

        // Write back sorted (sortedKey, originalIndex) pairs
        [unroll]
        for (uint32_t i = 0u; i < config::E; ++i)
        {
            KVPair output;
            output.first = elems[i].key;                      // Sorted key
            output.second = elems[i].workgroupRelativeIndex;  // Original index
            acc.template set<KVPair>(tid * config::E + i, output);
        }
    }
};

} // bitonic_sort
} // workgroup
} // hlsl
} // nbl

#endif