#ifndef _NBL_BUILTIN_HLSL_WORKGROUP_BITONIC_SORT_INCLUDED_
#define _NBL_BUILTIN_HLSL_WORKGROUP_BITONIC_SORT_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/functional.hlsl"
#include "nbl/builtin/hlsl/workgroup/basic.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/subgroup_basic.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/subgroup_shuffle.hlsl"

namespace nbl
{
namespace hlsl
{
namespace workgroup
{
namespace bitonic_sort
{

// Configuration Template
template<uint16_t E_log2,uint16_t WG_log2,typename Comparator,uint32_t SharedMemDWORDs = 16384u>
struct bitonic_sort_config
{
    static const uint32_t ElementsPerThreadLog2 = E_log2;
    static const uint32_t WorkgroupSizeLog2 = WG_log2;
    static const uint32_t E = 1u << E_log2;  
    static const uint32_t WG = 1u << WG_log2;
    static const uint32_t WorkgroupSize = WG;
    static const uint32_t TotalElements = WG * E; 
    static const uint32_t SharedmemDWORDs = SharedMemDWORDs;

    using comparator_t = Comparator;
};

template<typename T, typename Comp>
inline void compareAndSwap(bool ascending, NBL_REF_ARG(T) a, NBL_REF_ARG(T) b, NBL_CONST_REF_ARG(Comp) comp)
{
    bool needSwap = ascending ? comp(b, a) : comp(a, b);
    if (needSwap)
    {
        T tmp = a;
        a = b;
        b = tmp;
    }
}

// ===========================================
// In-Thread Shuffle + Sort
// For stride < E/2: operations within single thread's data
// Uses local_t = key + 2 bits for position tracking
// ===========================================
template<typename config, typename sortable_t, typename local_t, typename Comp>
inline void inThreadShuffleSortPairs(
    uint32_t stride,
    uint32_t stageSize,
    uint32_t tid,
    sortable_t lo[config::E / 2u],
    sortable_t hi[config::E / 2u],
    NBL_CONST_REF_ARG(Comp) comp)
{
    const uint32_t PairsPerThread = config::E / 2u;

    [unroll]
    for (uint32_t i = 0u; i < PairsPerThread; ++i)
    {
        uint32_t pairGlobalIdx = tid * PairsPerThread + i;
        bool ascending = ((pairGlobalIdx / stageSize) & 0x1u) == 0u;

        if (stride == 1u)
        {
            local_t loLocal = local_t::create(lo[i], i * 2u);
            local_t hiLocal = local_t::create(hi[i], i * 2u + 1u);

            compareAndSwap(ascending, loLocal, hiLocal, comp);

            lo[i] = sortable_t(loLocal.data >> local_t::SHIFT);
            hi[i] = sortable_t(hiLocal.data >> local_t::SHIFT);
        }
        else
        {
            uint32_t partner = i ^ stride;
            if (partner > i && partner < PairsPerThread)
            {
                if (((i / stride) & 0x1u) == 0u)
                {
                    compareAndSwap(ascending, lo[i], lo[partner], comp);
                    compareAndSwap(ascending, hi[i], hi[partner], comp);
                }
            }
        }
    }
}

// ===========================================
// Subgroup Shuffle + Sort
// For E < stride <= E * SubgroupSize
// Uses subgroup_t = key (upper 32) + position (lower 32)
// ===========================================
template<typename config, typename sortable_t, typename subgroup_t, typename Comp>
inline void subgroupShuffleSortPairs(
    uint32_t stride,
    uint32_t stageSize,
    uint32_t tid,
    sortable_t lo[config::E / 2u],
    sortable_t hi[config::E / 2u],
    uint32_t subgroupInvocationID,
    NBL_CONST_REF_ARG(Comp) comp)
{
    const uint32_t PairsPerThread = config::E / 2u;

    uint32_t laneStride = stride / PairsPerThread;

    bool oddThread = (subgroupInvocationID / laneStride) & 0x1u;

    uint32_t loPos[PairsPerThread];
    uint32_t hiPos[PairsPerThread];

    [unroll]
    for (uint32_t i = 0u; i < PairsPerThread; ++i)
    {
        uint32_t pairGlobalIdx = tid * PairsPerThread + i;
        uint32_t myLoPos = pairGlobalIdx * 2u;
        uint32_t myHiPos = pairGlobalIdx * 2u + 1u;

        sortable_t selected = oddThread ? lo[i] : hi[i];
        uint32_t selectedPos = oddThread ? myLoPos : myHiPos;

        subgroup_t packed = subgroup_t::create(selected, selectedPos);

        uint64_t shuffled = glsl::subgroupShuffleXor(packed.data, laneStride);

        sortable_t received = sortable_t(shuffled >> subgroup_t::SHIFT);
        uint32_t receivedPos = uint32_t(shuffled) & ((1u << subgroup_t::SHIFT) - 1u);

        if (oddThread)
        {
            lo[i] = received;
            loPos[i] = receivedPos;
            hiPos[i] = myHiPos;  // hi unchanged
        }
        else
        {
            hi[i] = received;
            hiPos[i] = receivedPos;
            loPos[i] = myLoPos;  // lo unchanged
        }
    }

    [unroll]
    for (uint32_t i = 0u; i < PairsPerThread; ++i)
    {
        uint32_t minPos = loPos[i] < hiPos[i] ? loPos[i] : hiPos[i];
        uint32_t blockSize = stageSize * 2u;
        bool ascending = ((minPos / blockSize) & 0x1u) == 0u;
        compareAndSwap(ascending, lo[i], hi[i], comp);
    }
}

// ===========================================
// Workgroup Shuffle + Sort (via Shared Memory)
// For stride >= E * SubgroupSize
// ===========================================
template<typename config, typename sortable_t, typename workgroup_t, typename SMem, typename Comp>
inline void workgroupShuffleSortPairs(
    uint32_t stride,
    uint32_t stageSize,
    uint32_t tid,
    sortable_t lo[config::E / 2u],
    sortable_t hi[config::E / 2u],
    NBL_REF_ARG(SMem) smem,
    NBL_CONST_REF_ARG(Comp) comp)
{
    const uint32_t PairsPerThread = config::E / 2u;

    uint32_t threadStride = stride / PairsPerThread;
    uint32_t partnerThread = tid ^ threadStride;

    bool oddThread = (tid / threadStride) & 0x1u;

    uint32_t loPos[PairsPerThread];
    uint32_t hiPos[PairsPerThread];

    [unroll]
    for (uint32_t i = 0u; i < PairsPerThread; ++i)
    {
        uint32_t pairGlobalIdx = tid * PairsPerThread + i;
        uint32_t myLoPos = pairGlobalIdx * 2u;
        uint32_t myHiPos = pairGlobalIdx * 2u + 1u;

        uint32_t smemIdx = (tid * PairsPerThread + i) * 2u;

        sortable_t selected = oddThread ? lo[i] : hi[i];
        uint32_t selectedPos = oddThread ? myLoPos : myHiPos;

        smem.set(smemIdx, selected);
        smem.set(smemIdx + 1u, selectedPos);
    }

    smem.workgroupExecutionAndMemoryBarrier();

    [unroll]
    for (uint32_t i = 0u; i < PairsPerThread; ++i)
    {
        uint32_t pairGlobalIdx = tid * PairsPerThread + i;
        uint32_t myLoPos = pairGlobalIdx * 2u;
        uint32_t myHiPos = pairGlobalIdx * 2u + 1u;

        uint32_t partnerSmemIdx = (partnerThread * PairsPerThread + i) * 2u;

        sortable_t received = smem.get(partnerSmemIdx);
        uint32_t receivedPos = smem.get(partnerSmemIdx + 1u);

        if (oddThread)
        {
            lo[i] = received;
            loPos[i] = receivedPos;
            hiPos[i] = myHiPos;  // hi unchanged
        }
        else
        {
            hi[i] = received;
            hiPos[i] = receivedPos;
            loPos[i] = myLoPos;  // lo unchanged
        }
    }

    smem.workgroupExecutionAndMemoryBarrier();

    [unroll]
    for (uint32_t i = 0u; i < PairsPerThread; ++i)
    {
        uint32_t minPos = loPos[i] < hiPos[i] ? loPos[i] : hiPos[i];
        uint32_t blockSize = stageSize * 2u;
        bool ascending = ((minPos / blockSize) & 0x1u) == 0u;
        compareAndSwap(ascending, lo[i], hi[i], comp);
    }
}

// Main BitonicSort Entry Point
template<typename config>
struct BitonicSort
{
    template<
        typename sortable_t,
        typename local_t,
        typename subgroup_t,
        typename workgroup_t,
        typename Accessor,
        typename SMem,
        typename Comp
    >
    static void __call(
        NBL_REF_ARG(Accessor) acc,
        NBL_REF_ARG(SMem) smem,
        NBL_CONST_REF_ARG(Comp) comp)
    {
        uint32_t tid = glsl::gl_LocalInvocationID().x;
        uint32_t subgroupInvocationID = glsl::gl_SubgroupInvocationID();
        uint32_t subgroupSize = glsl::gl_SubgroupSize();

        const uint32_t PairsPerThread = config::E / 2u;

        // Each thread holds E/2 pairs: lo[E/2] and hi[E/2]
        sortable_t lo[PairsPerThread];
        sortable_t hi[PairsPerThread];

        [unroll]
        for (uint32_t i = 0u; i < PairsPerThread; ++i)
        {
            uint32_t baseIdx = tid * config::E;
            acc.template get<sortable_t>(baseIdx + i * 2u, lo[i]);
            acc.template get<sortable_t>(baseIdx + i * 2u + 1u, hi[i]);
        }

        // Pair 0 ascending, Pair 1 descending, Pair 2 ascending, etc. (ADAD pattern)
        [unroll]
        for (uint32_t i = 0u; i < PairsPerThread; ++i)
        {
            uint32_t pairGlobalIdx = tid * PairsPerThread + i;
            bool ascending = (pairGlobalIdx & 0x1u) == 0u;
            compareAndSwap(ascending, lo[i], hi[i], comp);
        }

        // Main bitonic sort stages
        const uint32_t totalPairs = config::TotalElements / 2u;
        const uint32_t pairsPerSubgroup = subgroupSize * PairsPerThread;

        for (uint32_t stageSize = 2u; stageSize <= totalPairs; stageSize <<= 1u)
        {
            for (uint32_t stride = stageSize; stride >= 1u; stride >>= 1u)
            {
                // Thresholds use < (strictly less than) per pseudocode:
                // stride < PairsPerThread -> in-thread
                // stride < PairsPerThread * SubgroupSize -> subgroup
                // stride < PairsPerThread * WorkgroupSize -> workgroup
                if (stride < PairsPerThread)
                {
                    // In-thread operations: stride within single thread's data
                    inThreadShuffleSortPairs<config, sortable_t, local_t>(
                        stride, stageSize, tid, lo, hi, comp);
                }
                else if (stride < pairsPerSubgroup)
                {
                    // Subgroup shuffle operations: stride crosses threads within subgroup
                    subgroupShuffleSortPairs<config, sortable_t, subgroup_t>(
                        stride, stageSize, tid, lo, hi, subgroupInvocationID, comp);
                }
                else
                {
                    // Workgroup shuffle via shared memory: stride crosses subgroups
                    workgroupShuffleSortPairs<config, sortable_t, workgroup_t>(
                        stride, stageSize, tid, lo, hi, smem, comp);
                }
            }
        }

        [unroll]
        for (uint32_t i = 0u; i < PairsPerThread; ++i)
        {
            uint32_t baseIdx = tid * config::E;
            acc.template set<sortable_t>(baseIdx + i * 2u, lo[i]);
            acc.template set<sortable_t>(baseIdx + i * 2u + 1u, hi[i]);
        }
    }
};

} // namespace bitonic_sort
} // namespace workgroup
} // namespace hlsl
} // namespace nbl

#endif
