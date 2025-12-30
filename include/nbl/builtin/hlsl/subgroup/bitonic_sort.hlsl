#ifndef _NBL_BUILTIN_HLSL_SUBGROUP_BITONIC_SORT_INCLUDED_
#define _NBL_BUILTIN_HLSL_SUBGROUP_BITONIC_SORT_INCLUDED_

#include "nbl/builtin/hlsl/bitonic_sort/common.hlsl"
#include "nbl/builtin/hlsl/subgroup/basic.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/subgroup_shuffle.hlsl"

namespace nbl
{
namespace hlsl
{
namespace subgroup
{
namespace bitonic_sort
{
using namespace nbl::hlsl::bitonic_sort;

template<typename KeyType, typename Comparator, class device_capabilities = void>
struct bitonic_sort_wgtype
{
    using WGType = WorkgroupType<KeyType>;
    using key_t = KeyType;
    using comparator_t = Comparator;

    static void mergeStage(
        uint32_t stage,
        bool bitonicAscending,
        uint32_t invocationID,
        NBL_REF_ARG(WGType) lo,
        NBL_REF_ARG(WGType) hi)
    {
        comparator_t comp;

        [unroll]
        for (uint32_t pass = 0u; pass <= stage; ++pass)
        {
            uint32_t stride = 1u << (stage - pass);
            uint32_t partner = stride >> 1;

            if (partner == 0u)
            {
                bool swap = comp(hi.key, lo.key) == bitonicAscending;
                WGType tmp = lo;
                lo.key = swap ? hi.key : lo.key;
                lo.workgroupRelativeIndex = swap ? hi.workgroupRelativeIndex : lo.workgroupRelativeIndex;
                hi.key = swap ? tmp.key : hi.key;
                hi.workgroupRelativeIndex = swap ? tmp.workgroupRelativeIndex : hi.workgroupRelativeIndex;
            }
            else
            {
                bool isUpper = (invocationID & partner) != 0u;

                // Select which element to trade and shuffle members individually
                key_t tradingKey = isUpper ? hi.key : lo.key;
                uint32_t tradingIdx = isUpper ? hi.workgroupRelativeIndex : lo.workgroupRelativeIndex;

                tradingKey = glsl::subgroupShuffleXor(tradingKey, partner);
                tradingIdx = glsl::subgroupShuffleXor(tradingIdx, partner);

                lo.key = isUpper ? lo.key : tradingKey;
                lo.workgroupRelativeIndex = isUpper ? lo.workgroupRelativeIndex : tradingIdx;
                hi.key = isUpper ? tradingKey : hi.key;
                hi.workgroupRelativeIndex = isUpper ? tradingIdx : hi.workgroupRelativeIndex;

                bool swap = comp(hi.key, lo.key) == bitonicAscending;
                WGType tmp = lo;
                lo.key = swap ? hi.key : lo.key;
                lo.workgroupRelativeIndex = swap ? hi.workgroupRelativeIndex : lo.workgroupRelativeIndex;
                hi.key = swap ? tmp.key : hi.key;
                hi.workgroupRelativeIndex = swap ? tmp.workgroupRelativeIndex : hi.workgroupRelativeIndex;
            }
        }
    }

    static void __call(bool ascending, NBL_REF_ARG(WGType) lo, NBL_REF_ARG(WGType) hi)
    {
        uint32_t id = glsl::gl_SubgroupInvocationID();
        uint32_t log2 = glsl::gl_SubgroupSizeLog2();

        [unroll]
        for (uint32_t s = 0u; s <= log2; ++s)
        {
            bool dir = (s == log2) ? ascending : ((id & (1u << s)) != 0u);
            mergeStage(s, dir, id, lo, hi);
        }
    }
};

} // namespace bitonic_sort
} // namespace subgroup
} // namespace hlsl
} // namespace nbl

#endif
