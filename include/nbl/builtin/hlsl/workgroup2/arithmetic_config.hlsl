// Copyright (C) 2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_WORKGROUP2_ARITHMETIC_CONFIG_INCLUDED_
#define _NBL_BUILTIN_HLSL_WORKGROUP2_ARITHMETIC_CONFIG_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"

namespace nbl 
{
namespace hlsl
{
namespace workgroup2
{

namespace impl
{
template<uint16_t WorkgroupSizeLog2, uint16_t SubgroupSizeLog2>
struct virtual_wg_size_log2
{
    static_assert(WorkgroupSizeLog2>=SubgroupSizeLog2, "WorkgroupSize cannot be smaller than SubgroupSize");
    static_assert(WorkgroupSizeLog2<=SubgroupSizeLog2*3+4, "WorkgroupSize cannot be larger than (SubgroupSize^3)*16");
    NBL_CONSTEXPR_STATIC_INLINE uint16_t levels = conditional_value<(WorkgroupSizeLog2>SubgroupSizeLog2),uint16_t,conditional_value<(WorkgroupSizeLog2>SubgroupSizeLog2*2+2),uint16_t,3,2>::value,1>::value;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t value = mpl::max_v<uint32_t, SubgroupSizeLog2*levels, WorkgroupSizeLog2>;
    // must have at least enough level 0 outputs to feed a single subgroup
};

template<class VirtualWorkgroup, uint16_t BaseItemsPerInvocation, uint16_t WorkgroupSizeLog2, uint16_t SubgroupSizeLog2>
struct items_per_invocation
{
    NBL_CONSTEXPR_STATIC_INLINE uint16_t ItemsPerInvocationProductLog2 = mpl::max_v<int16_t,WorkgroupSizeLog2-SubgroupSizeLog2*VirtualWorkgroup::levels,0>;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t value0 = BaseItemsPerInvocation;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t value1 = uint16_t(0x1u) << conditional_value<VirtualWorkgroup::levels==3, uint16_t,mpl::min_v<uint16_t,ItemsPerInvocationProductLog2,2>, ItemsPerInvocationProductLog2>::value;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t value2 = uint16_t(0x1u) << mpl::max_v<int16_t,ItemsPerInvocationProductLog2-2,0>;
};
}

template<uint16_t _WorkgroupSizeLog2, uint16_t _SubgroupSizeLog2, uint16_t _ItemsPerInvocation>
struct ArithmeticConfiguration
{
    NBL_CONSTEXPR_STATIC_INLINE uint16_t WorkgroupSizeLog2 = _WorkgroupSizeLog2;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t WorkgroupSize = uint16_t(0x1u) << WorkgroupSizeLog2;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t SubgroupSizeLog2 = _SubgroupSizeLog2;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t SubgroupSize = uint16_t(0x1u) << SubgroupSizeLog2;

    using virtual_wg_t = impl::virtual_wg_size_log2<WorkgroupSizeLog2, SubgroupSizeLog2>;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t LevelCount = virtual_wg_t::levels;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t VirtualWorkgroupSize = uint16_t(0x1u) << virtual_wg_t::value;
    using items_per_invoc_t = impl::items_per_invocation<virtual_wg_t, _ItemsPerInvocation, WorkgroupSizeLog2, SubgroupSizeLog2>;
    // NBL_CONSTEXPR_STATIC_INLINE uint32_t2 ItemsPerInvocation;    TODO? doesn't allow inline definitions for uint32_t2 for some reason, uint32_t[2] as well ; declaring out of line results in not constant expression
    NBL_CONSTEXPR_STATIC_INLINE uint16_t ItemsPerInvocation_0 = items_per_invoc_t::value0;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t ItemsPerInvocation_1 = items_per_invoc_t::value1;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t ItemsPerInvocation_2 = items_per_invoc_t::value2;
    static_assert(ItemsPerInvocation_1<=4, "3 level scan would have been needed with this config!");

    NBL_CONSTEXPR_STATIC_INLINE uint32_t SharedScratchElementCount = conditional_value<LevelCount==1,uint16_t,
        0,
        conditional_value<LevelCount==3,uint16_t,
            SubgroupSize*ItemsPerInvocation_2,
            0
            >::value + SubgroupSize*ItemsPerInvocation_1
        >::value;

    static bool electLast()
    {
        return glsl::gl_SubgroupInvocationID()==SubgroupSize-1;
    }

    static uint32_t virtualSubgroupID(const uint32_t subgroupID, const uint32_t virtualIdx)
    {
        return virtualIdx * (WorkgroupSize >> SubgroupSizeLog2) + subgroupID;
    }

    static uint32_t sharedCoalescedIndexNextLevel(const uint32_t subgroupID, const uint32_t itemsPerInvocation)
    {
        return (subgroupID & (itemsPerInvocation-1)) * SubgroupSize + (subgroupID/itemsPerInvocation);
    }

    static uint32_t sharedCoalescedIndexByComponent(const uint32_t invocationIndex, const uint32_t component)
    {
        return component * SubgroupSize + invocationIndex;
    }
};

template<class T>
struct is_configuration : bool_constant<false> {};

template<uint16_t W, uint16_t S, uint16_t I>
struct is_configuration<ArithmeticConfiguration<W,S,I> > : bool_constant<true> {};

template<typename T>
NBL_CONSTEXPR bool is_configuration_v = is_configuration<T>::value;

}
}
}

#endif
