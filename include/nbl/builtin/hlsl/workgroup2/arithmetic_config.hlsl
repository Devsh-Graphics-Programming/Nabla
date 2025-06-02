// Copyright (C) 2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_WORKGROUP2_ARITHMETIC_CONFIG_INCLUDED_
#define _NBL_BUILTIN_HLSL_WORKGROUP2_ARITHMETIC_CONFIG_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/tuple.hlsl"

namespace nbl 
{
namespace hlsl
{
namespace workgroup2
{

namespace impl
{
template<uint16_t _WorkgroupSizeLog2, uint16_t _SubgroupSizeLog2>
struct virtual_wg_size_log2
{
    NBL_CONSTEXPR_STATIC_INLINE uint16_t WorkgroupSizeLog2 = _WorkgroupSizeLog2;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t SubgroupSizeLog2 = _SubgroupSizeLog2;
    static_assert(WorkgroupSizeLog2>=SubgroupSizeLog2, "WorkgroupSize cannot be smaller than SubgroupSize");
    static_assert(WorkgroupSizeLog2<=SubgroupSizeLog2*3+4, "WorkgroupSize cannot be larger than (SubgroupSize^3)*16");

    NBL_CONSTEXPR_STATIC_INLINE uint16_t levels = conditional_value<(WorkgroupSizeLog2>SubgroupSizeLog2),uint16_t,conditional_value<(WorkgroupSizeLog2>SubgroupSizeLog2*2+2),uint16_t,3,2>::value,1>::value;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t value = mpl::max_v<uint32_t, SubgroupSizeLog2*levels, WorkgroupSizeLog2>;
    // must have at least enough level 0 outputs to feed a single subgroup
};

template<class VirtualWorkgroup, uint16_t BaseItemsPerInvocation>
struct items_per_invocation
{
    NBL_CONSTEXPR_STATIC_INLINE uint16_t ItemsPerInvocationProductLog2 = mpl::max_v<int16_t,VirtualWorkgroup::WorkgroupSizeLog2-VirtualWorkgroup::SubgroupSizeLog2*VirtualWorkgroup::levels,0>;
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
    static_assert(VirtualWorkgroupSize<=WorkgroupSize*SubgroupSize);

    using items_per_invoc_t = impl::items_per_invocation<virtual_wg_t, _ItemsPerInvocation>;
    using ItemsPerInvocation = tuple<integral_constant<uint16_t,items_per_invoc_t::value0>,integral_constant<uint16_t,items_per_invoc_t::value1>,integral_constant<uint16_t,items_per_invoc_t::value2> >;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t ItemsPerInvocation_0 = items_per_invoc_t::value0;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t ItemsPerInvocation_1 = items_per_invoc_t::value1;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t ItemsPerInvocation_2 = items_per_invoc_t::value2;

    NBL_CONSTEXPR_STATIC_INLINE uint16_t __ItemsPerVirtualWorkgroupLog2 = mpl::max_v<uint16_t, WorkgroupSizeLog2-SubgroupSizeLog2, SubgroupSizeLog2>;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t __ItemsPerVirtualWorkgroup = uint16_t(0x1u) << __ItemsPerVirtualWorkgroupLog2;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t __SubgroupsPerVirtualWorkgroup = __ItemsPerVirtualWorkgroup / ItemsPerInvocation_1;

    // user specified the shared mem size of uint32_ts
    NBL_CONSTEXPR_STATIC_INLINE uint32_t SharedScratchElementCount = conditional_value<LevelCount==1,uint16_t,
        0,
        conditional_value<LevelCount==3,uint16_t,
            SubgroupSize*ItemsPerInvocation_2+__ItemsPerVirtualWorkgroup,
            SubgroupSize*ItemsPerInvocation_1
            >::value
        >::value;

    static bool electLast()
    {
        return glsl::gl_SubgroupInvocationID()==SubgroupSize-1;
    }

    // gets a subgroupID as if each workgroup has (VirtualWorkgroupSize/SubgroupSize) subgroups
    // each subgroup does work (VirtualWorkgroupSize/WorkgroupSize) times, the index denoted by workgroupInVirtualIndex
    static uint32_t virtualSubgroupID(const uint32_t subgroupID, const uint32_t workgroupInVirtualIndex)
    {
        return workgroupInVirtualIndex * (WorkgroupSize >> SubgroupSizeLog2) + subgroupID;
    }

    // get a coalesced index to store for the next level in shared mem, e.g. level 0 -> level 1
    // specify the next level to store values for in template param
    // at level==LevelCount-1, it is guaranteed to have SubgroupSize elements
    template<uint16_t level>
    static uint32_t sharedStoreIndex(const uint32_t subgroupID)
    {
        uint32_t offsetBySubgroup;
        if (level == LevelCount-1)
            offsetBySubgroup = SubgroupSize;
        else
            offsetBySubgroup = __SubgroupsPerVirtualWorkgroup;

        if (level<2)
            return (subgroupID & (ItemsPerInvocation_1-1)) * offsetBySubgroup + (subgroupID/ItemsPerInvocation_1);
        else
            return (subgroupID & (ItemsPerInvocation_2-1)) * offsetBySubgroup + (subgroupID/ItemsPerInvocation_2);
    }

    template<uint16_t level>
    static uint32_t sharedStoreIndexFromVirtualIndex(const uint32_t subgroupID, const uint32_t workgroupInVirtualIndex)
    {
        const uint32_t virtualID = virtualSubgroupID(subgroupID, workgroupInVirtualIndex);
        return sharedStoreIndex<level>(virtualID);
    }

    // get the coalesced index in shared mem at the current level
    template<uint16_t level>
    static uint32_t sharedLoadIndex(const uint32_t invocationIndex, const uint32_t component)
    {
        if (level == LevelCount-1)
            return component * SubgroupSize + invocationIndex;
        else
            return component * __SubgroupsPerVirtualWorkgroup + invocationIndex;
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
