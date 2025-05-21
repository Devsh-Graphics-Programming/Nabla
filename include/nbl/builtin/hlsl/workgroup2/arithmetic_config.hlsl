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
    // static_assert(WorkgroupSizeLog2<=SubgroupSizeLog2+4, "WorkgroupSize cannot be larger than SubgroupSize*16");
    NBL_CONSTEXPR_STATIC_INLINE uint16_t levels = conditional_value<(WorkgroupSizeLog2>SubgroupSizeLog2),uint16_t,conditional_value<(WorkgroupSizeLog2>SubgroupSizeLog2*2+2),uint16_t,3,2>::value,1>::value;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t value = mpl::max_v<uint32_t, WorkgroupSizeLog2-SubgroupSizeLog2, SubgroupSizeLog2>+SubgroupSizeLog2;
};

template<class VirtualWorkgroup, uint16_t BaseItemsPerInvocation, uint16_t WorkgroupSizeLog2, uint16_t SubgroupSizeLog2>
struct items_per_invocation
{
    NBL_CONSTEXPR_STATIC_INLINE uint16_t ItemsPerInvocationProductLog2 = mpl::max_v<int16_t,WorkgroupSizeLog2-SubgroupSizeLog2*VirtualWorkgroup::levels,0>;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t value0 = BaseItemsPerInvocation;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t value1 = uint16_t(0x1u) << conditional_value<VirtualWorkgroup::levels==3, uint16_t,mpl::min_v<uint16_t,ItemsPerInvocationProductLog2,2>, ItemsPerInvocationProductLog2>::value;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t value2 = uint16_t(0x1u) << mpl::max_v<int16_t,ItemsPerInvocationProductLog2-2,0>;
};

// explicit specializations for cases that don't fit
#define SPECIALIZE_VIRTUAL_WG_SIZE_CASE(WGLOG2, SGLOG2, LEVELS, VALUE) template<>\
struct virtual_wg_size_log2<WGLOG2, SGLOG2>\
{\
    NBL_CONSTEXPR_STATIC_INLINE uint16_t levels = LEVELS;\
    NBL_CONSTEXPR_STATIC_INLINE uint16_t value = VALUE;\
};\

SPECIALIZE_VIRTUAL_WG_SIZE_CASE(11,4,3,12);
SPECIALIZE_VIRTUAL_WG_SIZE_CASE(7,7,1,7);
SPECIALIZE_VIRTUAL_WG_SIZE_CASE(6,6,1,6);
SPECIALIZE_VIRTUAL_WG_SIZE_CASE(5,5,1,5);
SPECIALIZE_VIRTUAL_WG_SIZE_CASE(4,4,1,4);
SPECIALIZE_VIRTUAL_WG_SIZE_CASE(3,3,1,3);
SPECIALIZE_VIRTUAL_WG_SIZE_CASE(2,2,1,2);

#undef SPECIALIZE_VIRTUAL_WG_SIZE_CASE
}

template<uint16_t _WorkgroupSizeLog2, uint16_t _SubgroupSizeLog2, uint16_t _ItemsPerInvocation>
struct ArithmeticConfiguration
{
    NBL_CONSTEXPR_STATIC_INLINE uint16_t WorkgroupSizeLog2 = _WorkgroupSizeLog2;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t WorkgroupSize = uint16_t(0x1u) << WorkgroupSizeLog2;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t SubgroupSizeLog2 = _SubgroupSizeLog2;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t SubgroupSize = uint16_t(0x1u) << SubgroupSizeLog2;

    // must have at least enough level 0 outputs to feed a single subgroup
    NBL_CONSTEXPR_STATIC_INLINE uint16_t SubgroupsPerVirtualWorkgroupLog2 = mpl::max_v<uint16_t, WorkgroupSizeLog2-SubgroupSizeLog2, SubgroupSizeLog2>;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t SubgroupsPerVirtualWorkgroup = uint16_t(0x1u) << SubgroupsPerVirtualWorkgroupLog2;

    using virtual_wg_t = impl::virtual_wg_size_log2<WorkgroupSizeLog2, SubgroupSizeLog2>;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t LevelCount = virtual_wg_t::levels;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t VirtualWorkgroupSize = uint16_t(0x1u) << virtual_wg_t::value;
    using items_per_invoc_t = impl::items_per_invocation<virtual_wg_t, _ItemsPerInvocation, WorkgroupSizeLog2, SubgroupSizeLog2>;
    // NBL_CONSTEXPR_STATIC_INLINE uint32_t2 ItemsPerInvocation;    TODO? doesn't allow inline definitions for uint32_t2 for some reason, uint32_t[2] as well ; declaring out of line results in not constant expression
    NBL_CONSTEXPR_STATIC_INLINE uint16_t ItemsPerInvocation_0 = items_per_invoc_t::value0;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t ItemsPerInvocation_1 = items_per_invoc_t::value1;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t ItemsPerInvocation_2 = items_per_invoc_t::value2;
    static_assert(ItemsPerInvocation_1<=4, "3 level scan would have been needed with this config!");

    NBL_CONSTEXPR_STATIC_INLINE uint16_t ElementCount = conditional_value<LevelCount==1,uint16_t,0,conditional_value<LevelCount==3,uint16_t,SubgroupSize*ItemsPerInvocation_2,0>::value + SubgroupSize*ItemsPerInvocation_1>::value;

    static uint32_t virtualSubgroupID(const uint32_t id, const uint32_t offset)
    {
        return offset * (WorkgroupSize >> SubgroupSizeLog2) + id;
    }

    static uint32_t sharedMemCoalescedIndex(const uint32_t id, const uint32_t itemsPerInvocation)
    {
        return (id & (itemsPerInvocation-1)) * SubgroupsPerVirtualWorkgroup + (id/itemsPerInvocation);
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
