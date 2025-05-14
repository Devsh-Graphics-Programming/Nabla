// Copyright (C) 2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_WORKGROUP2_CONFIG_INCLUDED_
#define _NBL_BUILTIN_HLSL_WORKGROUP2_CONFIG_INCLUDED_

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
}

template<uint16_t _WorkgroupSizeLog2, uint16_t _SubgroupSizeLog2, uint16_t _ItemsPerInvocation>
struct Configuration
{
    NBL_CONSTEXPR_STATIC_INLINE uint16_t WorkgroupSizeLog2 = _WorkgroupSizeLog2;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t WorkgroupSize = uint16_t(0x1u) << WorkgroupSizeLog2;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t SubgroupSizeLog2 = _SubgroupSizeLog2;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t SubgroupSize = uint16_t(0x1u) << SubgroupSizeLog2;
    static_assert(WorkgroupSizeLog2>=_SubgroupSizeLog2, "WorkgroupSize cannot be smaller than SubgroupSize");

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

    NBL_CONSTEXPR_STATIC_INLINE uint16_t SharedMemSize = conditional_value<LevelCount==3,uint16_t,SubgroupSize*ItemsPerInvocation_2,0>::value + SubgroupsPerVirtualWorkgroup*ItemsPerInvocation_1;
};

// special case when workgroup size 2048 and subgroup size 16 needs 3 levels and virtual workgroup size 4096 to get a full subgroup scan each on level 1 and 2 16x16x16=4096
// specializing with macros because of DXC bug: https://github.com/microsoft/DirectXShaderCom0piler/issues/7007
#define SPECIALIZE_CONFIG_CASE_2048_16(ITEMS_PER_INVOC) template<>\
struct Configuration<11, 4, ITEMS_PER_INVOC>\
{\
    NBL_CONSTEXPR_STATIC_INLINE uint16_t WorkgroupSize = uint16_t(0x1u) << 11u;\
    NBL_CONSTEXPR_STATIC_INLINE uint16_t SubgroupSizeLog2 = uint16_t(4u);\
    NBL_CONSTEXPR_STATIC_INLINE uint16_t SubgroupSize = uint16_t(0x1u) << SubgroupSizeLog2;\
    NBL_CONSTEXPR_STATIC_INLINE uint16_t SubgroupsPerVirtualWorkgroupLog2 = 7u;\
    NBL_CONSTEXPR_STATIC_INLINE uint16_t SubgroupsPerVirtualWorkgroup = 128u;\
    NBL_CONSTEXPR_STATIC_INLINE uint16_t LevelCount = 3u;\
    NBL_CONSTEXPR_STATIC_INLINE uint16_t VirtualWorkgroupSize = uint16_t(0x1u) << 4096;\
    NBL_CONSTEXPR_STATIC_INLINE uint16_t ItemsPerInvocation_0 = ITEMS_PER_INVOC;\
    NBL_CONSTEXPR_STATIC_INLINE uint16_t ItemsPerInvocation_1 = 1u;\
    NBL_CONSTEXPR_STATIC_INLINE uint16_t ItemsPerInvocation_2 = 1u;\
};\

SPECIALIZE_CONFIG_CASE_2048_16(1)
SPECIALIZE_CONFIG_CASE_2048_16(2)
SPECIALIZE_CONFIG_CASE_2048_16(4)

}
}
}

#undef SPECIALIZE_CONFIG_CASE_2048_16

#endif
