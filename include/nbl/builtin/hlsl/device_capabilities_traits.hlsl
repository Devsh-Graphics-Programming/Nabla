// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_DEVICE_CAPABILITIES_TRAITS_INCLUDED_
#define _NBL_BUILTIN_HLSL_DEVICE_CAPABILITIES_TRAITS_INCLUDED_

#include <nbl/builtin/hlsl/member_test_macros.hlsl>

#ifdef __HLSL_VERSION

NBL_GENERATE_MEMBER_TESTER(shaderFloat64);
NBL_GENERATE_MEMBER_TESTER(subgroupArithmetic);
NBL_GENERATE_MEMBER_TESTER(fragmentShaderPixelInterlock);
NBL_GENERATE_MEMBER_TESTER(maxOptimallyResidentWorkgroupInvocations);
NBL_GENERATE_MEMBER_TESTER(maxResidentInvocations);

#define NBL_GENERATE_GET_OR_DEFAULT(field, ty, default) \
template<typename S, bool = has_member_##field<S>::value> struct get_or_default_##field : integral_constant<ty,S::field> {}; \
template<typename S> struct get_or_default_##field<S,false> : integral_constant<ty,default> {};

namespace nbl
{
namespace hlsl
{

namespace impl
{
NBL_GENERATE_GET_OR_DEFAULT(shaderFloat64, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(subgroupArithmetic, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(fragmentShaderPixelInterlock, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(maxOptimallyResidentWorkgroupInvocations, uint16_t, 0);
NBL_GENERATE_GET_OR_DEFAULT(maxResidentInvocations, uint32_t, 0);
}


template<typename device_capabilities>
struct device_capabilities_traits
{
    uint32_t computeOptimalPersistentWorkgroupDispatchSize(const uint32_t elementCount, const uint32_t workgroupSize, const uint32_t workgroupSpinningProtection)
    {
	    const uint32_t infinitelyWideDeviceWGCount = (elementCount - 1u) / (workgroupSize * workgroupSpinningProtection) + 1u; // need thtat divAndRoundUp
	    return min(infinitelyWideDeviceWGCount,maxResidentInvocations/maxOptimallyResidentWorkgroupInvocations);
    }
    uint32_t computeOptimalPersistentWorkgroupDispatchSize(const uint32_t elementCount, const uint32_t workgroupSize)
    {
	    return computeOptimalPersistentWorkgroupDispatchSize(elementCount,workgroupSize,1u);
    }

    NBL_CONSTEXPR_STATIC_INLINE bool shaderFloat64                                = impl::get_or_default_shaderFloat64<device_capabilities>::value;
    NBL_CONSTEXPR_STATIC_INLINE bool subgroupArithmetic                           = impl::get_or_default_subgroupArithmetic<device_capabilities>::value;
    NBL_CONSTEXPR_STATIC_INLINE bool fragmentShaderPixelInterlock                 = impl::get_or_default_fragmentShaderPixelInterlock<device_capabilities>::value;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t maxOptimallyResidentWorkgroupInvocations = impl::get_or_default_maxOptimallyResidentWorkgroupInvocations<device_capabilities>::value;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t maxResidentInvocations                   = impl::get_or_default_maxResidentInvocations<device_capabilities>::value;
};

#undef NBL_GENERATE_GET_OR_DEFAULT

}
}
#endif
#endif