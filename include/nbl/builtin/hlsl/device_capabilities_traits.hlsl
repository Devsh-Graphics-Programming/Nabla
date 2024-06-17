// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_DEVICE_CAPABILITIES_TRAITS_INCLUDED_
#define _NBL_BUILTIN_HLSL_DEVICE_CAPABILITIES_TRAITS_INCLUDED_

#include <nbl/builtin/hlsl/member_test_macros.hlsl>

#ifdef __HLSL_VERSION

#include "nbl/video/device_capabilities_traits_testers.h"

#define NBL_GENERATE_GET_OR_DEFAULT(field, ty, default) \
template<typename S, bool = has_member_##field<S>::value> struct get_or_default_##field : integral_constant<ty,S::field> {}; \
template<typename S> struct get_or_default_##field<S,false> : integral_constant<ty,default> {};

namespace nbl
{
namespace hlsl
{

namespace impl
{
    #include "nbl/video/device_capabilities_traits_defaults.h"
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

    #include "nbl/video/device_capabilities_traits_members.h"
};
#undef NBL_GENERATE_GET_OR_DEFAULT
}
}
#endif
#endif
