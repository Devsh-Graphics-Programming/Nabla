// Copyright (C) 2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SUBGROUP2_ARITHMETIC_PARAMS_INCLUDED_
#define _NBL_BUILTIN_HLSL_SUBGROUP2_ARITHMETIC_PARAMS_INCLUDED_


#include "nbl/builtin/hlsl/device_capabilities_traits.hlsl"
#include "nbl/builtin/hlsl/concepts.hlsl"


namespace nbl
{
namespace hlsl
{
namespace subgroup2
{

#ifdef __HLSL_VERSION
template<typename Config, class BinOp, int32_t _ItemsPerInvocation=1, class device_capabilities=void NBL_PRIMARY_REQUIRES(is_configuration_v<Config> && is_scalar_v<typename BinOp::type_t>)
struct ArithmeticParams
{
    using config_t = Config;
    using binop_t = BinOp;
    using scalar_t = typename BinOp::type_t;
    using type_t = vector<scalar_t, _ItemsPerInvocation>;
    using device_traits = device_capabilities_traits<device_capabilities>;

    NBL_CONSTEXPR_STATIC_INLINE int32_t ItemsPerInvocation = _ItemsPerInvocation;
    NBL_CONSTEXPR_STATIC_INLINE bool UseNativeIntrinsics = device_capabilities_traits<device_capabilities>::shaderSubgroupArithmetic /*&& /*some heuristic for when its faster*/;
    // TODO add a IHV enum to device_capabilities_traits to check !is_nvidia
};
#endif

#ifndef __HLSL_VERSION
#include <sstream>
#include <string>
struct SArithmeticParams
{
    void init(const uint16_t _SubgroupSizeLog2, const uint16_t _ItemsPerInvocation)
    {
        SubgroupSizeLog2 = _SubgroupSizeLog2;
        ItemsPerInvocation = _ItemsPerInvocation;
    }

    // alias should provide Binop and device_capabilities template parameters
    std::string getParamTemplateStructString()
    {
        std::ostringstream os;
        os << "nbl::hlsl::subgroup2::ArithmeticParams<nbl::hlsl::subgroup2::Configuration<" << SubgroupSizeLog2 << ">, Binop," << ItemsPerInvocation << ", device_capabilities>;";
        return os.str();
    }

    uint32_t SubgroupSizeLog2;
    uint32_t ItemsPerInvocation;
};
#endif

}
}
}

#endif
