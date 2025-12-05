// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_BASIC_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_BASIC_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/ieee754.hlsl>

namespace nbl
{
namespace hlsl
{
namespace sampling
{

template<typename T NBL_PRIMARY_REQUIRES(concepts::FloatingPointLikeScalar<T>)
struct PartitionRandVariable
{
    using floating_point_type = T;
    using uint_type = typename unsigned_integer_of_size<sizeof(floating_point_type)>::type;

    bool operator()(floating_point_type leftProb, NBL_REF_ARG(floating_point_type) xi, NBL_REF_ARG(floating_point_type) rcpChoiceProb)
    {
        const floating_point_type NEXT_ULP_AFTER_UNITY = bit_cast<floating_point_type>(bit_cast<uint_type>(floating_point_type(1.0)) + uint_type(1u));
        const bool pickRight = xi >= leftProb * NEXT_ULP_AFTER_UNITY;

        // This is all 100% correct taking into account the above NEXT_ULP_AFTER_UNITY
        xi -= pickRight ? leftProb : floating_point_type(0.0);

        rcpChoiceProb = floating_point_type(1.0) / (pickRight ? (floating_point_type(1.0) - leftProb) : leftProb);
        xi *= rcpChoiceProb;

        return pickRight;
    }
};


}
}
}

#endif
