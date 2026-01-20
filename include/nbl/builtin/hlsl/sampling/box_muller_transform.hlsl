// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_BOX_MULLER_TRANSFORM_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_BOX_MULLER_TRANSFORM_INCLUDED_

#include "nbl/builtin/hlsl/math/functions.hlsl"
#include "nbl/builtin/hlsl/numbers.hlsl"
#include "nbl/builtin/hlsl/sampling/warp_and_pdf.hlsl"

namespace nbl
{
namespace hlsl
{
namespace sampling
{

template<typename T NBL_PRIMARY_REQUIRES(concepts::FloatingPointLikeScalar<T>)
struct BoxMullerTransform
{
    using scalar_type = T;
    using vector2_type = vector<T, 2>;

    // ResamplableSampler concept types
    using domain_type = vector2_type;
    using codomain_type = vector2_type;
    using density_type = scalar_type;
    using sample_type = codomain_and_rcpPdf<codomain_type, density_type>;

    vector2_type backwardPdf(const vector2_type outPos)
    {
        const vector2_type outPos2 = outPos * outPos;
        return vector2_type(nbl::hlsl::exp(scalar_type(-0.5) * (outPos2.x + outPos2.y)), numbers::pi<scalar_type> * scalar_type(0.5) * hlsl::atan2(outPos.y, outPos.x));
    }

    T stddev;
};

} // namespace sampling
} // namespace hlsl
} // namespace nbl

#endif
