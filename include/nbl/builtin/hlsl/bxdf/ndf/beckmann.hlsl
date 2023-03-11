
// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_BXDF_NDF_BECKMANN_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_NDF_BECKMANN_INCLUDED_

#include <nbl/builtin/hlsl/math/constants.hlsl>
#include <nbl/builtin/hlsl/bxdf/ndf/common.hlsl>


namespace nbl
{
namespace hlsl
{
namespace bxdf
{
namespace ndf
{


float beckmann(in float a2, in float NdotH2)
{
    float nom = exp( (NdotH2-1.0)/(a2*NdotH2) ); // exp(x) == exp2(x/log(2)) ?
    float denom = a2*NdotH2*NdotH2;

    return math::RECIPROCAL_PI * nom/denom;
}

float beckmann(in float ax, in float ay, in float ax2, in float ay2, in float TdotH2, in float BdotH2, in float NdotH2)
{
    float nom = exp(-(TdotH2/ax2+BdotH2/ay2)/NdotH2);
    float denom = ax * ay * NdotH2 * NdotH2;

    return math::RECIPROCAL_PI * nom / denom;
}


}
}	
}
}

#endif