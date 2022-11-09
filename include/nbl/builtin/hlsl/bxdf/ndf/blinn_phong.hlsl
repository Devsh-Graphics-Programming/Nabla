
// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_BXDF_NDF_BLINN_PHONG_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_NDF_BLINN_PHONG_INCLUDED_

#include <nbl/builtin/hlsl/math/constants.hlsl>
#include <nbl/builtin/hlsl/bxdf/ndf/common.hlsl>


namespace nbl
{
namespace hlsl
{

float blinn_phong(in float NdotH, in float n)
{
    return isinf(n) ? FLT_INF : RECIPROCAL_PI*0.5*(n+2.0) * pow(NdotH,n);
}
//ashikhmin-shirley ndf
float blinn_phong(in float NdotH, in float one_minus_NdotH2_rcp, in float TdotH2, in float BdotH2, in float nx, in float ny)
{
    float n = (TdotH2*ny + BdotH2*nx) * one_minus_NdotH2_rcp;

    return (isinf(nx)||isinf(ny)) ?  FLT_INF : sqrt((nx + 2.0)*(ny + 2.0))*RECIPROCAL_PI*0.5 * pow(NdotH,n);
}

}
}

#endif