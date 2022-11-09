
// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_BXDF_GGX_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_GGX_INCLUDED_

#include <nbl/builtin/hlsl/math/constants.hlsl>
#include <nbl/builtin/hlsl/bxdf/ndf/common.hlsl>


namespace nbl
{
namespace hlsl
{
namespace ggx
{

float trowbridge_reitz(in float a2, in float NdotH2)
{
    float denom = NdotH2 * (a2 - 1.0) + 1.0;
    return a2* RECIPROCAL_PI / (denom*denom);
}

float burley_aniso(float anisotropy, float a2, float TdotH, float BdotH, float NdotH)
{
	float antiAniso = 1.0-anisotropy;
	float atab = a2*antiAniso;
	float anisoTdotH = antiAniso*TdotH;
	float anisoNdotH = antiAniso*NdotH;
	float w2 = antiAniso/(BdotH*BdotH+anisoTdotH*anisoTdotH+anisoNdotH*anisoNdotH*a2);
	return w2*w2*atab * RECIPROCAL_PI;
}

float aniso(in float TdotH2, in float BdotH2, in float NdotH2, in float ax, in float ay, in float ax2, in float ay2)
{
	float a2 = ax*ay;
	float denom = TdotH2/ax2 + BdotH2/ay2 + NdotH2;
	return RECIPROCAL_PI / (a2 * denom * denom);
}
	
}
}
}



#endif