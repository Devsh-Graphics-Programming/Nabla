// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_GLSL_BXDF_GGX_INCLUDED_
#define _NBL_BUILTIN_GLSL_BXDF_GGX_INCLUDED_

#include <irr/builtin/glsl/math/constants.glsl>
#include <irr/builtin/glsl/bxdf/ndf/common.glsl>

float nbl_glsl_ggx_trowbridge_reitz(in float a2, in float NdotH2)
{
    float denom = NdotH2 * (a2 - 1.0) + 1.0;
    return a2* nbl_glsl_RECIPROCAL_PI / (denom*denom);
}

float nbl_glsl_ggx_burley_aniso(float anisotropy, float a2, float TdotH, float BdotH, float NdotH)
{
	float antiAniso = 1.0-anisotropy;
	float atab = a2*antiAniso;
	float anisoTdotH = antiAniso*TdotH;
	float anisoNdotH = antiAniso*NdotH;
	float w2 = antiAniso/(BdotH*BdotH+anisoTdotH*anisoTdotH+anisoNdotH*anisoNdotH*a2);
	return w2*w2*atab * nbl_glsl_RECIPROCAL_PI;
}

float nbl_glsl_ggx_aniso(in float TdotH2, in float BdotH2, in float NdotH2, in float ax, in float ay, in float ax2, in float ay2)
{
	float a2 = ax*ay;
	float denom = TdotH2/ax2 + BdotH2/ay2 + NdotH2;
	return nbl_glsl_RECIPROCAL_PI / (a2 * denom * denom);
}

#endif
