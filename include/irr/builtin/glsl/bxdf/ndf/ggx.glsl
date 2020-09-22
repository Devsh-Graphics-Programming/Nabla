#ifndef _IRR_BUILTIN_GLSL_BXDF_GGX_INCLUDED_
#define _IRR_BUILTIN_GLSL_BXDF_GGX_INCLUDED_

#include <irr/builtin/glsl/math/constants.glsl>
#include <irr/builtin/glsl/bxdf/ndf/common.glsl>

float irr_glsl_ggx_trowbridge_reitz(in float a2, in float NdotH2)
{
    float denom = NdotH2 * (a2 - 1.0) + 1.0;
    return a2* irr_glsl_RECIPROCAL_PI / (denom*denom);
}

float irr_glsl_ggx_burley_aniso(float anisotropy, float a2, float TdotH, float BdotH, float NdotH)
{
	float antiAniso = 1.0-anisotropy;
	float atab = a2*antiAniso;
	float anisoTdotH = antiAniso*TdotH;
	float anisoNdotH = antiAniso*NdotH;
	float w2 = antiAniso/(BdotH*BdotH+anisoTdotH*anisoTdotH+anisoNdotH*anisoNdotH*a2);
	return w2*w2*atab * irr_glsl_RECIPROCAL_PI;
}

float irr_glsl_ggx_aniso(in float TdotH2, in float BdotH2, in float NdotH2, in float ax, in float ay, in float ax2, in float ay2)
{
	float a2 = ax*ay;
	float denom = TdotH2/ax2 + BdotH2/ay2 + NdotH2;
	return irr_glsl_RECIPROCAL_PI / (a2 * denom * denom);
}

#endif
