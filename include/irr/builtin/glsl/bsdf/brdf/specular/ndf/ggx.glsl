#ifndef _BRDF_SPECULAR_NDF_GGX_INCLUDED_
#define _BRDF_SPECULAR_NDF_GGX_INCLUDED_

#include <irr/builtin/glsl/bsdf/common.glsl>

float irr_glsl_ggx_trowbridge_reitz(in float a2, in float NdotH2)
{
    float denom = NdotH2 * (a2 - 1.0) + 1.0;
    return a2 / (irr_glsl_PI * denom*denom);
}

float irr_glsl_ggx_burley_aniso(float anisotropy, float a2, float TdotH, float BdotH, float NdotH) {
	float antiAniso = 1.0-anisotropy;
	float atab = a2*antiAniso;
	float anisoTdotH = antiAniso*TdotH;
	float anisoNdotH = antiAniso*NdotH;
	float w2 = antiAniso/(BdotH*BdotH+anisoTdotH*anisoTdotH+anisoNdotH*anisoNdotH*a2);
	return w2*w2*atab * irr_glsl_RECIPROCAL_PI;
}

#endif
