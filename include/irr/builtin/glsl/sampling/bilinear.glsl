#ifndef _NBL_BUILTIN_GLSL_SAMPLING_BILINEAR_INCLUDED_
#define _NBL_BUILTIN_GLSL_SAMPLING_BILINEAR_INCLUDED_

#include <irr/builtin/glsl/sampling/linear.glsl>

// The square's vertex values are defined in Z-order, so indices 0,1,2,3 (xyzw) correspond to (0,0),(1,0),(0,1),(1,1)
vec2 irr_glsl_sampling_generateBilinearSample(out float rcpPdf, in vec4 bilinearCoeffs, vec2 u)
{
    const vec2 twiceAreasUnderXCurve = vec2(bilinearCoeffs[0]+bilinearCoeffs[1],bilinearCoeffs[2]+bilinearCoeffs[3]);
    u.y = irr_glsl_sampling_generateLinearSample(twiceAreasUnderXCurve,u.y);

    const vec2 ySliceEndPoints = vec2(mix(bilinearCoeffs[0],bilinearCoeffs[2],u.y),mix(bilinearCoeffs[1],bilinearCoeffs[3],u.y));
    u.x = irr_glsl_sampling_generateLinearSample(ySliceEndPoints,u.x);

    rcpPdf = (twiceAreasUnderXCurve[0]+twiceAreasUnderXCurve[1])/(4.0*mix(ySliceEndPoints[0],ySliceEndPoints[1],u.x));

    return u;
}

float irr_glsl_sampling_probBilinearSample(in vec4 bilinearCoeffs, vec2 u)
{
    return 4.0*mix(mix(bilinearCoeffs[0],bilinearCoeffs[1],u.x),mix(bilinearCoeffs[2],bilinearCoeffs[3],u.x),u.y)/(bilinearCoeffs[0]+bilinearCoeffs[1]+bilinearCoeffs[2]+bilinearCoeffs[3]);
}

#endif
