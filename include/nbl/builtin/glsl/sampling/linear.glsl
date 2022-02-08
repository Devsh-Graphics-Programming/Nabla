#ifndef _NBL_BUILTIN_GLSL_SAMPLING_LINEAR_INCLUDED_
#define _NBL_BUILTIN_GLSL_SAMPLING_LINEAR_INCLUDED_


float nbl_glsl_sampling_generateLinearSample(in vec2 linearCoeffs, in float u)
{
    const float rcpDiff = 1.0/(linearCoeffs[0]-linearCoeffs[1]);
    const vec2 squaredCoeffs = linearCoeffs*linearCoeffs;
    return abs(rcpDiff)<nbl_glsl_FLT_MAX ? (linearCoeffs[0]-sqrt(mix(squaredCoeffs[0],squaredCoeffs[1],u)))*rcpDiff:u;
}

#endif
