#ifndef _NBL_BUILTIN_GLSL_RE_WEIGHTED_MONTE_CARLO_SPLATTING_INCLUDED_
#define _NBL_BUILTIN_GLSL_RE_WEIGHTED_MONTE_CARLO_SPLATTING_INCLUDED_

#ifdef __cplusplus
#define log2 std::log2f
#endif

struct nbl_glsl_RWMC_CascadeParameters
{
    uint penultimateCascadeIx;
    float log2_start;
    float base;
    float rcp_base_minus_1;
    float log2_base;
    float rcp_log2_base;
    float firstCascadeUpperBound;
};

nbl_glsl_RWMC_CascadeParameters nbl_glsl_RWMC_computeCascadeParameters(uint cascadeCount, float start, float base)
{
    nbl_glsl_RWMC_CascadeParameters retval;
    retval.penultimateCascadeIx = cascadeCount-2u;
    retval.log2_start = log2(start);
    retval.base = base;
    retval.rcp_base_minus_1 = 1.f/(base-1.f);
    retval.log2_base = log2(base);
    retval.rcp_log2_base = 1.f/retval.log2_base;
    retval.firstCascadeUpperBound = base*start;
    return retval;
}


#ifdef __cplusplus

#undef log2

#else
struct nbl_glsl_RWMC_SplattingParameters
{
    vec2 cascadeWeights;
    uint lowerCascade;
};

#include <nbl/builtin/glsl/limits/numeric.glsl>
nbl_glsl_RWMC_SplattingParameters nbl_glsl_RWMC_getCascade(in nbl_glsl_RWMC_CascadeParameters params, in float luminance)
{
    //assert(!isnan(luminance));

    // default initialize for values luminance<=params.start
    nbl_glsl_RWMC_SplattingParameters retval;
    retval.cascadeWeights = vec2(1.f,0.f);
    retval.lowerCascade = 0u;

    const float log2_luminance = log2(luminance);
    const float fractionalCascade = (log2_luminance-params.log2_start)*params.rcp_log2_base;
    if (fractionalCascade>0.f)
    {
        const uint baseIndex = uint(fractionalCascade);
        // out of cascade range, we don't expect this to converge
        const bool outOfRange = baseIndex>params.penultimateCascadeIx;
        retval.lowerCascade = outOfRange ? params.penultimateCascadeIx:baseIndex;

        const float upperBound_over_luminance = exp2(float(retval.lowerCascade)*params.log2_base)*params.firstCascadeUpperBound/luminance;
        if (outOfRange)
            retval.cascadeWeights = vec2(0.f,upperBound_over_luminance);
        else
        {
            retval.cascadeWeights.y = (params.base-upperBound_over_luminance)*params.rcp_base_minus_1;
            retval.cascadeWeights.x = 1.f-retval.cascadeWeights.y;
        }
    }
    return retval;
}
#endif


#endif