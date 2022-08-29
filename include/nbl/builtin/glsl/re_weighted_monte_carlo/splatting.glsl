#version 460 core
#ifndef _NBL_BUILTIN_GLSL_RE_WEIGHTED_MONTE_CARLO_SPLATTING_INCLUDED_
#define _NBL_BUILTIN_GLSL_RE_WEIGHTED_MONTE_CARLO_SPLATTING_INCLUDED_

#include <nbl/builtin/glsl/limits/numeric.glsl>

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

nbl_glsl_RWMC_CascadeParameters nbl_glsl_RWMC_ComputeCascadeParameters(uint cascadeCount, float start, float base)
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

#ifndef __cplusplus
struct nbl_glsl_RWMC_SplattingParameters
{
    vec2 cascadeWeights;
    uint lowerCascade;
};

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

#if 0
uniform vec2 cropOffset;
uniform vec2 size;

uniform bool hasPicture;   // not the lowest layer
uniform bool hasMore;      // more higher layers in the cascade

uniform sampler2D prev; // lower layer in the cascade
uniform sampler2D curr; // current layer in the cascade
uniform sampler2D next; // higher layer in the cascade

uniform float oneOverK;    // 1/kappa
uniform float cascadeBase; // b


// sum of reliabilities in <curr> layer, <prev> layer and <next> layer
float sampleReliability(vec2 coord, const int r) {
	float rel = sampleLayer(curr, coord, r);
	if (hasPicture)
		// scale by N/kappa / b^i_<prev>
		rel += sampleLayer(prev, coord, r) * cascadeBase;
	if (hasMore)
		// scale by N/kappa / b^i_<next>
		rel += sampleLayer(next, coord, r) / cascadeBase;
	// reliability is simply the luminance of the brightness-normalized layer pixels
	return rel;
}

void main()
{
	/* sample counting-based reliability estimation */

	float globalReliability = sampleReliability(gl_FragCoord.xy, 1);
	// reliability of curent pixel
	float localReliability = sampleReliability(gl_FragCoord.xy, 0);

	float reliability = globalReliability - oneOverK;
	// check if above minimum sampling threshold
	if (reliability >= 0.)
		// then use per-pixel reliability
		reliability = localReliability - oneOverK;

	/* color-based reliability estimation */

	float colorReliability = luminance(gl_FragColor) * N_over_kappa_over_base_i;

	// a minimum image brightness that we always consider reliable
	colorReliability = max(colorReliability, 0.05 * N_over_kappa_over_base_i);

	// if not interested in exact expected value estimation, can usually accept a bit
	// more variance relative to the image brightness we already have
	float optimizeForError = max(.0, min(1., oneOverK));
	// allow up to ~<cascadeBase> more energy in one sample to lessen bias in some cases
	colorReliability *= mix(mix(1., cascadeBase, .6), 1., optimizeForError);
	
	reliability = (reliability + colorReliability) * .5;
	reliability = clamp(reliability, 0., 1.);
}
#endif


float nbl_glsl_RWMC_luma(in vec3 val);
#ifndef NBL_GLSL_RWMC_LUMA_DEFINED
//#error
#endif

vec3 nbl_glsl_RWMC_sampleCascadeTexel(in ivec2 coord, in ivec2 offset, in uint cascadeIndex);
#ifndef NBL_GLSL_RWMC_SAMPLE_CASCADE_TEXEL_DEFINED
//#error
#endif

struct nbl_glsl_RWMC_CascadeSample
{
	vec3 centerValue;
	float centerLuma;
	float neighbourhoodAverageLuma;
};

nbl_glsl_RWMC_CascadeSample nbl_glsl_RWMC_sampleCascade(in ivec2 coord, in uint cascadeIndex)
{
	vec3 neighbourhood[9];
	neighbourhood[0] = nbl_glsl_RWMC_sampleCascadeTexel(coord,ivec2(-1,-1),cascadeIndex);
	neighbourhood[1] = nbl_glsl_RWMC_sampleCascadeTexel(coord,ivec2( 0,-1),cascadeIndex);
	neighbourhood[2] = nbl_glsl_RWMC_sampleCascadeTexel(coord,ivec2( 1,-1),cascadeIndex);
	neighbourhood[3] = nbl_glsl_RWMC_sampleCascadeTexel(coord,ivec2(-1, 0),cascadeIndex);
	neighbourhood[4] = nbl_glsl_RWMC_sampleCascadeTexel(coord,ivec2( 0, 0),cascadeIndex);
	neighbourhood[5] = nbl_glsl_RWMC_sampleCascadeTexel(coord,ivec2( 1, 0),cascadeIndex);
	neighbourhood[6] = nbl_glsl_RWMC_sampleCascadeTexel(coord,ivec2(-1, 1),cascadeIndex);
	neighbourhood[7] = nbl_glsl_RWMC_sampleCascadeTexel(coord,ivec2( 0, 1),cascadeIndex);
	neighbourhood[8] = nbl_glsl_RWMC_sampleCascadeTexel(coord,ivec2( 1, 1),cascadeIndex);

	// numerical robustness
	vec3 excl_hood_sum = ((neighbourhood[0]+neighbourhood[1])+(neighbourhood[2]+neighbourhood[3]))+
		((neighbourhood[5]+neighbourhood[6])+(neighbourhood[7]+neighbourhood[8]));
	
	nbl_glsl_RWMC_CascadeSample retval;
	retval.centerValue = neighbourhood[4];
	retval.neighbourhoodAverageLuma = retval.centerLuma = nbl_glsl_RWMC_luma(neighbourhood[4]);
	retval.neighbourhoodAverageLuma += nbl_glsl_RWMC_luma(excl_hood_sum);
	retval.neighbourhoodAverageLuma /= 9.f;
	return retval;
}

struct nbl_glsl_RWMC_ReweighingParameters
{
	uint lastCascadeIx;
	float base;
	float N_over_kappa_base_i; // N / (kappa * b^i_<curr>)
};
vec3 nbl_glsl_RWMC_reweigh(in nbl_glsl_RWMC_ReweighingParameters params, in ivec2 coord)
{
	vec3 accum = vec3(0.f);

	float prev_centerLuma, prev_neighbourhoodAverageLuma;
	nbl_glsl_RWMC_CascadeSample curr = nbl_glsl_RWMC_sampleCascade(coord,0u);
	for (uint i=0; i<=params.lastCascadeIx; i++)
	{
		const bool notFirstCascade = i!=0u;
		const bool notLastCascade = i!=params.lastCascadeIx;

		nbl_glsl_RWMC_CascadeSample next;
		if (notLastCascade)
			next = nbl_glsl_RWMC_sampleCascade(coord,i+1u);


		float reliability = 1.f;
		{
			float localReliability = curr.centerLuma;
			// reliability in 3x3 pixel block (see robustness)
			float globalReliability = curr.neighbourhoodAverageLuma;
			if (notFirstCascade)
			{
				globalReliability += prev_neighbourhoodAverageLuma*params.base;
			}
			if (notLastCascade)
			{
				globalReliability += next.neighbourhoodAverageLuma/params.base;
			}

			globalReliability *= params.N_over_kappa_base_i;
		}
		accum += curr.centerValue*reliability;


		prev_centerLuma = curr.centerLuma;
		prev_neighbourhoodAverageLuma = curr.neighbourhoodAverageLuma;
		curr = next;
	}

	return accum;
}