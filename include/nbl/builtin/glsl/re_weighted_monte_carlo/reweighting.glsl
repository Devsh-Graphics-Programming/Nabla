#ifndef _NBL_BUILTIN_GLSL_RE_WEIGHTED_MONTE_CARLO_SPLATTING_INCLUDED_
#define _NBL_BUILTIN_GLSL_RE_WEIGHTED_MONTE_CARLO_SPLATTING_INCLUDED_


struct nbl_glsl_RWMC_ReweightingParameters
{
	uint lastCascadeIx;
	float initialEmin; // a minimum image brightness that we always consider reliable
	float rcp_base;
	float rcp_N; // 1/N
	float one_over_kappa;
	float colorReliabilityFactor;
	float N_over_kappa; // N / kappa
};

nbl_glsl_RWMC_ReweightingParameters nbl_glsl_RWMC_computeReweightingParameters(uint cascadeCount, float base, uint spp, float minReliableLuma, float kappa)
{
	nbl_glsl_RWMC_ReweightingParameters retval;
	retval.lastCascadeIx = cascadeCount-1u;
	retval.initialEmin = minReliableLuma;
	retval.rcp_base = 1.f/base;
	const float N = float(spp);
	retval.rcp_N = 1.f/N;
	retval.one_over_kappa = 1.f/kappa;
	// if not interested in exact expected value estimation (kappa!=1.f), can usually accept a bit more variance relative to the image brightness we already have
	// allow up to ~<cascadeBase> more energy in one sample to lessen bias in some cases
	retval.colorReliabilityFactor = mix(base,1.f,retval.one_over_kappa);
	retval.N_over_kappa = N*retval.one_over_kappa;
	return retval;
}

#ifndef __cplusplus

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
	float normalizedCenterLuma;
	float normalizedNeighbourhoodAverageLuma;
};

nbl_glsl_RWMC_CascadeSample nbl_glsl_RWMC_sampleCascade(in ivec2 coord, in uint cascadeIndex, in float rcp_base_i)
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
	retval.normalizedNeighbourhoodAverageLuma = retval.normalizedCenterLuma = nbl_glsl_RWMC_luma(neighbourhood[4])*rcp_base_i;
	retval.normalizedNeighbourhoodAverageLuma = (nbl_glsl_RWMC_luma(excl_hood_sum)*rcp_base_i+retval.normalizedNeighbourhoodAverageLuma)/9.f;
	return retval;
}

vec3 nbl_glsl_RWMC_reweight(in nbl_glsl_RWMC_ReweightingParameters params, in ivec2 coord)
{
	float rcp_base_i = 1.f;
	nbl_glsl_RWMC_CascadeSample curr = nbl_glsl_RWMC_sampleCascade(coord,0u,rcp_base_i);

	vec3 accum = vec3(0.f);
	float Emin = params.initialEmin;

	float prev_normalizedCenterLuma, prev_normalizedNeighbourhoodAverageLuma;
	for (uint i=0; i<=params.lastCascadeIx; i++)
	{
		const bool notFirstCascade = i!=0u;
		const bool notLastCascade = i!=params.lastCascadeIx;

		nbl_glsl_RWMC_CascadeSample next;
		if (notLastCascade)
		{
			rcp_base_i *= params.rcp_base;
			next = nbl_glsl_RWMC_sampleCascade(coord,i+1u,rcp_base_i);
		}


		float reliability = 1.f;
		// sample counting-based reliability estimation
		if (params.one_over_kappa<=1.f)
		{
			float localReliability = curr.normalizedCenterLuma;
			// reliability in 3x3 pixel block (see robustness)
			float globalReliability = curr.normalizedNeighbourhoodAverageLuma;
			if (notFirstCascade)
			{
				localReliability += prev_normalizedCenterLuma;
				globalReliability += prev_normalizedNeighbourhoodAverageLuma;
			}
			if (notLastCascade)
			{
				localReliability += next.normalizedCenterLuma;
				globalReliability += next.normalizedNeighbourhoodAverageLuma;
			}
			// check if above minimum sampling threshold (avg 9 sample occurences in 3x3 neighbourhood), then use per-pixel reliability (NOTE: tertiary op is in reverse)
			reliability = globalReliability<params.rcp_N ? globalReliability:localReliability;
			{
				const float accumLuma = nbl_glsl_RWMC_luma(accum);
				if (accumLuma>Emin)
					Emin = accumLuma;

				const float colorReliability = Emin*rcp_base_i*params.colorReliabilityFactor;
	
				reliability += colorReliability;
				reliability *= params.N_over_kappa;
				reliability -= params.one_over_kappa;
				reliability = clamp(reliability*0.5f,0.f,1.f);
			}
		}
		accum += curr.centerValue*reliability;


		prev_normalizedCenterLuma = curr.normalizedCenterLuma;
		prev_normalizedNeighbourhoodAverageLuma = curr.normalizedNeighbourhoodAverageLuma;
		curr = next;
	}

	return accum;
}
#endif


#endif