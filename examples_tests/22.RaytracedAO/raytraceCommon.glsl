#ifndef _RAYTRACE_COMMON_GLSL_INCLUDED_
#define _RAYTRACE_COMMON_GLSL_INCLUDED_

#include "virtualGeometry.glsl"


layout(push_constant, row_major) uniform PushConstants
{
	RaytraceShaderCommonData_t cummon;
} pc;

// lights
layout(set = 1, binding = 3, std430) restrict readonly buffer CumulativeLightPDF
{
	uint lightCDF[];
};
layout(set = 1, binding = 4, std430, row_major) restrict readonly buffer Lights
{
	SLight light[];
};

layout(set = 2, binding = 0, row_major) uniform StaticViewData
{
	StaticViewData_t staticViewData;
};
// rng
layout(set = 2, binding = 1) uniform usamplerBuffer quantizedSampleSequence;
// accumulation
layout(set = 2, binding = 2, rg32ui) restrict uniform uimage2DArray accumulation;
// ray data
#include <nbl/builtin/glsl/ext/RadeonRays/ray.glsl>
layout(set = 2, binding = 3, std430) restrict writeonly buffer SinkRays
{
	nbl_glsl_ext_RadeonRays_ray sinkRays[];
};
#include <nbl/builtin/glsl/utils/indirect_commands.glsl>
layout(set = 2, binding = 4) restrict coherent buffer RayCount // maybe remove coherent keyword
{
	uint rayCount[RAYCOUNT_N_BUFFERING];
};
// aovs
layout(set = 2, binding = 5, r32ui) restrict uniform uimage2DArray albedoAOV;
layout(set = 2, binding = 6, r32ui) restrict uniform uimage2DArray normalAOV;
// environment emitter
layout(set = 2, binding = 7) uniform sampler2D envMap; 

void clear_raycount()
{
	if (all(equal(uvec3(0u),gl_GlobalInvocationID)))
		rayCount[(pc.cummon.rayCountWriteIx+1u)&uint(RAYCOUNT_N_BUFFERING_MASK)] = 0u;
}

//
uvec3 get_triangle_indices(in nbl_glsl_ext_Mitsuba_Loader_instance_data_t batchInstanceData, in uint triangleID)
{
	const uint baseTriangleVertex = triangleID*3u+batchInstanceData.padding0;
	return uvec3(
		nbl_glsl_VG_fetchTriangleVertexIndex(baseTriangleVertex,0u),
		nbl_glsl_VG_fetchTriangleVertexIndex(baseTriangleVertex,1u),
		nbl_glsl_VG_fetchTriangleVertexIndex(baseTriangleVertex,2u)
	);
}

// for per pixel inputs
#include <nbl/builtin/glsl/random/xoroshiro.glsl>
#include <nbl/builtin/glsl/utils/transform.glsl>

#include <nbl/builtin/glsl/format/decode.glsl>
#include <nbl/builtin/glsl/format/encode.glsl>
vec3 fetchAccumulation(in uvec3 coord)
{
	const uvec2 data = imageLoad(accumulation,ivec3(coord)).rg;
	return nbl_glsl_decodeRGB19E7(data);
}
void storeAccumulation(in vec3 color, in uvec3 coord)
{
	const uvec2 data = nbl_glsl_encodeRGB19E7(color);
	imageStore(accumulation,ivec3(coord),uvec4(data,0u,0u));
}
void storeAccumulation(in vec3 prev, in vec3 delta, in uvec3 coord)
{
	const vec3 newVal = prev+delta;
	const uvec3 diff = floatBitsToUint(newVal)^floatBitsToUint(prev);
	if (bool((diff.x|diff.y|diff.z)&0x7ffffff0u))
		storeAccumulation(newVal,coord);
}

vec3 fetchAlbedo(in uvec3 coord)
{
	const uint data = imageLoad(albedoAOV,ivec3(coord)).r;
	return nbl_glsl_decodeRGB10A2_UNORM(data).rgb;
}
void storeAlbedo(in vec3 color, in uvec3 coord)
{
	const uint data = nbl_glsl_encodeRGB10A2_UNORM(vec4(color,1.f));
	imageStore(albedoAOV,ivec3(coord),uvec4(data,0u,0u,0u));
}
void storeAlbedo(in vec3 prev, in vec3 delta, in uvec3 coord)
{
	if (any(greaterThan(abs(delta),vec3(1.f/1024.f))))
		storeAlbedo(prev+delta,coord);
}

vec3 fetchWorldspaceNormal(in uvec3 coord)
{
	const uint data = imageLoad(normalAOV,ivec3(coord)).r;
	return nbl_glsl_decodeRGB10A2_SNORM(data).xyz;
}
void storeWorldspaceNormal(in vec3 normal, in uvec3 coord)
{
	const uint data = nbl_glsl_encodeRGB10A2_SNORM(vec4(normal,1.f));
	imageStore(normalAOV,ivec3(coord),uvec4(data,0u,0u,0u));
}
void storeWorldspaceNormal(in vec3 prev, in vec3 delta, in uvec3 coord)
{
	if (any(greaterThan(abs(delta),vec3(1.f/512.f))))
		storeWorldspaceNormal(prev+delta,coord);
}

// due to memory limitations we can only do 6k renders
// so that's 13 bits for width, 12 bits for height, which leaves us with 7 bits for throughput
void packOutPixelLocationAndAoVThroughputFactor(out float val, in uvec2 outPixelLocation, in float aovThroughputFactor)
{
	uint data = outPixelLocation.x;
	data |= outPixelLocation.y<<13u;
	data |= uint(aovThroughputFactor*127.f+0.5f)<<25u;
	val = uintBitsToFloat(data);
}
void unpackOutPixelLocationAndAoVThroughputFactor(in float val, out uvec2 outPixelLocation, out float aovThroughputFactor)
{
	const uint asUint = floatBitsToUint(val);
	outPixelLocation = uvec2(asUint,asUint>>13u)&uvec2(0x1fffu,0x0fffu);
	aovThroughputFactor = float(asUint>>25u)/127.f;
}

#include "bin/runtime_defines.glsl"
#include <nbl/builtin/glsl/ext/MitsubaLoader/material_compiler_compatibility_impl.glsl>
vec3 normalizedV;
vec3 nbl_glsl_MC_getNormalizedWorldSpaceV()
{
	return normalizedV;
}
vec3 normalizedN;
vec3 nbl_glsl_MC_getNormalizedWorldSpaceN()
{
	return normalizedN;
}


bool has_world_transform(in nbl_glsl_ext_Mitsuba_Loader_instance_data_t batchInstanceData)
{
	return true;
}

#include <nbl/builtin/glsl/barycentric/utils.glsl>
mat2x3 dPdBary;
vec3 load_positions(out vec3 geomNormal, in nbl_glsl_ext_Mitsuba_Loader_instance_data_t batchInstanceData, in uvec3 indices)
{
	mat3 positions = mat3(
		nbl_glsl_fetchVtxPos(indices[0],batchInstanceData),
		nbl_glsl_fetchVtxPos(indices[1],batchInstanceData),
		nbl_glsl_fetchVtxPos(indices[2],batchInstanceData)
	);
	const bool tform = has_world_transform(batchInstanceData);
	if (tform)
		positions = mat3(batchInstanceData.tform)*positions;
	//
	for (int i=0; i<2; i++)
		dPdBary[i] = positions[i]-positions[2];
	geomNormal = normalize(cross(dPdBary[0],dPdBary[1]));
	//
	if (tform)
		positions[2] += batchInstanceData.tform[3];
	return positions[2];
}

#ifdef TEX_PREFETCH_STREAM
mat2x3 nbl_glsl_perturbNormal_dPdSomething()
{
	return dPdBary;
}
mat2 dUVdBary;
mat2 nbl_glsl_perturbNormal_dUVdSomething()
{
    return dUVdBary;
}
#endif
#define _NBL_USER_PROVIDED_MATERIAL_COMPILER_GLSL_BACKEND_FUNCTIONS_
#include <nbl/builtin/glsl/material_compiler/common.glsl>


bool needs_texture_prefetch(in nbl_glsl_ext_Mitsuba_Loader_instance_data_t batchInstanceData)
{
	return true;
}

vec3 load_normal_and_prefetch_textures(
	in nbl_glsl_ext_Mitsuba_Loader_instance_data_t batchInstanceData,
	in uvec3 indices, in vec2 compactBary, in vec3 geomNormal,
	in nbl_glsl_MC_oriented_material_t material
#ifdef TEX_PREFETCH_STREAM
	,in mat2 dBarydScreen
#endif
)
{
	// if we ever support spatially varying emissive, we'll need to hoist barycentric computation and UV fetching to the position fetching
	#ifdef TEX_PREFETCH_STREAM
	if (needs_texture_prefetch(batchInstanceData))
	{
		const mat3x2 uvs = mat3x2(
			nbl_glsl_fetchVtxUV(indices[0],batchInstanceData),
			nbl_glsl_fetchVtxUV(indices[1],batchInstanceData),
			nbl_glsl_fetchVtxUV(indices[2],batchInstanceData)
		);
	
		const nbl_glsl_MC_instr_stream_t tps = nbl_glsl_MC_oriented_material_t_getTexPrefetchStream(material);

		dUVdBary = mat2(uvs[0]-uvs[2],uvs[1]-uvs[2]);
		const vec2 UV = dUVdBary*compactBary+uvs[2];
		const mat2 dUVdScreen = nbl_glsl_applyChainRule2D(dUVdBary,dBarydScreen);
		nbl_glsl_MC_runTexPrefetchStream(tps,UV,dUVdScreen*pc.cummon.textureFootprintFactor);
	}
	#endif
	// the rest is always only needed for continuing rays


	// while waiting for the scramble state
	// TODO: optimize, add loads more flags to control this
	const bool needsSmoothNormals = true;
	if (needsSmoothNormals)
	{
		const mat3 normals = mat3(
			nbl_glsl_fetchVtxNormal(indices[0],batchInstanceData),
			nbl_glsl_fetchVtxNormal(indices[1],batchInstanceData),
			nbl_glsl_fetchVtxNormal(indices[2],batchInstanceData)
		);

		// not needed for NEE unless doing Area or Projected Solid Angle Sampling
		vec3 smoothNormal = normals*nbl_glsl_barycentric_expand(compactBary);
		if (has_world_transform(batchInstanceData))
		{
			smoothNormal = vec3(
				dot(batchInstanceData.normalMatrixRow0,smoothNormal),
				dot(batchInstanceData.normalMatrixRow1,smoothNormal),
				dot(batchInstanceData.normalMatrixRow2,smoothNormal)
			);
		}
		// TODO: this check wouldn't be needed if we had `needsSmoothNormals` implemented
		if (!isnan(smoothNormal.x))
			return normalize(smoothNormal);
	}
	return geomNormal;
}

#include <nbl/builtin/glsl/sampling/quantized_sequence.glsl>
vec3 rand3d(in uvec3 scramble_key, in int _sample, int depth)
{
	// decrement depth because first vertex is rasterized and picked with a different sample sequence
	--depth;
	//
	const nbl_glsl_sampling_quantized3D quant = texelFetch(quantizedSampleSequence,int(_sample)*SAMPLE_SEQUENCE_STRIDE+depth).xy;
    return nbl_glsl_sampling_decodeSample3Dimensions(quant,scramble_key);
}

nbl_glsl_MC_quot_pdf_aov_t gen_sample_ray(
	out vec3 direction,
	in uvec3 scramble_key,
	in uint sampleID, in uint depth,
	in nbl_glsl_MC_precomputed_t precomp,
	in nbl_glsl_MC_instr_stream_t gcs,
	in nbl_glsl_MC_instr_stream_t rnps
)
{
	vec3 rand = rand3d(scramble_key,int(sampleID),int(depth));
	
	nbl_glsl_LightSample s;
	nbl_glsl_MC_quot_pdf_aov_t result = nbl_glsl_MC_runGenerateAndRemainderStream(precomp,gcs,rnps,rand,s);

	// russian roulette
	const uint noRussianRouletteDepth = bitfieldExtract(staticViewData.pathDepth_noRussianRouletteDepth_samplesPerPixelPerDispatch,8,8);
	if (depth>noRussianRouletteDepth)
	{
		const float rrContinuationFactor = 0.25f;
		const float survivalProb = min(nbl_glsl_MC_colorToScalar(result.quotient)/rrContinuationFactor,1.f);
		result.pdf *= survivalProb;
		float dummy; // not going to use it, because we can optimize out better
		const bool kill = nbl_glsl_partitionRandVariable(survivalProb,rand.z,dummy);
		result.quotient *= kill ? 0.f:(1.f/survivalProb);
	}

	direction = s.L;
	return result;
}

void generate_next_rays(
	in uint maxRaysToGen, in nbl_glsl_MC_oriented_material_t material, in bool frontfacing, in uint vertex_depth,
	nbl_glsl_xoroshiro64star_state_t scramble_state, in uint sampleID, in uvec2 outPixelLocation,
	in vec3 origin, in vec3 prevThroughput, in float prevAoVThroughputScale, inout vec3 albedo, out vec3 worldspaceNormal)
{
	// get material streams as well
	const nbl_glsl_MC_instr_stream_t gcs = nbl_glsl_MC_oriented_material_t_getGenChoiceStream(material);
	const nbl_glsl_MC_instr_stream_t rnps = nbl_glsl_MC_oriented_material_t_getRemAndPdfStream(material);


	// need to do this after we have worldspace V and N ready
	const nbl_glsl_MC_precomputed_t precomputed = nbl_glsl_MC_precomputeData(frontfacing);
	worldspaceNormal = precomputed.N*nbl_glsl_MC_colorToScalar(albedo);
#ifdef NORM_PRECOMP_STREAM
	const nbl_glsl_MC_instr_stream_t nps = nbl_glsl_MC_oriented_material_t_getNormalPrecompStream(material);
	nbl_glsl_MC_runNormalPrecompStream(nps,precomputed);
#endif
	
	// prepare rays
	uint raysToAllocate = 0u;
	vec3 direction[MAX_RAYS_GENERATED];
	float maxT[MAX_RAYS_GENERATED];
	vec3 nextThroughput[MAX_RAYS_GENERATED];
	float nextAoVThroughputScale[MAX_RAYS_GENERATED];
	{
		const uvec3 scramble_key = uvec3(nbl_glsl_xoroshiro64star(scramble_state),nbl_glsl_xoroshiro64star(scramble_state),nbl_glsl_xoroshiro64star(scramble_state));
		for (uint i=0u; i<maxRaysToGen; i++)
		{
			maxT[i] = 0.f;
			// TODO: When generating NEE rays, advance the dimension, NOT the sampleID
			const nbl_glsl_MC_quot_pdf_aov_t result = gen_sample_ray(direction[i],scramble_key,sampleID+i,vertex_depth,precomputed,gcs,rnps);
			albedo += result.aov.albedo/float(maxRaysToGen);
			worldspaceNormal += result.aov.normal/float(maxRaysToGen);

			nextThroughput[i] = prevThroughput*result.quotient;
			// TODO: add some sort of factor to this inequality that could account for highest possible emission (direct or indirect) we could encounter
			if (max(max(nextThroughput[i].x,nextThroughput[i].y),nextThroughput[i].z)>exp2(-19.f)) // match output mantissa (won't contribute anything afterwards)
			{
				maxT[i] = nbl_glsl_FLT_MAX;
				nextAoVThroughputScale[i] = prevAoVThroughputScale*result.aov.throughputFactor;
				raysToAllocate++;
			}
		}
	}
	// TODO: investigate workgroup reductions here
	const uint baseOutputID = atomicAdd(rayCount[pc.cummon.rayCountWriteIx],raysToAllocate);

	// the 1.03125f adjusts for the fact that the normal might be too short (inversesqrt precision)
	const float inversesqrt_precision = 1.03125f;
	// TODO: investigate why we can't use `normalizedN` here
	const vec3 ray_offset_vector = normalize(cross(dPdBary[0],dPdBary[1]))*inversesqrt_precision;
	float origin_offset = nbl_glsl_numeric_limits_float_epsilon(44u); // I pulled the constants out of my @$$
	origin_offset += dot(abs(ray_offset_vector),abs(origin))*nbl_glsl_numeric_limits_float_epsilon(32u);
	// TODO: in the future run backward error analysis of
	// dot(mat3(WorldToObj)*(origin+offset*geomNormal/length(geomNormal))+(WorldToObj-vx_pos[1]),geomNormal)
	// where
	// origin = mat3x2(vx_pos[2]-vx_pos[1],vx_pos[0]-vx_pos[1])*barys+vx_pos[1]
	// geonNormal = cross(vx_pos[2]-vx_pos[1],vx_pos[0]-vx_pos[1])
	// and we assume only `WorldToObj`, `vx_pos[i]` and `barys` are accurate values. So far:
	// offset > (1+gamma(2))/(1-gamma(2))*(dot(abs(geomNormal),omega_error)+dot(abs(omega),geomNormal_error)+dot(omega_error,geomNormal_error))
	//const vec3 geomNormal = cross(dPdBary[0],dPdBary[1]);
	//float ray_offset = ?;
	//ray_offset = nbl_glsl_ieee754_next_ulp_away_from_zero(ray_offset);
	const vec3 ray_offset = ray_offset_vector*origin_offset;
	const vec3 ray_origin[2] = {origin+ray_offset,origin-ray_offset};
	uint offset = 0u;
	for (uint i=0u; i<maxRaysToGen; i++)
	if (maxT[i]!=0.f)
	{
		nbl_glsl_ext_RadeonRays_ray newRay;
		if (dot(ray_offset_vector,direction[i])<0.f)
			newRay.origin = ray_origin[1];
		else
			newRay.origin = ray_origin[0];
		newRay.maxT = maxT[i];
		newRay.direction = direction[i];
		packOutPixelLocationAndAoVThroughputFactor(newRay.time,outPixelLocation,nextAoVThroughputScale[i]);
		newRay.mask = int(scramble_state[0]);
		newRay._active = int(scramble_state[1]);
		newRay.useless_padding[0] = packHalf2x16(nextThroughput[i].rg);
		newRay.useless_padding[1] = bitfieldInsert(packHalf2x16(nextThroughput[i].bb),sampleID+i,16,16);
		const uint outputID = baseOutputID+(offset++);
		sinkRays[outputID] = newRay;
	}
}


struct Contribution
{
	vec3 color;
	vec3 albedo;
	vec3 worldspaceNormal;
};

vec2 SampleSphericalMap(vec3 v)
{
    vec2 uv = vec2(atan(v.z,v.x),acos(v.y));
    uv.x *= nbl_glsl_RECIPROCAL_PI*0.5;
    uv.x += 0.25; 
    uv.y *= nbl_glsl_RECIPROCAL_PI;
    return uv;
}

void Contribution_initMiss(out Contribution contrib, in float aovThroughputScale)
{
	vec2 uv = SampleSphericalMap(-normalizedV);
	// funny little trick borrowed from things like Progressive Photon Mapping
	const float bias = 0.0625f*(1.f-aovThroughputScale)*pow(pc.cummon.rcpFramesDispatched,0.08f);
	contrib.albedo = contrib.color = textureGrad(envMap, uv, vec2(bias*0.5,0.f), vec2(0.f,bias)).rgb;
	contrib.worldspaceNormal = normalizedV;
}

void Contribution_normalizeAoV(inout Contribution contrib)
{
	// could do some other normalization factor, whats important is that albedo looks somewhat like the HDR value, albeit scaled down
	contrib.albedo = contrib.albedo/max(max(contrib.albedo.r,contrib.albedo.g),max(contrib.albedo.b,1.f));
	contrib.worldspaceNormal *= inversesqrt(max(dot(contrib.worldspaceNormal,contrib.worldspaceNormal),1.f));
}
#endif