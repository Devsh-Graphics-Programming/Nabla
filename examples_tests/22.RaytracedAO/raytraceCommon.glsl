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
layout(set = 2, binding = 1, rg32ui) uniform uimage2DArray scramblebuf;
layout(set = 2, binding = 2) uniform usamplerBuffer sampleSequence;
// accumulation
layout(set = 2, binding = 3, rg32ui) restrict uniform uimage2DArray accumulation;
// ray data
#include <nbl/builtin/glsl/ext/RadeonRays/ray.glsl>
layout(set = 2, binding = 4, std430) restrict writeonly buffer SinkRays
{
	nbl_glsl_ext_RadeonRays_ray sinkRays[];
};
#include <nbl/builtin/glsl/utils/indirect_commands.glsl>
layout(set = 2, binding = 5) restrict coherent buffer RayCount // maybe remove coherent keyword
{
	uint rayCount[RAYCOUNT_N_BUFFERING];
};

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

bool record_emission_common(out vec3 acc, in uvec3 accumulationLocation, vec3 emissive, in bool first_accumulating_path_vertex)
{
	acc = vec3(0.0);
	const bool notFirstFrame = pc.cummon.rcpFramesDispatched!=1.f;
	if (!first_accumulating_path_vertex || notFirstFrame)
		acc = fetchAccumulation(accumulationLocation);
	if (first_accumulating_path_vertex) // a bit useless to add && notFirstFrame) its a tautology with acc=vec3(0.0)
		emissive -= acc;
	emissive *= pc.cummon.rcpFramesDispatched;
	
	const bool anyChange = any(greaterThan(abs(emissive),vec3(nbl_glsl_FLT_MIN)));
	acc += emissive;
	return anyChange;
}



float packOutPixelLocation(in uvec2 outPixelLocation)
{
	return uintBitsToFloat(bitfieldInsert(outPixelLocation.x,outPixelLocation.y,16,16));
}
uvec2 unpackOutPixelLocation(in float packed)
{
	const uint asUint = floatBitsToUint(packed);
	return uvec2(asUint&0xffffu,asUint>>16u);
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

#include <nbl/builtin/glsl/barycentric/utils.glsl>
mat2x3 dPdBary;
vec3 load_positions(in uvec3 indices, in nbl_glsl_ext_Mitsuba_Loader_instance_data_t batchInstanceData)
{
	mat3 positions = mat3(
		nbl_glsl_fetchVtxPos(indices[0],batchInstanceData),
		nbl_glsl_fetchVtxPos(indices[1],batchInstanceData),
		nbl_glsl_fetchVtxPos(indices[2],batchInstanceData)
	);
	const mat4x3 tform = batchInstanceData.tform;
	positions = mat3(tform)*positions;
	//
	for (int i=0; i<2; i++)
		dPdBary[i] = positions[i]-positions[2];
	return positions[2]+tform[3];
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

nbl_glsl_xoroshiro64star_state_t load_aux_vertex_attrs(
	in vec2 compactBary, in uvec3 indices, in nbl_glsl_ext_Mitsuba_Loader_instance_data_t batchInstanceData,
	in nbl_glsl_MC_oriented_material_t material,
	in uvec2 outPixelLocation, in uint vertex_depth_mod_2
#ifdef TEX_PREFETCH_STREAM
	,in mat2 dBarydScreen
#endif
)
{
	// if we ever support spatially varying emissive, we'll need to hoist barycentric computation and UV fetching to the position fetching
	#ifdef TEX_PREFETCH_STREAM
	const mat3x2 uvs = mat3x2(
		nbl_glsl_fetchVtxUV(indices[0],batchInstanceData),
		nbl_glsl_fetchVtxUV(indices[1],batchInstanceData),
		nbl_glsl_fetchVtxUV(indices[2],batchInstanceData)
	);
	const nbl_glsl_MC_instr_stream_t tps = nbl_glsl_MC_oriented_material_t_getTexPrefetchStream(material);
	#endif
	// only needed for continuing
	const mat3 normals = mat3(
		nbl_glsl_fetchVtxNormal(indices[0],batchInstanceData),
		nbl_glsl_fetchVtxNormal(indices[1],batchInstanceData),
		nbl_glsl_fetchVtxNormal(indices[2],batchInstanceData)
	);

	#ifdef TEX_PREFETCH_STREAM
	dUVdBary = mat2(uvs[0]-uvs[2],uvs[1]-uvs[2]);
	const vec2 UV = dUVdBary*compactBary+uvs[2];
	const mat2 dUVdScreen = nbl_glsl_applyChainRule2D(dUVdBary,dBarydScreen);
	nbl_glsl_MC_runTexPrefetchStream(tps,UV,dUVdScreen);
	#endif
	// not needed for NEE unless doing Area or Projected Solid Angle Sampling
	const vec3 normal = normals*nbl_glsl_barycentric_expand(compactBary);

	// init scramble while waiting for getting the instance's normal matrix
	const nbl_glsl_xoroshiro64star_state_t scramble_start_state = imageLoad(scramblebuf,ivec3(outPixelLocation,1u/*vertex_depth_mod_2*/)).rg;

	// while waiting for the scramble state
	normalizedN.x = dot(batchInstanceData.normalMatrixRow0,normal);
	normalizedN.y = dot(batchInstanceData.normalMatrixRow1,normal);
	normalizedN.z = dot(batchInstanceData.normalMatrixRow2,normal);
	normalizedN = normalize(normalizedN);

	return scramble_start_state;
}

vec3 rand3d(inout nbl_glsl_xoroshiro64star_state_t scramble_state, in int _sample, in int depth)
{
	uvec3 seqVal = texelFetch(sampleSequence,int(_sample)+(depth-1)*MAX_ACCUMULATED_SAMPLES).xyz;
	seqVal ^= uvec3(nbl_glsl_xoroshiro64star(scramble_state),nbl_glsl_xoroshiro64star(scramble_state),nbl_glsl_xoroshiro64star(scramble_state));
    return vec3(seqVal)*uintBitsToFloat(0x2f800004u);
}

void gen_sample_ray(
	out float maxT, out vec3 direction, out vec3 throughput,
	inout nbl_glsl_xoroshiro64star_state_t scramble_state, in uint sampleID, in uint depth,
	in nbl_glsl_MC_precomputed_t precomp, in nbl_glsl_MC_instr_stream_t gcs, in nbl_glsl_MC_instr_stream_t rnps
)
{
	maxT = nbl_glsl_FLT_MAX;
	
	vec3 rand = rand3d(scramble_state,int(sampleID),int(depth));
	
	float pdf;
	nbl_glsl_LightSample s;
	throughput = nbl_glsl_MC_runGenerateAndRemainderStream(precomp,gcs,rnps,rand,pdf,s);

	direction = s.L;
}


void generate_next_rays(
	in uint maxRaysToGen, in nbl_glsl_MC_oriented_material_t material, in bool frontfacing, in uint vertex_depth,
	in nbl_glsl_xoroshiro64star_state_t scramble_start_state, in uint sampleID, in uvec2 outPixelLocation,
	in vec3 origin, in vec3 prevThroughput)
{
	// get material streams as well
	const nbl_glsl_MC_instr_stream_t gcs = nbl_glsl_MC_oriented_material_t_getGenChoiceStream(material);
	const nbl_glsl_MC_instr_stream_t rnps = nbl_glsl_MC_oriented_material_t_getRemAndPdfStream(material);


	// need to do this after we have worldspace V and N ready
	const nbl_glsl_MC_precomputed_t precomputed = nbl_glsl_MC_precomputeData(frontfacing);
#ifdef NORM_PRECOMP_STREAM
	const nbl_glsl_MC_instr_stream_t nps = nbl_glsl_MC_oriented_material_t_getNormalPrecompStream(material);
	nbl_glsl_MC_runNormalPrecompStream(nps,precomputed);
#endif
	
	const uint vertex_depth_mod_2 = vertex_depth&0x1u;
	const uint vertex_depth_mod_2_inv = vertex_depth_mod_2^0x1u;
	// prepare rays
	uint raysToAllocate = 0u;
	float maxT[MAX_RAYS_GENERATED]; vec3 direction[MAX_RAYS_GENERATED]; vec3 nextThroughput[MAX_RAYS_GENERATED];	
for (uint i=1u; i!=vertex_depth; i++)
{
	nbl_glsl_xoroshiro64star(scramble_start_state);
	nbl_glsl_xoroshiro64star(scramble_start_state);
	nbl_glsl_xoroshiro64star(scramble_start_state);
}
	for (uint i=0u; i<maxRaysToGen; i++)
	{
		nbl_glsl_xoroshiro64star_state_t scramble_state = scramble_start_state;
		// TODO: When generating NEE rays, advance the dimension, NOT the sampleID
		gen_sample_ray(maxT[i],direction[i],nextThroughput[i],scramble_state,sampleID+i,vertex_depth,precomputed,gcs,rnps);
// TODO: bad idea, invent something else
//		if (i==0u)
//			imageStore(scramblebuf,ivec3(outPixelLocation,vertex_depth_mod_2_inv),uvec4(scramble_state,0u,0u));
		nextThroughput[i] *= prevThroughput;
		if (max(max(nextThroughput[i].x,nextThroughput[i].y),nextThroughput[i].z)>exp2(-19.f)) // TODO: reverse tonemap to adjust the threshold
			raysToAllocate++;
		else
			maxT[i] = 0.f;
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
		newRay.time = packOutPixelLocation(outPixelLocation);
		newRay.mask = -1;
		newRay._active = 1;
		newRay.useless_padding[0] = packHalf2x16(nextThroughput[i].rg);
		newRay.useless_padding[1] = bitfieldInsert(packHalf2x16(nextThroughput[i].bb),sampleID+i,16,16);
		const uint outputID = baseOutputID+(offset++);
		sinkRays[outputID] = newRay;
	}
}

/* TODO: optimize and reorganize
void main()
{
	clear_raycount();
	const bool alive = useful_invocation();
	uint raysToAllocate = 0u;
	vec3 emissive;
	if (alive)
	{
		emissive = staticViewData.envmapBaseColor;

		raysToAllocate = main_prolog(emissive,...);
	}

	const uint raysLocalEnd = nbl_glsl_workgroupInclusiveAdd(raysToAllocate);
	uint baseOutputID;
	if (gl_LocalInvocationIndex==WORKGROUP_SIZE-1)
		baseOutputID = atomicAdd(rayCount[pc.cummon.rayCountWriteIx],raysLocalEnd);
	baseOutputID = nbl_glsl_workgroupBroadcast(baseOutputID,WORKGROUP_SIZE-1);

	// coalesce rays
	for ()
	{
	}
	// write them out to global mem
	for ()
	{
	}

	if (alive)
	{
		// store accumulation
		main_epilog();
	}
}
*/
#endif