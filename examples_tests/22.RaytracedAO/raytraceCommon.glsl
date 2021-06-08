#ifndef _RAYTRACE_COMMON_GLSL_INCLUDED_
#define _RAYTRACE_COMMON_GLSL_INCLUDED_

#include "virtualGeometry.glsl"


layout(push_constant, row_major) uniform PushConstants
{
	RaytraceShaderCommonData_t cummon;
} pc;


// lights
layout(set = 1, binding = 4, std430) restrict readonly buffer CumulativeLightPDF
{
	uint lightCDF[];
};
layout(set = 1, binding = 5, std430, row_major) restrict readonly buffer Lights
{
	SLight light[];
};
layout(set = 1, binding = 6, std430, row_major) restrict readonly buffer LightRadiances
{
	uvec2 lightRadiance[]; // Watts / steriadian / steradian in rgb19e7
};


layout(set = 2, binding = 0, row_major) uniform StaticViewData
{
	StaticViewData_t staticViewData;
};
layout(set = 2, binding = 1) readonly restrict buffer ExtraBatchData
{
	uint firstIndex[];
} extraBatchData;
// rng
layout(set = 2, binding = 2) uniform usamplerBuffer sampleSequence;
// accumulation
layout(set = 2, binding = 3, rg32ui) restrict uniform uimage2DArray accumulation;
// ray data
#include <nbl/builtin/glsl/ext/RadeonRays/ray.glsl>
layout(set = 2, binding = 4, std430) restrict buffer Rays
{
	nbl_glsl_ext_RadeonRays_ray data[];
} rays[2];
#include <nbl/builtin/glsl/utils/indirect_commands.glsl>
layout(set = 2, binding = 5) restrict coherent buffer RayCount
{
	uint rayCount;
	nbl_glsl_DispatchIndirectCommand_t params;
} traceIndirect[2];

//
uvec3 get_triangle_indices(in uint batchInstanceGUID, in uint triangleID)
{
	const uint baseTriangleVertex = triangleID*3u+extraBatchData.firstIndex[batchInstanceGUID];
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

bool record_emission_common(out vec3 acc, in uvec3 accumulationLocation, vec3 emissive, in bool later_path_vertices)
{	
	acc = vec3(0.0);
	const bool notFirstFrame = pc.cummon.rcpFramesDispatched!=1.f;
	const bool needToReadAccumulation = later_path_vertices||notFirstFrame;
	if (needToReadAccumulation)
		acc = fetchAccumulation(accumulationLocation);

	if (!later_path_vertices)
		emissive -= acc;
	emissive *= pc.cummon.rcpFramesDispatched;
	
	const bool anyChange = any(greaterThan(abs(emissive),vec3(FLT_MIN)));
	acc += emissive;
	return anyChange || !needToReadAccumulation;
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

#include "bin/material_declarations.glsl"
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


vec3 rand3d(inout nbl_glsl_xoroshiro64star_state_t scramble_state, in uint _sample)
{
	uvec3 seqVal = texelFetch(sampleSequence,int(_sample)).xyz;
	seqVal ^= uvec3(nbl_glsl_xoroshiro64star(scramble_state),nbl_glsl_xoroshiro64star(scramble_state),nbl_glsl_xoroshiro64star(scramble_state));
    return vec3(seqVal)*uintBitsToFloat(0x2f800004u);
}

void gen_sample_ray(
	out float maxT, out vec3 direction, out vec3 throughput,
	inout nbl_glsl_xoroshiro64star_state_t scramble_state, in uint sampleID,
	in nbl_glsl_MC_precomputed_t precomp, in nbl_glsl_MC_instr_stream_t gcs, in nbl_glsl_MC_instr_stream_t rnps
)
{
	maxT = FLT_MAX;
	
	vec3 rand = rand3d(scramble_state,sampleID);
	
	float pdf;
	nbl_glsl_LightSample s;
	throughput = nbl_glsl_MC_runGenerateAndRemainderStream(precomp, gcs, rnps, rand, pdf, s);

	direction = s.L;
}

#endif