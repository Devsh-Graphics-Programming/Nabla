#include "common.glsl"

#extension GL_KHR_shader_subgroup_ballot : require

struct BoneData
{
    mat4 boneMatrix;
    mat4x3 normalMatrix;
};

layout(std430, set = 0, binding = 0, row_major) readonly buffer BoneMatrices_struct
{
    BoneData data[];
} boneSSBO_structs;
layout(std430, set = 0, binding = 1) readonly buffer BoneMatrices_dword
{
    uint data[];
} boneSSBO_dwords;

#ifndef BENCHMARK
layout(location = 0) in vec3 pos;
layout(location = 3) in vec3 normal;
layout(location = 0) out vec3 vNormal;
#endif
layout(location = 4) in uint boneID;

#define OBJ_DWORDS 32 // sizeof(BoneData), must be PoT
struct BoneData_dword
{
    uint data[OBJ_DWORDS];
};
BoneData toBoneData(in BoneData_dword bone)
{
    BoneData retval;
    //tpose because it was loaded as row_major
    //one-liner because glslang doesnt support multiline preproc definitions
#define GET_BONE_MATRIX_COL(c) retval.boneMatrix[c].x = uintBitsToFloat(bone.data[c]);retval.boneMatrix[c].y = uintBitsToFloat(bone.data[c+4]);retval.boneMatrix[c].z = uintBitsToFloat(bone.data[c+8]);retval.boneMatrix[c].w = uintBitsToFloat(bone.data[c+12])
    
    GET_BONE_MATRIX_COL(0);
    GET_BONE_MATRIX_COL(1);
    GET_BONE_MATRIX_COL(2);
    GET_BONE_MATRIX_COL(3);

    //tpose because it was loaded as row_major
    //one-liner because glslang doesnt support multiline preproc definitions
#define GET_NORMAL_MATRIX_COL(c) retval.normalMatrix[c].x = uintBitsToFloat(bone.data[16+c]);retval.normalMatrix[c].y = uintBitsToFloat(bone.data[16+c+4]);retval.normalMatrix[c].z = uintBitsToFloat(bone.data[16+c+8])

    GET_NORMAL_MATRIX_COL(0);
    GET_NORMAL_MATRIX_COL(1);
    GET_NORMAL_MATRIX_COL(2);
    GET_NORMAL_MATRIX_COL(3);

    return retval;
}
#define COALESCING_DWORDS_LOG2 4 // GCN can fetch only 64bytes in a single request
#define SUBGROUP_THRESH 16

BoneData getBone(uint _boneID)
{
//#ifdef IRR_GL_KHR_shader_subgroup_basic_size
  // if a set of invocations are active without gaps we can do a fast path
  const uvec4 activeMask = subgroupBallot(true);
  const int incr = int(subgroupBallotBitCount(activeMask));
  const int incrLog2 = int(subgroupBallotFindMSB(activeMask));
  if ((0x1<<incrLog2)==incr && incrLog2>=COALESCING_DWORDS_LOG2) // contiguous segment of active warps is required
  {
    BoneData_dword retval;
    uint boneID = _boneID*uint(OBJ_DWORDS);

    // basically fetch bones for one target invocation at a time
    uvec2 outstandingLoadsMask = activeMask.xy;
    // maybe unroll a few times manually
    while (any(notEqual(outstandingLoadsMask,uvec2(0u))))
    {
		// more work required to make this work with gl_SubgroupSize > OBJ_DWORDS but good enough to benchmark
        uint subgroupBoneID = subgroupBroadcast(boneID,subgroupBallotFindLSB(uvec4(outstandingLoadsMask,0u,0u)));
        bool willLoadBone = subgroupBoneID==boneID;
        outstandingLoadsMask ^= subgroupBallot(willLoadBone).xy;
        
	
		uint dynamically_uniform_addr = boneID+gl_SubgroupInvocationID;
		// use all SIMD lanes to load but then only some to read from subgroup registers
		uint tmp = boneSSBO_dwords.data[dynamically_uniform_addr];
		const bool notEnoughInvocations = incrLog2<OBJ_DWORDS;

		if (willLoadBone)
		{
		  int oit=0, iit=0;
		  for (int j=0; j<SUBGROUP_THRESH; j++)
			retval.data[oit++] = subgroupBroadcast(tmp,iit++);
		}
		if (notEnoughInvocations)
		{
		  tmp = boneSSBO_dwords.data[dynamically_uniform_addr+incr];
		}
		if (willLoadBone)
		{
		  int oit=SUBGROUP_THRESH, iit=notEnoughInvocations ? SUBGROUP_THRESH:0;
		  for (int j=0; j<SUBGROUP_THRESH; j++)
			retval.data[oit++] = subgroupBroadcast(tmp,iit++);
		}
    }

    return toBoneData(retval);
  }
  else
//#endif
  return boneSSBO_structs.data[_boneID];
}

void main()
{
#ifdef BENCHMARK
    const vec3 pos = vec3(1.0, 2.0, 3.0);
    const vec3 normal = vec3(1.0, 2.0, 3.0);
#endif
    BoneData bone = getBone(boneID);
#ifndef BENCHMARK
    gl_Position = bone.boneMatrix * vec4(pos, 1.0);
    vNormal = mat3(bone.normalMatrix) * normalize(normal);
#else
    gl_Position = bone.boneMatrix * vec4(pos, 1.0);
    gl_Position.xyz += mat3(bone.normalMatrix) * normal;
#endif

}