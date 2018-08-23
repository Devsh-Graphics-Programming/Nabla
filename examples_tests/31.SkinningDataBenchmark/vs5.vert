#version 430 core
//layout(std140, binding = 0) uniform U { mat4 MVP; };
uniform mat4 MVP;

layout(location = 0) in vec3 vPos;
layout(location = 3) in vec3 vNormal;
layout(location = 5) in ivec4 vBoneIDs;
layout(location = 6) in vec4 vBoneWeights;

#define BONE_CNT 46
#define INSTANCE_CNT 100
struct PerInstanceData
{
	float gmat_00[BONE_CNT];
	float gmat_01[BONE_CNT];
	float gmat_02[BONE_CNT];
	float gmat_03[BONE_CNT];
	float gmat_10[BONE_CNT];
	float gmat_11[BONE_CNT];
	float gmat_12[BONE_CNT];
	float gmat_13[BONE_CNT];
	float gmat_20[BONE_CNT];
	float gmat_21[BONE_CNT];
	float gmat_22[BONE_CNT];
	float gmat_23[BONE_CNT];
	
	float nmat_00[BONE_CNT];
	float nmat_01[BONE_CNT];
	float nmat_02[BONE_CNT];
	float nmat_10[BONE_CNT];
	float nmat_11[BONE_CNT];
	float nmat_12[BONE_CNT];
	float nmat_20[BONE_CNT];
	float nmat_21[BONE_CNT];
	float nmat_22[BONE_CNT];
};
layout(std430, binding = 0) readonly buffer InstData
{
    PerInstanceData data[INSTANCE_CNT];
};

out vec3 Normal;

void fetchMatrices(out mat4x3 _g, out mat3 _n, in int _off)
{
	_g = mat4x3(
		data[gl_InstanceID].gmat_00[_off],
		data[gl_InstanceID].gmat_01[_off],
		data[gl_InstanceID].gmat_02[_off],
		data[gl_InstanceID].gmat_03[_off],
		data[gl_InstanceID].gmat_10[_off],
		data[gl_InstanceID].gmat_11[_off],
		data[gl_InstanceID].gmat_12[_off],
		data[gl_InstanceID].gmat_13[_off],
		data[gl_InstanceID].gmat_20[_off],
		data[gl_InstanceID].gmat_21[_off],
		data[gl_InstanceID].gmat_22[_off],
		data[gl_InstanceID].gmat_23[_off]
	);
	_n = mat3(
		data[gl_InstanceID].nmat_00[_off],
		data[gl_InstanceID].nmat_01[_off],
		data[gl_InstanceID].nmat_02[_off],
		data[gl_InstanceID].nmat_10[_off],
		data[gl_InstanceID].nmat_11[_off],
		data[gl_InstanceID].nmat_12[_off],
		data[gl_InstanceID].nmat_20[_off],
		data[gl_InstanceID].nmat_21[_off],
		data[gl_InstanceID].nmat_22[_off]
	);
}


void linearSkin(out vec3 skinnedPos, out vec3 skinnedNormal, in ivec4 boneIDs, in vec4 boneWeightsXYZBoneCountNormalized)
{
    //adding transformed weighted vertices is better than adding weighted matrices and then transforming
    //averaging matrices            = [1,4]*(21 fmads) + 15 fmads
    //averaging transformed verts   = [1,4]*(15 fmads + 7 muls)
	int off = boneIDs.x;
	mat4x3 g;
	mat3 n;
	fetchMatrices(g, n, off);
	
    skinnedPos = g*vec4(vPos*boneWeightsXYZBoneCountNormalized.x,boneWeightsXYZBoneCountNormalized.x);
    skinnedNormal = n*(vNormal*boneWeightsXYZBoneCountNormalized.x);
	// we have a choice, either branch and waste 8 cycles on if-statements, or waste 84 on zeroing out the arrays
    if (boneWeightsXYZBoneCountNormalized.w>0.25) //0.33333
    {
		off = boneIDs.y;
		fetchMatrices(g, n, off);
		
        skinnedPos += g*vec4(vPos*vBoneWeights.y,vBoneWeights.y);
        skinnedNormal += n*(vNormal*vBoneWeights.y);
    }
    if (boneWeightsXYZBoneCountNormalized.w>0.5) //0.666666
    {
		off = boneIDs.z;
		fetchMatrices(g, n, off);
		
        skinnedPos += g*vec4(vPos*vBoneWeights.z,vBoneWeights.z);
        skinnedNormal += n*(vNormal*vBoneWeights.z);
    }
    if (boneWeightsXYZBoneCountNormalized.w>0.75) //1.0
    {
		off = boneIDs.w;
		fetchMatrices(g, n, off);
		
        float lastWeight = 1.0-boneWeightsXYZBoneCountNormalized.x-boneWeightsXYZBoneCountNormalized.y-boneWeightsXYZBoneCountNormalized.z;
        skinnedPos += g*vec4(vPos*lastWeight,lastWeight);
        skinnedNormal += n*(vNormal*lastWeight);
    }
}

void main()
{
    vec3 pos,nml;
    linearSkin(pos,nml,vBoneIDs,vBoneWeights);

    gl_Position = MVP*vec4(pos,1.0);
    Normal = normalize(nml); //have to normalize twice because of normal quantization
}
