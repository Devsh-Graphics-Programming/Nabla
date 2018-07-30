#version 430 core
layout(std140, binding = 0) uniform U { mat4 MVP; };
//uniform mat4 MVP;

layout(location = 0) in vec3 vPos; //only a 3d position is passed from irrlicht, but last (the W) coordinate gets filled with default 1.0
layout(location = 2) in vec2 vTC;
layout(location = 3) in vec3 vNormal;
layout(location = 5) in ivec4 vBoneIDs;
layout(location = 6) in vec4 vBoneWeights;

#define BONE_CNT 46
#define INSTANCE_CNT 100
layout(std430, binding = 0) readonly buffer InstData
{
	//boneCnt*instanceCnt == 46*100
	float gmat_00[4600];
	float gmat_01[4600];
	float gmat_02[4600];
	float gmat_03[4600];
	float gmat_10[4600];
	float gmat_11[4600];
	float gmat_12[4600];
	float gmat_13[4600];
	float gmat_20[4600];
	float gmat_21[4600];
	float gmat_22[4600];
	float gmat_23[4600];
	
	float nmat_00[4600];
	float nmat_01[4600];
	float nmat_02[4600];
	float nmat_10[4600];
	float nmat_11[4600];
	float nmat_12[4600];
	float nmat_20[4600];
	float nmat_21[4600];
	float nmat_22[4600];
};

out vec3 Normal;
out vec2 TexCoord;
out vec3 lightDir;

void fetchMatrices(out mat4x3 _g, out mat3 _n, in int _off)
{
	_g = mat4x3(
		gmat_00[BONE_CNT*gl_InstanceID + _off],
		gmat_01[BONE_CNT*gl_InstanceID + _off],
		gmat_02[BONE_CNT*gl_InstanceID + _off],
		gmat_03[BONE_CNT*gl_InstanceID + _off],
		gmat_10[BONE_CNT*gl_InstanceID + _off],
		gmat_11[BONE_CNT*gl_InstanceID + _off],
		gmat_12[BONE_CNT*gl_InstanceID + _off],
		gmat_13[BONE_CNT*gl_InstanceID + _off],
		gmat_20[BONE_CNT*gl_InstanceID + _off],
		gmat_21[BONE_CNT*gl_InstanceID + _off],
		gmat_22[BONE_CNT*gl_InstanceID + _off],
		gmat_23[BONE_CNT*gl_InstanceID + _off]
	);
	_n = mat3(
		nmat_00[BONE_CNT*gl_InstanceID + _off],
		nmat_01[BONE_CNT*gl_InstanceID + _off],
		nmat_02[BONE_CNT*gl_InstanceID + _off],
		nmat_10[BONE_CNT*gl_InstanceID + _off],
		nmat_11[BONE_CNT*gl_InstanceID + _off],
		nmat_12[BONE_CNT*gl_InstanceID + _off],
		nmat_20[BONE_CNT*gl_InstanceID + _off],
		nmat_21[BONE_CNT*gl_InstanceID + _off],
		nmat_22[BONE_CNT*gl_InstanceID + _off]
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

    gl_Position = MVP*vec4(pos,1.0); //only thing preventing the shader from being core-compliant
    Normal = normalize(nml); //have to normalize twice because of normal quantization
    lightDir = vec3(100.0,0.0,0.0)-vPos.xyz;
    TexCoord = vTC;
}
