#version 430 core

layout(std140, binding = 0) uniform U { mat4 MVP; };

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
	vec4 gmtx_0[4600];
	vec4 gmtx_1[4600];
	vec4 gmtx_2[4600];
	
	vec3 nmtx_0[4600];
	vec3 nmtx_1[4600];
	vec3 nmtx_2[4600];
};

out vec3 Normal;
out vec2 TexCoord;
out vec3 lightDir;


void linearSkin(out vec3 skinnedPos, out vec3 skinnedNormal, in ivec4 boneIDs, in vec4 boneWeightsXYZBoneCountNormalized)
{
    //adding transformed weighted vertices is better than adding weighted matrices and then transforming
    //averaging matrices            = [1,4]*(21 fmads) + 15 fmads
    //averaging transformed verts   = [1,4]*(15 fmads + 7 muls)
	int off = boneIDs.x;
	mat4x3 g = mat4x3(
		gmtx_0[BONE_CNT*gl_InstanceID+off],
		gmtx_1[BONE_CNT*gl_InstanceID+off],
		gmtx_2[BONE_CNT*gl_InstanceID+off]
	);
	mat3 n = mat3(
		nmtx_0[BONE_CNT*gl_InstanceID+off],
		nmtx_1[BONE_CNT*gl_InstanceID+off],
		nmtx_2[BONE_CNT*gl_InstanceID+off]
	);
	
    skinnedPos = g*vec4(vPos*boneWeightsXYZBoneCountNormalized.x,boneWeightsXYZBoneCountNormalized.x);
    skinnedNormal = n*(vNormal*boneWeightsXYZBoneCountNormalized.x);
	// we have a choice, either branch and waste 8 cycles on if-statements, or waste 84 on zeroing out the arrays
    if (boneWeightsXYZBoneCountNormalized.w>0.25) //0.33333
    {
		off = boneIDs.y;
		g = mat4x3(
			gmtx_0[BONE_CNT*gl_InstanceID+off],
			gmtx_1[BONE_CNT*gl_InstanceID+off],
			gmtx_2[BONE_CNT*gl_InstanceID+off]
		);
		n = mat3(
			nmtx_0[BONE_CNT*gl_InstanceID+off],
			nmtx_1[BONE_CNT*gl_InstanceID+off],
			nmtx_2[BONE_CNT*gl_InstanceID+off]
		);
	
        skinnedPos += g*vec4(vPos*vBoneWeights.y,vBoneWeights.y);
        skinnedNormal += n*(vNormal*vBoneWeights.y);
    }
    if (boneWeightsXYZBoneCountNormalized.w>0.5) //0.666666
    {
		off = boneIDs.z;
		g = mat4x3(
			gmtx_0[BONE_CNT*gl_InstanceID+off],
			gmtx_1[BONE_CNT*gl_InstanceID+off],
			gmtx_2[BONE_CNT*gl_InstanceID+off]
		);
		n = mat3(
			nmtx_0[BONE_CNT*gl_InstanceID+off],
			nmtx_1[BONE_CNT*gl_InstanceID+off],
			nmtx_2[BONE_CNT*gl_InstanceID+off]
		);
		
        skinnedPos += g*vec4(vPos*vBoneWeights.z,vBoneWeights.z);
        skinnedNormal += n*(vNormal*vBoneWeights.z);
    }
    if (boneWeightsXYZBoneCountNormalized.w>0.75) //1.0
    {
		off = boneIDs.w;
		g = mat4x3(
			gmtx_0[BONE_CNT*gl_InstanceID+off],
			gmtx_1[BONE_CNT*gl_InstanceID+off],
			gmtx_2[BONE_CNT*gl_InstanceID+off]
		);
		n = mat3(
			nmtx_0[BONE_CNT*gl_InstanceID+off],
			nmtx_1[BONE_CNT*gl_InstanceID+off],
			nmtx_2[BONE_CNT*gl_InstanceID+off]
		);
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
