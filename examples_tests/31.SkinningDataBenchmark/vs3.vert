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
	mat4x3 gmat[4600]; //boneCnt*instanceCnt == 46*100
	mat3 nmat[4600];
};

out vec3 Normal;
out vec2 TexCoord;
out vec3 lightDir;


void linearSkin(out vec3 skinnedPos, out vec3 skinnedNormal, in ivec4 boneIDs, in vec4 boneWeightsXYZBoneCountNormalized)
{
    //adding transformed weighted vertices is better than adding weighted matrices and then transforming
    //averaging matrices            = [1,4]*(21 fmads) + 15 fmads
    //averaging transformed verts   = [1,4]*(15 fmads + 7 muls)
    skinnedPos = gmat[BONE_CNT*gl_IntanceID + boneIDs.x]*vec4(vPos*boneWeightsXYZBoneCountNormalized.x,boneWeightsXYZBoneCountNormalized.x);
    skinnedNormal = nmat[BONE_CNT*gl_InstanceID + boneIDs.x]*(vNormal*boneWeightsXYZBoneCountNormalized.x);
	// we have a choice, either branch and waste 8 cycles on if-statements, or waste 84 on zeroing out the arrays
    if (boneWeightsXYZBoneCountNormalized.w>0.25) //0.33333
    {
        skinnedPos += gmat[BONE_CNT*gl_IntanceID + boneIDs.y]*vec4(vPos*vBoneWeights.y,vBoneWeights.y);
        skinnedNormal += nmat[BONE_CNT*gl_InstanceID + boneIDs.y]*(vNormal*vBoneWeights.y);
    }
    if (boneWeightsXYZBoneCountNormalized.w>0.5) //0.666666
    {
        skinnedPos += gmat[BONE_CNT*gl_IntanceID + boneIDs.z]*vec4(vPos*vBoneWeights.z,vBoneWeights.z);
        skinnedNormal += nmat[BONE_CNT*gl_InstanceID + boneIDs.z]*(vNormal*vBoneWeights.z);
    }
    if (boneWeightsXYZBoneCountNormalized.w>0.75) //1.0
    {
        float lastWeight = 1.0-boneWeightsXYZBoneCountNormalized.x-boneWeightsXYZBoneCountNormalized.y-boneWeightsXYZBoneCountNormalized.z;
        skinnedPos += gmat[BONE_CNT*gl_IntanceID + boneIDs.w]*vec4(vPos*lastWeight,lastWeight);
        skinnedNormal += nmat[BONE_CNT*gl_InstanceID + boneIDs.w]*(vNormal*lastWeight);
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
