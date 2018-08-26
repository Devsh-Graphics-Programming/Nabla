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
	mat4x3 gmat[BONE_CNT];
	mat3 nmat[BONE_CNT];   
};
layout(std430, binding = 0) readonly buffer InstData
{
	PerInstanceData data[INSTANCE_CNT];
};

out vec3 Normal;

void linearSkin(out vec3 skinnedPos, out vec3 skinnedNormal, in ivec4 boneIDs, in vec4 boneWeightsXYZBoneCountNormalized)
{
    //adding transformed weighted vertices is better than adding weighted matrices and then transforming
    //averaging matrices            = [1,4]*(21 fmads) + 15 fmads
    //averaging transformed verts   = [1,4]*(15 fmads + 7 muls)
    skinnedPos = data[gl_InstanceID].gmat[boneIDs.x]*vec4(vPos*boneWeightsXYZBoneCountNormalized.x,boneWeightsXYZBoneCountNormalized.x);
    skinnedNormal = data[gl_InstanceID].nmat[boneIDs.x]*(vNormal*boneWeightsXYZBoneCountNormalized.x);
	// we have a choice, either branch and waste 8 cycles on if-statements, or waste 84 on zeroing out the arrays
    if (boneWeightsXYZBoneCountNormalized.w>0.25) //0.33333
    {
        skinnedPos += data[gl_InstanceID].gmat[boneIDs.y]*vec4(vPos*vBoneWeights.y,vBoneWeights.y);
        skinnedNormal += data[gl_InstanceID].nmat[boneIDs.y]*(vNormal*vBoneWeights.y);
    }
    if (boneWeightsXYZBoneCountNormalized.w>0.5) //0.666666
    {
        skinnedPos += data[gl_InstanceID].gmat[boneIDs.z]*vec4(vPos*vBoneWeights.z,vBoneWeights.z);
        skinnedNormal += data[gl_InstanceID].nmat[boneIDs.z]*(vNormal*vBoneWeights.z);
    }
    if (boneWeightsXYZBoneCountNormalized.w>0.75) //1.0
    {
        float lastWeight = 1.0-boneWeightsXYZBoneCountNormalized.x-boneWeightsXYZBoneCountNormalized.y-boneWeightsXYZBoneCountNormalized.z;
        skinnedPos += data[gl_InstanceID].gmat[boneIDs.w]*vec4(vPos*lastWeight,lastWeight);
        skinnedNormal += data[gl_InstanceID].nmat[boneIDs.w]*(vNormal*lastWeight);
    }
}

void main()
{
    vec3 pos,nml;
    linearSkin(pos,nml,vBoneIDs,vBoneWeights);

    gl_Position = MVP*(vec4(pos,1.0) + 10.f*vec4(float(gl_InstanceID), 0.f, float(gl_InstanceID), 0.f));
    Normal = normalize(nml); //have to normalize twice because of normal quantization
}
