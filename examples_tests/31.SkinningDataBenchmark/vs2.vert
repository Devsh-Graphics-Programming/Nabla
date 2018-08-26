#version 430 core
//layout(std140, binding = 0) uniform U { mat4 MVP; };
uniform mat4 MVP;

layout(location = 0) in vec3 vPos;
layout(location = 3) in vec3 vNormal;
layout(location = 5) in ivec4 vBoneIDs;
layout(location = 6) in vec4 vBoneWeights;

#define BONE_CNT 46
#define INST_CNT 100
struct PerBoneData
{
    mat4x3 g;
    mat3 n;
};
struct PerInstData
{
    PerBoneData data[BONE_CNT];
};
layout(std430, binding = 0) readonly buffer InstData
{
	PerInstData mat[INST_CNT];
};

out vec3 Normal;

void linearSkin(out vec3 skinnedPos, out vec3 skinnedNormal, in ivec4 boneIDs, in vec4 boneWeightsXYZBoneCountNormalized)
{
    //adding transformed weighted vertices is better than adding weighted matrices and then transforming
    //averaging matrices            = [1,4]*(21 fmads) + 15 fmads
    //averaging transformed verts   = [1,4]*(15 fmads + 7 muls)

    //skinnedPos = mat4x3(1.f) * vec4(vPos*boneWeightsXYZBoneCountNormalized.x,boneWeightsXYZBoneCountNormalized.x);
    skinnedPos = mat[gl_InstanceID].data[boneIDs.x].g * vec4(vPos*boneWeightsXYZBoneCountNormalized.x,boneWeightsXYZBoneCountNormalized.x);
    skinnedNormal = mat[gl_InstanceID].data[boneIDs.x].n * (vNormal*boneWeightsXYZBoneCountNormalized.x);
	// we have a choice, either branch and waste 8 cycles on if-statements, or waste 84 on zeroing out the arrays
    if (boneWeightsXYZBoneCountNormalized.w>0.25) //0.33333
    {
        //skinnedPos += mat4x3(1.f) * vec4(vPos*vBoneWeights.y,vBoneWeights.y);
        skinnedPos += mat[gl_InstanceID].data[boneIDs.y].g * vec4(vPos*vBoneWeights.y,vBoneWeights.y);
        skinnedNormal += mat[gl_InstanceID].data[boneIDs.y].n * (vNormal*vBoneWeights.y);
    }
    if (boneWeightsXYZBoneCountNormalized.w>0.5) //0.666666
    {
        //skinnedPos += mat4x3(1.f) * vec4(vPos*vBoneWeights.z,vBoneWeights.z);
        skinnedPos += mat[gl_InstanceID].data[boneIDs.z].g * vec4(vPos*vBoneWeights.z,vBoneWeights.z);
        skinnedNormal += mat[gl_InstanceID].data[boneIDs.z].n * (vNormal*vBoneWeights.z);
    }
    if (boneWeightsXYZBoneCountNormalized.w>0.75) //1.0
    {
        float lastWeight = 1.0-boneWeightsXYZBoneCountNormalized.x-boneWeightsXYZBoneCountNormalized.y-boneWeightsXYZBoneCountNormalized.z;
        //skinnedPos += mat4x3(1.f) * vec4(vPos*lastWeight,lastWeight);
        skinnedPos += mat[gl_InstanceID].data[boneIDs.w].g * vec4(vPos*lastWeight,lastWeight);
        skinnedNormal += mat[gl_InstanceID].data[boneIDs.w].n * (vNormal*lastWeight);
    }
}

void main()
{
    vec3 pos,nml;
    linearSkin(pos,nml,vBoneIDs,vBoneWeights);

    gl_Position = MVP*(vec4(pos,1.0) + 10.f*vec4(float(gl_InstanceID), 0.f, float(gl_InstanceID), 0.f));
    Normal = normalize(nml); //have to normalize twice because of normal quantization
}
