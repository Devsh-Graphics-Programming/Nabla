#version 430 core
layout(std140, binding = 0) uniform U { mat4 MVP; };
//uniform mat4 MVP;

layout(location = 0) in vec3 vPos; //only a 3d position is passed from irrlicht, but last (the W) coordinate gets filled with default 1.0
layout(location = 2) in vec2 vTC;
layout(location = 3) in vec3 vNormal;
layout(location = 5) in ivec4 vBoneIDs;
layout(location = 6) in vec4 vBoneWeights;

#define BONE_CNT 46
#define INST_CNT 100
layout(std430, binding = 0) readonly buffer InstData
{
	struct {
		mat4x3 g[BONE_CNT];
		mat3 n[BONE_CNT];
	} mat[INST_CNT];
};

out vec3 Normal;
out vec2 TexCoord;
out vec3 lightDir;


void linearSkin(out vec3 skinnedPos, out vec3 skinnedNormal, in ivec4 boneIDs, in vec4 boneWeightsXYZBoneCountNormalized)
{
    //adding transformed weighted vertices is better than adding weighted matrices and then transforming
    //averaging matrices            = [1,4]*(21 fmads) + 15 fmads
    //averaging transformed verts   = [1,4]*(15 fmads + 7 muls)
    skinnedPos = mat[gl_InstanceID].g[boneIDs.x] * vec4(vPos*boneWeightsXYZBoneCountNormalized.x,boneWeightsXYZBoneCountNormalized.x);
    skinnedNormal = mat[gl_InstanceID].n[boneIDs.x] * (vNormal*boneWeightsXYZBoneCountNormalized.x);
	// we have a choice, either branch and waste 8 cycles on if-statements, or waste 84 on zeroing out the arrays
    if (boneWeightsXYZBoneCountNormalized.w>0.25) //0.33333
    {
        skinnedPos += mat[gl_InstanceID].g[boneIDs.y] * vec4(vPos*vBoneWeights.y,vBoneWeights.y);
        skinnedNormal += mat[gl_InstanceID].n[boneIDs.y] * (vNormal*vBoneWeights.y);
    }
    if (boneWeightsXYZBoneCountNormalized.w>0.5) //0.666666
    {
        skinnedPos += mat[gl_InstanceID].g[boneIDs.z] * vec4(vPos*vBoneWeights.z,vBoneWeights.z);
        skinnedNormal += mat[gl_InstanceID].n[boneIDs.z] * (vNormal*vBoneWeights.z);
    }
    if (boneWeightsXYZBoneCountNormalized.w>0.75) //1.0
    {
        float lastWeight = 1.0-boneWeightsXYZBoneCountNormalized.x-boneWeightsXYZBoneCountNormalized.y-boneWeightsXYZBoneCountNormalized.z;
        skinnedPos += mat[gl_InstanceID].g[boneIDs.w] * vec4(vPos*lastWeight,lastWeight);
        skinnedNormal += mat[gl_InstanceID].n[boneIDs.w] * (vNormal*lastWeight);
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
