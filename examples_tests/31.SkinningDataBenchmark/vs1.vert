#version 430 core
layout(binding = 3) uniform samplerBuffer boneTBO;
//layout(std140, binding = 0) uniform U { mat4 MVP; };
uniform mat4 MVP;

layout(location = 0) in vec3 vPos;
layout(location = 3) in vec3 vNormal;
layout(location = 5) in ivec4 vBoneIDs;
layout(location = 6) in vec4 vBoneWeights;

out vec3 Normal;

#define INST_CNT 100
#define BONE_CNT 46

void linearSkin(out vec3 skinnedPos, out vec3 skinnedNormal, in ivec4 boneIDs, in vec4 boneWeightsXYZBoneCountNormalized)
{
    vec4 boneData[20]; // some vendors don't initialize these arrays to 0 after they've already been written to
    float lastBoneData[4];  // some vendors don't initialize these arrays to 0 after they've already been written to

    int boneOffset = gl_InstanceID*BONE_CNT*6 + boneIDs.x*6;
    //global matrix
    boneData[0] = texelFetch(boneTBO,boneOffset);
    boneData[1] = texelFetch(boneTBO,boneOffset+1);
    boneData[2] = texelFetch(boneTBO,boneOffset+2);
    //normal matrix
    boneData[3] = texelFetch(boneTBO,boneOffset+3);
    boneData[4] = texelFetch(boneTBO,boneOffset+4);
    lastBoneData[0] = texelFetch(boneTBO,boneOffset+5).x;

    if (boneWeightsXYZBoneCountNormalized.w>0.25) //0.33333
    {
        boneOffset = gl_InstanceID*BONE_CNT*6 + boneIDs.y*6;
        //global matrix
        boneData[5+0] = texelFetch(boneTBO,boneOffset);
        boneData[5+1] = texelFetch(boneTBO,boneOffset+1);
        boneData[5+2] = texelFetch(boneTBO,boneOffset+2);
        //normal matrix
        boneData[5+3] = texelFetch(boneTBO,boneOffset+3);
        boneData[5+4] = texelFetch(boneTBO,boneOffset+4);
        lastBoneData[1] = texelFetch(boneTBO,boneOffset+5).x;
    }
    if (boneWeightsXYZBoneCountNormalized.w>0.5) //0.666666
    {
        boneOffset = gl_InstanceID*BONE_CNT*6 + boneIDs.z*6;
        //global matrix
        boneData[10+0] = texelFetch(boneTBO,boneOffset);
        boneData[10+1] = texelFetch(boneTBO,boneOffset+1);
        boneData[10+2] = texelFetch(boneTBO,boneOffset+2);
        //normal matrix
        boneData[10+3] = texelFetch(boneTBO,boneOffset+3);
        boneData[10+4] = texelFetch(boneTBO,boneOffset+4);
        lastBoneData[2] = texelFetch(boneTBO,boneOffset+5).x;
    }
    if (boneWeightsXYZBoneCountNormalized.w>0.75) //1.0
    {
        boneOffset = gl_InstanceID*BONE_CNT*6 + boneIDs.w*6;
        //global matrix
        boneData[15+0] = texelFetch(boneTBO,boneOffset);
        boneData[15+1] = texelFetch(boneTBO,boneOffset+1);
        boneData[15+2] = texelFetch(boneTBO,boneOffset+2);
        //normal matrix
        boneData[15+3] = texelFetch(boneTBO,boneOffset+3);
        boneData[15+4] = texelFetch(boneTBO,boneOffset+4);
        lastBoneData[3] = texelFetch(boneTBO,boneOffset+5).x;
    }

    //adding transformed weighted vertices is better than adding weighted matrices and then transforming
    //averaging matrices            = [1,4]*(21 fmads) + 15 fmads
    //averaging transformed verts   = [1,4]*(15 fmads + 7 muls)
    skinnedPos = mat4x3(boneData[0],boneData[1],boneData[2])*vec4(vPos*boneWeightsXYZBoneCountNormalized.x,boneWeightsXYZBoneCountNormalized.x);
    skinnedNormal = mat3(boneData[3],boneData[4],lastBoneData[0])*(vNormal*boneWeightsXYZBoneCountNormalized.x);
	// we have a choice, either branch and waste 8 cycles on if-statements, or waste 84 on zeroing out the arrays
    if (boneWeightsXYZBoneCountNormalized.w>0.25) //0.33333
    {
        skinnedPos += mat4x3(boneData[5],boneData[5+1],boneData[5+2])*vec4(vPos*vBoneWeights.y,vBoneWeights.y);
        skinnedNormal += mat3(boneData[5+3],boneData[5+4],lastBoneData[1])*(vNormal*vBoneWeights.y);
    }
    if (boneWeightsXYZBoneCountNormalized.w>0.5) //0.666666
    {
        skinnedPos += mat4x3(boneData[10],boneData[10+1],boneData[10+2])*vec4(vPos*vBoneWeights.z,vBoneWeights.z);
        skinnedNormal += mat3(boneData[10+3],boneData[10+4],lastBoneData[2])*(vNormal*vBoneWeights.z);
    }
    if (boneWeightsXYZBoneCountNormalized.w>0.75) //1.0
    {
        float lastWeight = 1.0-boneWeightsXYZBoneCountNormalized.x-boneWeightsXYZBoneCountNormalized.y-boneWeightsXYZBoneCountNormalized.z;
        skinnedPos += mat4x3(boneData[15],boneData[15+1],boneData[15+2])*vec4(vPos*lastWeight,lastWeight);
        skinnedNormal += mat3(boneData[15+3],boneData[15+4],lastBoneData[3])*(vNormal*lastWeight);
    }
}

void main()
{
    vec3 pos,nml;
    linearSkin(pos,nml,vBoneIDs,vBoneWeights);

    if (gl_InstanceID%2 == 0)
        gl_Position = MVP*(vec4(pos,1.0) + 10.f*vec4(float(gl_InstanceID), 0.f, float(gl_InstanceID), 0.f));
    else
        gl_Position = vec4(0.f, 0.f, 0.f, 1.f);
    Normal = normalize(nml); //have to normalize twice because of normal quantization
}
