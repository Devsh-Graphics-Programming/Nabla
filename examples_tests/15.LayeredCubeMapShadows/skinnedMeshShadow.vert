#version 420 core
layout(binding = 3) uniform samplerBuffer tex3;

layout(location = 0) in vec3 vPos; //only a 3d position is passed from irrlicht, but last (the W) coordinate gets filled with default 1.0
layout(location = 5) in ivec4 vBoneIDs;
layout(location = 6) in vec4 vBoneWeights;


void linearSkin(out vec3 skinnedPos, in ivec4 boneIDs, in vec4 boneWeightsXYZBoneCountNormalized)
{
    vec4 boneData[12]; // some vendors don't initialize these arrays to 0 after they've already been written to

    int boneOffset = boneIDs.x*7;
    //global matrix
    boneData[0] = texelFetch(tex3,boneOffset);
    boneData[1] = texelFetch(tex3,boneOffset+1);
    boneData[2] = texelFetch(tex3,boneOffset+2);

    if (boneWeightsXYZBoneCountNormalized.w>0.25) //0.33333
    {
        boneOffset = boneIDs.y*7;
        //global matrix
        boneData[3+0] = texelFetch(tex3,boneOffset);
        boneData[3+1] = texelFetch(tex3,boneOffset+1);
        boneData[3+2] = texelFetch(tex3,boneOffset+2);
    }
    if (boneWeightsXYZBoneCountNormalized.w>0.5) //0.666666
    {
        boneOffset = boneIDs.z*7;
        //global matrix
        boneData[6+0] = texelFetch(tex3,boneOffset);
        boneData[6+1] = texelFetch(tex3,boneOffset+1);
        boneData[6+2] = texelFetch(tex3,boneOffset+2);
    }
    if (boneWeightsXYZBoneCountNormalized.w>0.75) //1.0
    {
        boneOffset = boneIDs.w*7;
        //global matrix
        boneData[9+0] = texelFetch(tex3,boneOffset);
        boneData[9+1] = texelFetch(tex3,boneOffset+1);
        boneData[9+2] = texelFetch(tex3,boneOffset+2);
    }

    skinnedPos = mat4x3(boneData[0],boneData[1],boneData[2])*vec4(vPos*boneWeightsXYZBoneCountNormalized.x,boneWeightsXYZBoneCountNormalized.x);
	// we have a choice, either branch and waste 8 cycles on if-statements, or waste 84 on zeroing out the arrays
    if (boneWeightsXYZBoneCountNormalized.w>0.25) //0.33333
    {
        skinnedPos += mat4x3(boneData[3],boneData[3+1],boneData[3+2])*vec4(vPos*vBoneWeights.y,vBoneWeights.y);
    }
    if (boneWeightsXYZBoneCountNormalized.w>0.5) //0.666666
    {
        skinnedPos += mat4x3(boneData[6],boneData[6+1],boneData[8+2])*vec4(vPos*vBoneWeights.z,vBoneWeights.z);
    }
    if (boneWeightsXYZBoneCountNormalized.w>0.75) //1.0
    {
        float lastWeight = 1.0-boneWeightsXYZBoneCountNormalized.x-boneWeightsXYZBoneCountNormalized.y-boneWeightsXYZBoneCountNormalized.z;
        skinnedPos += mat4x3(boneData[9],boneData[9+1],boneData[9+2])*vec4(vPos*lastWeight,lastWeight);
    }
}

void main()
{
    vec3 pos;
    linearSkin(pos,vBoneIDs,vBoneWeights);

    gl_Position = vec4(pos,1.0);
}

