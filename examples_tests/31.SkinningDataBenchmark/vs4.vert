#version 430 core

layout(std140, binding = 0) uniform U { mat4 MVP; };
//uniform mat4 MVP;

layout(location = 0) in vec3 vPos;
layout(location = 3) in vec3 vNormal;
layout(location = 5) in ivec4 vBoneIDs;
layout(location = 6) in vec4 vBoneWeights;

#define BONE_CNT 46
#define INSTANCE_CNT 100
struct PerInstanceData
{
	vec3 gmtx_0[BONE_CNT];
	vec3 gmtx_1[BONE_CNT];
	vec3 gmtx_2[BONE_CNT];
    vec3 gmtx_3[BONE_CNT];
	
	vec3 nmtx_0[BONE_CNT];
	vec3 nmtx_1[BONE_CNT];
	vec3 nmtx_2[BONE_CNT];
};
layout(std430, binding = 0) readonly buffer InstData
{
	//boneCnt*instanceCnt == 46*100
	PerInstanceData data[INSTANCE_CNT];
};

out vec3 Normal;

void linearSkin(out vec3 skinnedPos, out vec3 skinnedNormal, in ivec4 boneIDs, in vec4 boneWeightsXYZBoneCountNormalized)
{
    //adding transformed weighted vertices is better than adding weighted matrices and then transforming
    //averaging matrices            = [1,4]*(21 fmads) + 15 fmads
    //averaging transformed verts   = [1,4]*(15 fmads + 7 muls)
	int off = boneIDs.x;
	mat4x3 g = mat4x3(
		data[gl_InstanceID].gmtx_0[off],
		data[gl_InstanceID].gmtx_1[off],
		data[gl_InstanceID].gmtx_2[off],
        data[gl_InstanceID].gmtx_3[off]
	);
	mat3 n = mat3(
		data[gl_InstanceID].nmtx_0[off],
		data[gl_InstanceID].nmtx_1[off],
		data[gl_InstanceID].nmtx_2[off]
	);
	
    skinnedPos = g*vec4(vPos*boneWeightsXYZBoneCountNormalized.x,boneWeightsXYZBoneCountNormalized.x);
    skinnedNormal = n*(vNormal*boneWeightsXYZBoneCountNormalized.x);
	// we have a choice, either branch and waste 8 cycles on if-statements, or waste 84 on zeroing out the arrays
    if (boneWeightsXYZBoneCountNormalized.w>0.25) //0.33333
    {
		off = boneIDs.y;
		g = mat4x3(
			data[gl_InstanceID].gmtx_0[off],
		    data[gl_InstanceID].gmtx_1[off],
		    data[gl_InstanceID].gmtx_2[off],
            data[gl_InstanceID].gmtx_3[off]
		);
		n = mat3(
	        data[gl_InstanceID].nmtx_0[off],
	        data[gl_InstanceID].nmtx_1[off],
	        data[gl_InstanceID].nmtx_2[off]
		);
	
        skinnedPos += g*vec4(vPos*vBoneWeights.y,vBoneWeights.y);
        skinnedNormal += n*(vNormal*vBoneWeights.y);
    }
    if (boneWeightsXYZBoneCountNormalized.w>0.5) //0.666666
    {
		off = boneIDs.z;
		g = mat4x3(
			data[gl_InstanceID].gmtx_0[off],
		    data[gl_InstanceID].gmtx_1[off],
		    data[gl_InstanceID].gmtx_2[off],
            data[gl_InstanceID].gmtx_3[off]
		);
		n = mat3(
		    data[gl_InstanceID].nmtx_0[off],
		    data[gl_InstanceID].nmtx_1[off],
		    data[gl_InstanceID].nmtx_2[off]
		);
		
        skinnedPos += g*vec4(vPos*vBoneWeights.z,vBoneWeights.z);
        skinnedNormal += n*(vNormal*vBoneWeights.z);
    }
    if (boneWeightsXYZBoneCountNormalized.w>0.75) //1.0
    {
		off = boneIDs.w;
		g = mat4x3(
			data[gl_InstanceID].gmtx_0[off],
		    data[gl_InstanceID].gmtx_1[off],
		    data[gl_InstanceID].gmtx_2[off],
            data[gl_InstanceID].gmtx_3[off]
		);
		n = mat3(
		    data[gl_InstanceID].nmtx_0[off],
		    data[gl_InstanceID].nmtx_1[off],
		    data[gl_InstanceID].nmtx_2[off]
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

    gl_Position = MVP*(vec4(pos,1.0) + 10.f*vec4(float(gl_InstanceID), 0.f, float(gl_InstanceID), 0.f));
    Normal = normalize(nml); //have to normalize twice because of normal quantization
}
