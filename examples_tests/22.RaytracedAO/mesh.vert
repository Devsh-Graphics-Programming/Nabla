#version 430 core

#include "InstanceDataPerCamera.glsl"
layout(set=2, binding=0, row_major) readonly restrict buffer SSBO
{
    InstanceDataPerCamera data[];
} instanceDataPerCamera;

layout(location = 0) in vec3 vPosition;
layout(location = 2) in vec2 vUV;
layout(location = 3) in vec3 vNormal;

layout(location = 0) flat out uint ObjectID;
layout(location = 1) out vec3 Normal;
layout(location = 2) out vec2 UV;

#include "irr/builtin/glsl/utils/transform.glsl"

void main()
{
    ObjectID = gl_InstanceIndex|(floatBitsToUint(NormalMatAndFlags[0].w)&0x80000000u); // use MSB to denote if face orientation should be flipped
    InstanceDataPerCamera self = instanceDataPerCamera.data[ObjectID];

    gl_Position = irr_glsl_pseudoMul4x4with3x1(self.MVP,vPosition);
    Normal = mat3(self.NormalMatAndFlags)*normalize(vNormal); //have to normalize twice because of normal quantization
	UV = vUV;
}
