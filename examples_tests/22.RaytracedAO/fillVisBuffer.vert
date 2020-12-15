#version 430 core

#include "drawCommon.glsl"
layout(set=1, binding=0, std430, row_major) restrict readonly buffer PerInstanceStatic
{
    ObjectStaticData_t staticData[];
};
layout(set=1, binding=1, row_major) readonly restrict buffer PerInstancePerCamera
{
    DrawData_t data[];
} instanceDataPerCamera;

layout(location = 0) in vec3 vPosition;
layout(location = 2) in vec2 vUV;
layout(location = 3) in vec3 vNormal;

layout(location = 0) flat out uint ObjectID;
layout(location = 1) out vec3 Normal;
layout(location = 2) out vec2 UV;

#include <nbl/builtin/glsl/utils/transform.glsl>

void main()
{
    DrawData_t self = instanceDataPerCamera.data[gl_InstanceIndex];
    ObjectID = self.objectID|(floatBitsToUint(self.detMVP)&0x80000000u); // use MSB to denote if face orientation should be flipped

    gl_Position = nbl_glsl_pseudoMul4x4with3x1(self.MVP,vPosition);
    
    const vec3 localNormal = normalize(vNormal); //have to normalize twice because of normal quantization
    Normal[0] = dot(staticData[self.objectID].normalMatrixRow0,localNormal);
    Normal[1] = dot(staticData[self.objectID].normalMatrixRow1,localNormal);
    Normal[2] = dot(staticData[self.objectID].normalMatrixRow2,localNormal);
	
    UV = vUV;
}
