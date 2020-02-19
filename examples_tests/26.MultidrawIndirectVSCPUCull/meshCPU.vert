#version 430 core

#include <irr/builtin/glsl/broken_driver_workarounds/amd.glsl>

struct ModelData_t
{
    mat4 MVP;
    mat3 normalMat;
};

layout(std430, row_major, binding = 0) buffer PerObject
{
    ModelData_t modelData[];
};

uniform uint drawID;


layout(location = 0 ) in vec3 vPos; //only a 3d position is passed from irrlicht, but last (the W) coordinate gets filled with default 1.0
layout(location = 3 ) in vec3 vNormal;

out vec4 Color; //per vertex output color, will be interpolated across the triangle
flat out vec3 Normal;

void main()
{
	mat4 mvp = irr_builtin_glsl_workaround_AMD_broken_row_major_qualifier_mat4(modelData[drawID].MVP);

    gl_Position = mvp[0]*vPos.x+mvp[1]*vPos.y+mvp[2]*vPos.z+mvp[3];
    Color = vec4(0.4,0.4,1.0,1.0);
    Normal = normalize(modelData[drawID].normalMat*vNormal); //have to normalize twice because of normal quantization
}
