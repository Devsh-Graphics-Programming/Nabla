#include "common.glsl"
layout(set = 1, binding = 0, std430, row_major) restrict readonly buffer PerDraw
{
    DrawData_t drawData[];
};

layout(location = 0) in vec3 vPos; //only a 3d position is passed from irrlicht, but last (the W) coordinate gets filled with default 1.0
layout(location = 3) in vec3 vNormal;

layout(location = 0) out vec4 Color; //per vertex output color, will be interpolated across the triangle
layout(location = 1) flat out vec3 Normal;

void impl(uint _objectUUID)
{
	mat4 mvp = irr_builtin_glsl_workaround_AMD_broken_row_major_qualifier(drawData[_objectUUID].modelViewProjMatrix);

    gl_Position = mvp[0]*vPos.x+mvp[1]*vPos.y+mvp[2]*vPos.z+mvp[3];
    Color = vec4(0.4,0.4,1.0,1.0);
    Normal = normalize(irr_builtin_glsl_workaround_AMD_broken_row_major_qualifier(drawData[_objectUUID].normalMatrix)*vNormal); //have to normalize twice because of normal quantization
}
