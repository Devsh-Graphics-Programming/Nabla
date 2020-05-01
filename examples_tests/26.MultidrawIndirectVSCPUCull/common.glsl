struct ModelData_t
{
#ifdef __cplusplus
    core::matrix4SIMD   modelViewProjMatrix;
    core::matrix3x4SIMD normalMat;
    core::vectorSIMDf   bbox[2];
#else
    mat4 MVP;
    mat3 normalMat;
    vec3 bbox;
#endif
};

#ifndef __cplusplus
#include <irr/builtin/glsl/broken_driver_workarounds/amd.glsl>

layout(set = 1, binding = 0, std430, row_major) restrict readonly buffer PerObject
{
    ModelData_t modelData[];
};
#endif