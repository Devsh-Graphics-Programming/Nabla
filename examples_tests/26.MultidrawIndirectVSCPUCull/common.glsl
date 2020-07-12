struct ModelData_t
{
#ifdef __cplusplus
    core::matrix3x4SIMD worldMatrix;
    core::matrix3x4SIMD normalMatrix;
    core::vectorSIMDf   bbox[2];
#else
    mat4x3  worldMatrix;
    mat3    normalMatrix;
    vec3    bbox[2];
#endif
};

struct DrawData_t
{
#ifdef __cplusplus
    core::matrix4SIMD   modelViewProjMatrix;
    core::matrix3x4SIMD normalMatrix;
#else
    mat4 modelViewProjMatrix;
    mat3 normalMatrix;
#endif
};

#ifndef __cplusplus
#include <irr/builtin/glsl/utils/vertex.glsl>
#endif