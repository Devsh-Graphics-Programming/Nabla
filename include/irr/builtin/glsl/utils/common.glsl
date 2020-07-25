#ifndef _IRR_BUILTIN_GLSL_UTILS_COMMON_INCLUDED_
#define _IRR_BUILTIN_GLSL_UTILS_COMMON_INCLUDED_

#include "irr/builtin/glsl/broken_driver_workarounds/amd.glsl"

#include "irr/builtin/glsl/math/functions.glsl"


struct irr_glsl_SBasicViewParameters
{
    mat4 MVP;
    mat4x3 MV;
    mat4x3 NormalMatAndEyePos;
};

mat3 irr_glsl_SBasicViewParameters_GetNormalMat(in mat4x3 _NormalMatAndEyePos)
{
    return mat3(irr_builtin_glsl_workaround_AMD_broken_row_major_qualifier(_NormalMatAndEyePos));
}
vec3 irr_glsl_SBasicViewParameters_GetEyePos(in mat4x3 _NormalMatAndEyePos)
{
    return irr_builtin_glsl_workaround_AMD_broken_row_major_qualifier(_NormalMatAndEyePos)[3];
}

#endif
