#include "irr/builtin/shaders/loaders/gltf/common.glsl"

#ifndef _IRR_VERT_INPUTS_DEFINED_
#define _IRR_VERT_INPUTS_DEFINED_

layout (location = _IRR_POSITION_ATTRIBUTE_ID) in vec3 vPos;
layout(location = _IRR_NORMAL_ATTRIBUTE_ID) in vec3 vNormal;

#ifndef _DISABLE_COLOR_ATTRIBUTES
layout(location = _IRR_COLOR_ATTRIBUTE_BEGINING_ID) in vec4 vColor_0;
#ifdef _IRR_EXTRA_COLOR_ATTRIBUTE_ID_1_ENABLE
#define _IRR_EXTRA_COLOR_ATTRIBUTE_ID_1 _IRR_COLOR_ATTRIBUTE_BEGINING_ID + 1
layout(location = _IRR_EXTRA_COLOR_ATTRIBUTE_ID_1) in vec4 vColor_1
#endif // _IRR_EXTRA_COLOR_ATTRIBUTE_ID_1_ENABLE
#ifdef _IRR_EXTRA_COLOR_ATTRIBUTE_ID_2_ENABLE
#define _IRR_EXTRA_COLOR_ATTRIBUTE_ID_2 _IRR_COLOR_ATTRIBUTE_BEGINING_ID + 2
layout(location = _IRR_EXTRA_COLOR_ATTRIBUTE_ID_2) in vec4 vColor_2
#endif // _IRR_EXTRA_UV_ATTRIBUTE_ID_2_ENABLE
#endif // _DISABLE_COLOR_ATTRIBUTES

#ifndef _DISABLE_UV_ATTRIBUTES
layout (location = _IRR_UV_ATTRIBUTE_BEGINING_ID) in vec2 vUV_0;
#ifdef _IRR_EXTRA_UV_ATTRIBUTE_ID_1_ENABLE
#define _IRR_EXTRA_UV_ATTRIBUTE_ID_1 _IRR_UV_ATTRIBUTE_BEGINING_ID + 1
layout (location = _IRR_EXTRA_UV_ATTRIBUTE_ID_1) in vec2 vUV_1
#endif // _IRR_EXTRA_UV_ATTRIBUTE_ID_1_ENABLE
#ifdef _IRR_EXTRA_UV_ATTRIBUTE_ID_2_ENABLE
#define _IRR_EXTRA_UV_ATTRIBUTE_ID_2 _IRR_UV_ATTRIBUTE_BEGINING_ID + 2
layout(location = _IRR_EXTRA_UV_ATTRIBUTE_ID_2) in vec2 vUV_2
#endif // _IRR_EXTRA_UV_ATTRIBUTE_ID_2_ENABLE
#endif // _DISABLE_UV_ATTRIBUTES

#if !defined(_DISABLE_COLOR_ATTRIBUTES) && !defined(_DISABLE_UV_ATTRIBUTES)
#error "ERROR: UV and Color attributes must not be defined both either!"
#endif

#endif //_IRR_VERT_INPUTS_DEFINED_

#ifndef _IRR_VERT_OUTPUTS_DEFINED_
#define _IRR_VERT_OUTPUTS_DEFINED_

/*
    glTF Loader's shaders don't support multiple out attributes
    at the moment, to change in future
*/

layout (location = 0) out vec3 LocalPos;
layout(location = 1) out vec3 Normal;
layout (location = 2) out vec3 ViewPos;
#ifndef _DISABLE_UV_ATTRIBUTES
layout (location = _IRR_UV_ATTRIBUTE_BEGINING_ID) out vec2 UV_0;
#endif // _DISABLE_UV_ATTRIBUTES
#ifndef _DISABLE_COLOR_ATTRIBUTES
layout(location = _IRR_COLOR_ATTRIBUTE_BEGINING_ID) out vec4 Color_0;
#endif // _DISABLE_COLOR_ATTRIBUTES

#endif //_IRR_VERT_OUTPUTS_DEFINED_

#include <irr/builtin/glsl/utils/common.glsl>
#include <irr/builtin/glsl/utils/transform.glsl>

#ifndef _IRR_VERT_SET1_BINDINGS_DEFINED_
#define _IRR_VERT_SET1_BINDINGS_DEFINED_
layout (set = 1, binding = 0, row_major, std140) uniform UBO
{
    irr_glsl_SBasicViewParameters params;
} CameraData;
#endif //_IRR_VERT_SET1_BINDINGS_DEFINED_

#ifndef _IRR_VERT_MAIN_DEFINED_
#define _IRR_VERT_MAIN_DEFINED_
void main()
{
    LocalPos = vPos;
    gl_Position = irr_glsl_pseudoMul4x4with3x1(irr_builtin_glsl_workaround_AMD_broken_row_major_qualifier(CameraData.params.MVP), vPos);
    ViewPos = irr_glsl_pseudoMul3x4with3x1(irr_builtin_glsl_workaround_AMD_broken_row_major_qualifier(CameraData.params.MV), vPos);
    mat3 normalMat = irr_glsl_SBasicViewParameters_GetNormalMat(irr_builtin_glsl_workaround_AMD_broken_row_major_qualifier(CameraData.params.NormalMatAndEyePos));
    Normal = normalMat*normalize(vNormal);

#ifndef _DISABLE_UV_ATTRIBUTES
    UV_0 = vUV_0;
#endif // _DISABLE_UV_ATTRIBUTES

#ifndef _DISABLE_COLOR_ATTRIBUTES
    Color_0 = vColor_0;
#endif // _DISABLE_COLOR_ATTRIBUTES
}
#endif //_IRR_VERT_MAIN_DEFINED_