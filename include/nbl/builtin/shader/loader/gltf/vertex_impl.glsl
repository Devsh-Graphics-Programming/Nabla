#include "nbl/builtin/shader/loader/gltf/common.glsl"

#ifndef _NBL_VERT_INPUTS_DEFINED_
#define _NBL_VERT_INPUTS_DEFINED_

layout (location = _NBL_V_IN_POSITION_ATTRIBUTE_ID) in vec3 vPos;
layout(location = _NBL_V_IN_NORMAL_ATTRIBUTE_ID) in vec3 vNormal;

#ifndef _DISABLE_COLOR_ATTRIBUTES
layout(location = _NBL_V_IN_COLOR_ATTRIBUTE_ID) in vec4 vColor_0; //! multiple color attributes aren't supported
#endif // _DISABLE_COLOR_ATTRIBUTES

#ifndef _DISABLE_UV_ATTRIBUTES
layout (location = _NBL_V_IN_UV_ATTRIBUTE_ID) in vec2 vUV_0; //! multiple color attributes aren't supported
#endif // _DISABLE_UV_ATTRIBUTES

#if !defined(_DISABLE_COLOR_ATTRIBUTES) && !defined(_DISABLE_UV_ATTRIBUTES)
#error "ERROR: UV and Color attributes must not be defined both either!"
#endif

#endif //_NBL_VERT_INPUTS_DEFINED_

#ifndef _NBL_VERT_OUTPUTS_DEFINED_
#define _NBL_VERT_OUTPUTS_DEFINED_

/*
    glTF Loader's shaders don't support multiple out attributes
    at the moment, to change in future
*/

layout (location = _NBL_V_OUT_LOCAL_POSITION_ID) out vec3 LocalPos;
layout (location = _NBL_V_OUT_VIEW_POS_ID) out vec3 ViewPos;
#ifndef _DISABLE_UV_ATTRIBUTES
layout (location = _NBL_V_OUT_UV_ID) out vec2 UV_0;
#endif // _DISABLE_UV_ATTRIBUTES
#ifndef _DISABLE_COLOR_ATTRIBUTES
layout(location = _NBL_V_OUT_COLOR_ID) out vec4 Color_0;
#endif // _DISABLE_COLOR_ATTRIBUTES

#endif //_NBL_VERT_OUTPUTS_DEFINED_

#include <nbl/builtin/glsl/utils/common.glsl>
#include <nbl/builtin/glsl/utils/transform.glsl>

#ifndef _NBL_VERT_SET1_BINDINGS_DEFINED_
#define _NBL_VERT_SET1_BINDINGS_DEFINED_
layout (set = 1, binding = 0, row_major, std140) uniform UBO
{
    nbl_glsl_SBasicViewParameters params;
} CameraData;
#endif //_NBL_VERT_SET1_BINDINGS_DEFINED_

#ifndef _NBL_VERT_MAIN_DEFINED_
#define _NBL_VERT_MAIN_DEFINED_
void main()
{
    LocalPos = vPos;
    gl_Position = nbl_glsl_pseudoMul4x4with3x1(CameraData.params.MVP, vPos);
    ViewPos = nbl_glsl_pseudoMul3x4with3x1(CameraData.params.MV, vPos);

#ifndef _DISABLE_UV_ATTRIBUTES
    UV_0 = vUV_0;
#endif // _DISABLE_UV_ATTRIBUTES

#ifndef _DISABLE_COLOR_ATTRIBUTES
    Color_0 = vColor_0;
#endif // _DISABLE_COLOR_ATTRIBUTES
}
#endif //_NBL_VERT_MAIN_DEFINED_