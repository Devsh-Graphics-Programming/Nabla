#include "nbl/builtin/shader/loader/gltf/common.glsl"

#ifndef _NBL_VERT_INPUTS_DEFINED_
#define _NBL_VERT_INPUTS_DEFINED_

layout (location = _NBL_GLTF_POSITION_ATTRIBUTE_ID) in vec3 vPos;
layout(location = _NBL_GLTF_NORMAL_ATTRIBUTE_ID) in vec3 vNormal;

#ifndef _DISABLE_COLOR_ATTRIBUTES
layout(location = _NBL_GLTF_COLOR_ATTRIBUTE_ID) in vec4 vColor_0; //! multiple color attributes aren't supported
#endif // _DISABLE_COLOR_ATTRIBUTES

#ifndef _DISABLE_UV_ATTRIBUTES
layout (location = _NBL_GLTF_UV_ATTRIBUTE_ID) in vec2 vUV_0; //! multiple color attributes aren't supported
#endif // _DISABLE_UV_ATTRIBUTES

#if !defined(_DISABLE_COLOR_ATTRIBUTES) && !defined(_DISABLE_UV_ATTRIBUTES)
#error "ERROR: UV and Color attributes must not be defined both either!"
#endif

#ifdef _NBL_SKINNING_ENABLED_
layout (location = _NBL_GLTF_JOINTS_ATTRIBUTE_ID) in uvec4 vJoints;
layout (location = _NBL_GLTF_WEIGHTS_ATTRIBUTE_ID) in vec3 vWeights;
#include <nbl/builtin/glsl/skinning/linear.glsl>
#endif

#endif //_NBL_VERT_INPUTS_DEFINED_

#ifndef _NBL_VERT_OUTPUTS_DEFINED_
#define _NBL_VERT_OUTPUTS_DEFINED_

/*
    glTF Loader's shaders don't support multiple out attributes
    at the moment, to change in future
*/

layout (location = _NBL_GLTF_LOCAL_POS_INTERPOLANT) out vec3 LocalPos;
layout (location = _NBL_GLTF_NORMAL_INTERPOLANT) out vec3 Normal;
#ifndef _DISABLE_UV_ATTRIBUTES
layout (location = _NBL_GLTF_UV_INTERPOLANT) out vec2 UV_0;
#endif // _DISABLE_UV_ATTRIBUTES
#ifndef _DISABLE_COLOR_ATTRIBUTES
layout(location = _NBL_GLTF_COLOR_INTERPOLANT) out vec4 Color_0;
#endif // _DISABLE_COLOR_ATTRIBUTES

#endif //_NBL_VERT_OUTPUTS_DEFINED_

#include <nbl/builtin/glsl/utils/common.glsl>
#include <nbl/builtin/glsl/utils/transform.glsl>

#ifdef _NBL_SKINNING_ENABLED_
layout (set = 0, binding = 0, row_major, std140) readonly buffer NodeGlobalTransforms
{
    mat4x3 data[];
} nodeGlobalTransforms;

uint nbl_glsl_skinning_translateJointIDtoNodeID(in uint objectID, in uint jointID) // assuming only one instance at the moment! TODO: multi-instances!
{
	return jointID; // @devshgraphicsprogramming it seems to me glTF stores indices in "local" space in vertex inputs stream, not global
}
#endif // _NBL_SKINNING_ENABLED_

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
	Normal = vNormal;
	
#ifdef _NBL_SKINNING_ENABLED_
	nbl_glsl_skinning_linear(LocalPos,Normal,0x45u,vJoints,vWeights);
#endif

	gl_Position = nbl_glsl_pseudoMul4x4with3x1(CameraData.params.MVP,LocalPos);

#ifndef _DISABLE_UV_ATTRIBUTES
    UV_0 = vUV_0;
#endif // _DISABLE_UV_ATTRIBUTES

#ifndef _DISABLE_COLOR_ATTRIBUTES
    Color_0 = vColor_0;
#endif // _DISABLE_COLOR_ATTRIBUTES
}
#endif //_NBL_VERT_MAIN_DEFINED_