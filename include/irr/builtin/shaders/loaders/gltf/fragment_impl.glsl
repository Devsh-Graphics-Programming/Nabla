#include "irr/builtin/shaders/loaders/gltf/common.glsl"

#ifndef _IRR_FRAG_INPUTS_DEFINED_
#define _IRR_FRAG_INPUTS_DEFINED_
layout (location = 0) in vec3 LocalPos;
layout(location = 1) in vec3 Normal;
layout (location = 2) in vec3 ViewPos;
#ifndef _DISABLE_UV_ATTRIBUTES
layout (location = _IRR_UV_ATTRIBUTE_BEGINING_ID) in vec2 UV_0;
#endif
#ifndef _DISABLE_COLOR_ATTRIBUTES
layout (location = _IRR_COLOR_ATTRIBUTE_BEGINING_ID) in vec4 Color_0;
#endif // _DISABLE_COLOR_ATTRIBUTES
#endif //_IRR_FRAG_INPUTS_DEFINED_

#ifndef _IRR_FRAG_OUTPUTS_DEFINED_
#define _IRR_FRAG_OUTPUTS_DEFINED_
layout (location = 0) out vec4 OutColor;
#endif //_IRR_FRAG_OUTPUTS_DEFINED_

#if !defined(_IRR_FRAG_SET3_BINDINGS_DEFINED_) && !defined(_DISABLE_UV_ATTRIBUTES)
#define _IRR_FRAG_SET3_BINDINGS_DEFINED_
layout (set = 3, binding = 0) uniform sampler2D texture_0;
#endif //_IRR_FRAG_SET3_BINDINGS_DEFINED_

#ifndef _IRR_FRAG_MAIN_DEFINED_
#define _IRR_FRAG_MAIN_DEFINED_
void main()
{
#ifndef _DISABLE_UV_ATTRIBUTES
	OutColor = texture(texture_0, UV_0);
#endif // _DISABLE_UV_ATTRIBUTES

#ifndef _DISABLE_COLOR_ATTRIBUTES
	OutColor = Color_0;
#endif // _DISABLE_COLOR_ATTRIBUTES

#if defined(_DISABLE_COLOR_ATTRIBUTES) && defined(_DISABLE_UV_ATTRIBUTES)
	OutColor = vec4(LocalPos.xyz, 1.0f);
#endif
}
#endif //_IRR_FRAG_MAIN_DEFINED_