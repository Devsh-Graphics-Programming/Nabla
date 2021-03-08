#include "nbl/builtin/shader/loader/gltf/common.glsl"

#ifndef _NBL_FRAG_INPUTS_DEFINED_
#define _NBL_FRAG_INPUTS_DEFINED_
layout (location = _NBL_F_IN_LOCAL_POSITION_ATTRIBUTE_ID) in vec3 LocalPos;
layout (location = _NBL_F_IN_VIEW_POS_ATTRIBUTE_ID) in vec3 ViewPos;
#ifndef _DISABLE_UV_ATTRIBUTES
layout (location = _NBL_F_IN_UV_ATTRIBUTE_ID) in vec2 UV_0;
#endif
#ifndef _DISABLE_COLOR_ATTRIBUTES
layout (location = _NBL_F_IN_COLOR_ATTRIBUTE_ID) in vec4 Color_0;
#endif // _DISABLE_COLOR_ATTRIBUTES
#endif //_NBL_FRAG_INPUTS_DEFINED_

#ifndef _NBL_FRAG_OUTPUTS_DEFINED_
#define _NBL_FRAG_OUTPUTS_DEFINED_
layout (location = 0) out vec4 OutColor;
#endif //_NBL_FRAG_OUTPUTS_DEFINED_

layout( push_constant, row_major ) uniform Block {
	vec4 baseColorFactor;
	float metallicFactor;
	float roughnessFactor;
	
	vec3 emissiveFactor;
	uint alphaMode;
	float alphaCutoff;
	
	uint availableTextures;
} pushConstants;

#define EGT_BASE_COLOR_TEXTURE_BIT 1
#define EGT_METALLIC_ROUGHNESS_TEXTURE_BIT 2
#define EGT_NORMAL_TEXTURE_BIT 4
#define EGT_OCCLUSION_TEXTURE_BIT 8
#define EGT_EMISSIVE_TEXTURE_BIT 16

#if !defined(_NBL_FRAG_SET3_BINDINGS_DEFINED_) && !defined(_DISABLE_UV_ATTRIBUTES)
#define _NBL_FRAG_SET3_BINDINGS_DEFINED_
layout (set = 3, binding = 0) uniform sampler2D baseColorTexture;
#endif //_NBL_FRAG_SET3_BINDINGS_DEFINED_

// TODO use push constants to determine which textures are in use
layout (set = 3, binding = 1) uniform sampler2D metallicRoughnessTexture;
layout (set = 3, binding = 2) uniform sampler2D normalDerivativeTexture;
layout (set = 3, binding = 3) uniform sampler2D occlusionTexture;
layout (set = 3, binding = 4) uniform sampler2D emissiveTexture;

#ifndef _NBL_FRAG_MAIN_DEFINED_
#define _NBL_FRAG_MAIN_DEFINED_
void main()
{
#ifndef _DISABLE_UV_ATTRIBUTES
	
	const uint baseColorTextureCheck = pushConstants.availableTextures & EGT_BASE_COLOR_TEXTURE_BIT;
	vec4 baseColor;
	
	if(baseColorTextureCheck != 0u)
		baseColor = textureGrad(baseColorTexture, UV_0, UV_0, UV_0); // TODO: partial derivative parameters & UV index
	
	const uint metallicRoughnessTextureCheck = pushConstants.availableTextures & EGT_METALLIC_ROUGHNESS_TEXTURE_BIT;
	vec4 metallicRoughness;
	
	if(metallicRoughnessTextureCheck != 0u)
		metallicRoughness = textureGrad(metallicRoughnessTexture, UV_0, UV_0, UV_0); // TODO: partial derivative parameters & UV index
	
	const uint normalDerivativeTextureCheck = pushConstants.availableTextures & EGT_NORMAL_TEXTURE_BIT;
	vec4 normalDerivative;
	
	if(normalDerivativeTextureCheck != 0u)
		normalDerivative = textureGrad(normalDerivativeTexture, UV_0, UV_0, UV_0); // TODO: partial derivative parameters & UV index
	
	const uint occlusionTextureCheck = pushConstants.availableTextures & EGT_OCCLUSION_TEXTURE_BIT;
	vec4 occlusion;
	
	if(occlusionTextureCheck != 0u)
		occlusion = textureGrad(occlusionTexture, UV_0, UV_0, UV_0); // TODO: partial derivative parameters & UV index
	
	const uint emissiveTextureCheck = pushConstants.availableTextures & EGT_EMISSIVE_TEXTURE_BIT;
	vec4 emissive;
	
	if(emissiveTextureCheck != 0u)
		emissive = textureGrad(emissiveTexture, UV_0, UV_0, UV_0); // TODO: partial derivative parameters & UV index

	OutColor = texture(baseColorTexture, UV_0);
#endif // _DISABLE_UV_ATTRIBUTES

#ifndef _DISABLE_COLOR_ATTRIBUTES
	OutColor = Color_0;
#endif // _DISABLE_COLOR_ATTRIBUTES

#if defined(_DISABLE_COLOR_ATTRIBUTES) && defined(_DISABLE_UV_ATTRIBUTES)
	OutColor = vec4(LocalPos.xyz, 1.0f);
#endif
}
#endif //_NBL_FRAG_MAIN_DEFINED_