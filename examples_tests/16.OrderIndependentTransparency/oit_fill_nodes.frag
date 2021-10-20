#version 450 core

layout (location = 2) in vec3 Normal;
layout (location = 3) in vec2 UV;

layout (set = 3, binding = 5) uniform sampler2D map_d;

#include <nbl/builtin/glsl/ext/OIT/insert_node.glsl>

void main()
{
	const ivec2 coord = ivec2(gl_FragCoord.xy);

    vec4 fragcolor = texture(map_d, UV);
    if (fragcolor.w < 1.0/255.0)
    {
        discard;
    }
	// abs cause BSDF
	//fragcolor.rgb *= abs(dot(normalize(Normal),normalize(vec3(1.f))));
	
	uint mydepth = nbl_glsl_oit_encode_depth(nbl_glsl_oit_get_rev_depth());
	float myvis = 1.0-fragcolor.a;
	uint mycolor = packUnorm4x8(vec4(fragcolor.rgb*fragcolor.a,0.0));

	NBL_GLSL_OIT_CRITICAL_SECTION(nbl_glsl_oit_insert_node(coord, mydepth, myvis, mycolor));
}