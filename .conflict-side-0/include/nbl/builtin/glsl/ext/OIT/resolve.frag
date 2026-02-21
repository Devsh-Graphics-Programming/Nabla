#ifndef _NBL_GLSL_OIT_GLSL_RESOLVE_FRAG_
#define _NBL_GLSL_OIT_GLSL_RESOLVE_FRAG_

#include <nbl/builtin/glsl/ext/OIT/oit.glsl>

layout (location = 0) out vec4 OutColor;

void main()
{
	ivec2 coord = ivec2(gl_FragCoord.xy);
	
	nbl_glsl_oit_color_nodes_t color;
	nbl_glsl_oit_depth_nodes_t depth;
	nbl_glsl_oit_vis_nodes_t vis;
#if NBL_GLSL_OIT_NODE_COUNT==4
    vis = imageLoad(g_vis, coord);
#elif NBL_GLSL_OIT_NODE_COUNT==2
    vis = imageLoad(g_vis, coord).rg;
#endif

	// set all nodes as invalid
	imageStore(g_vis, coord, vec4(1.0));

	// we don't insert nodes with alpha == 0
	if (vis.x==1.0)
		discard;
		
#if NBL_GLSL_OIT_NODE_COUNT==4
    depth = imageLoad(g_depth, coord);
	color = imageLoad(g_color, coord);
#elif NBL_GLSL_OIT_NODE_COUNT==2
    depth = imageLoad(g_depth, coord).rg;
	color = imageLoad(g_color, coord).rg;
#endif

	float v;
	vec3 fragcolor;
	fragcolor = unpackUnorm4x8(color[0]).rgb;
	v = vis[0];
	fragcolor += unpackUnorm4x8(color[1]).rgb*v;
	v *= vis[1];
#if NBL_GLSL_OIT_NODE_COUNT>2
	fragcolor += unpackUnorm4x8(color[2]).rgb*v;
	v *= vis[2];
	fragcolor += unpackUnorm4x8(color[3]).rgb*v;
	v *= vis[3];
#endif

	OutColor = vec4(fragcolor,v);
}

#endif